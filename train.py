from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from copy import deepcopy

from dataset import Dataset, data_preprocessing
from config import (BATCH_SIZE, LAMBDA_PROP, MODEL_SAVE_PATH, set_seed, CLIPPING_EPS, TRIM, WINSORIZATION_QUANTILES, 
                    EPOCHS, LEARNING_RATE, PATIENCE, WEIGHT_DECAY, 
                    LAMBDA_STAB, LAMBDA_MMD, LAMBDA_AC)
from model import (T_STEPS, InputProcessing, Encoder, PropensityHead, OutcomeModel, 
                   ITEHead, Denoiser, make_schedule, sample_C,
                   augment, MMD, anti_collapse)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
SEED = set_seed()


@dataclass
class Models:
    inp: torch.nn.Module
    enc: torch.nn.Module
    prop: torch.nn.Module
    out: torch.nn.Module
    ite: torch.nn.Module
    denoiser: torch.nn.Module


class EarlyStopping:
    """Early stopping mechanism for training."""
    def __init__(self, patience):
        self.patience, self.best, self.counter, self.best_state = patience, float("inf"), 0, None

    def step(self, metric, modules: dict) -> bool:
        if metric < self.best:
            self.best, self.counter = metric, 0
            self.best_state = {k: deepcopy(m.state_dict()) for k, m in modules.items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, modules: dict):
        for k, m in modules.items():
            m.load_state_dict(self.best_state[k])   # type: ignore


@torch.no_grad()
def evaluate_propensity(inp, enc, prop, loader, device=DEVICE) -> tuple[float, float]: 
    """
    evaluate_propensity(inp, enc, prop, loader) -> (Loss_CE, Accuracy)
    Evaluate the Propensity Head performance on the val/test set.
    """
    for m in [inp, enc, prop]: m.eval()                 # Set to eval mode
    ce_fn = nn.CrossEntropyLoss(reduction="sum")
    n, correct, ce_sum = 0, 0, 0.0
    for num, cat, t, _ in loader:
        num, cat, t = num.to(device), cat.to(device), t.to(device)
        x = inp(num, cat)
        _, C = enc(x)                               # Enchode C, no aug
        logits = prop(C)                            # predict logits
        ce_sum  += ce_fn(logits, t).item()
        correct += (logits.argmax(dim=1) == t).sum().item()
        n += t.size(0)
    return ce_sum / n, correct / n


@torch.no_grad()
def evaluate_outcome(inp, enc, out, loader, device=DEVICE):
    """Evaluate the outcome prediction MSE on the val/test set."""
    for m in [inp, enc, out]: m.eval()      # Set to eval mode
    mse = nn.MSELoss()
    n, mse_sum = 0, 0.0
    for z in loader:
        num, cat, t, y = (b.to(device) for b in z)
        S, C = enc(inp(num, cat))           # Enchode C, no aug
        y_pred = out(S, C, t)               # predict outcome
        mse_sum += mse(y_pred, y).item()    # Accumulate MSE
        n += 1
    return mse_sum / n                      # Return average MSE


@torch.no_grad()
def _compute_dr_targets(inp, enc, prop, out, loader, K, device=DEVICE, eps=CLIPPING_EPS):
    """Frozen nuisances -> doubly-robust pseudo-OUTCOMES Y_dr (with propensity CLIPPING).
       Returns S, C, Y_dr (B,K), and observed-treatment propensity e_t (pre-clip, for TRIMMING).
       Effect-vs-baseline and WINSORIZATION are applied by the caller."""
    for m in (inp, enc, prop, out):
        m.eval()
    S_b, C_b, ydr_b, et_b = [], [], [], []
    for z in loader:
        num, cat, t, y = (b.to(device) for b in z)
        S, C = enc(inp(num, cat))
        e   = torch.softmax(prop(C), dim=1)                     # propensity scores (B, K)
        e_t = e.gather(1, t.view(-1, 1)).squeeze(1)             # pre-clip, for trimming
        e   = e.clamp(eps, 1 - eps)                             # CLIPPING
        mu  = torch.stack([out(S, C, torch.full_like(t, k)).squeeze(1)
                           for k in range(K)], dim=1)            # outcome under each treatment
        Y_dr = mu + F.one_hot(t, K).float() / e * (y.view(-1, 1) - mu)   # DR pseudo-outcome
        S_b.append(S); C_b.append(C); ydr_b.append(Y_dr); et_b.append(e_t)
    return torch.cat(S_b), torch.cat(C_b), torch.cat(ydr_b), torch.cat(et_b)


def _build_ite_data(val_dls: list[DataLoader], inp, enc, 
                   props: list[PropensityHead], outs: list[OutcomeModel], 
                   K, t0, shuffle=True, trim=TRIM, 
                   wins=WINSORIZATION_QUANTILES) -> DataLoader:
    """
    built_ite_data(data, inp, enc) -> DataLoader(S, C, ITE)
    DR pseudo-outcomes -> trimming + winsorization -> ITE-head loader.
    """
    S_all, C_all, ydr_all, et_all = [], [], [], []
    for i in range(len(val_dls)):
        val_dl = val_dls[i]
        prop = props[i]
        out = outs[i]
        S, C, Y_dr, e_t = _compute_dr_targets(inp, enc, prop, out, val_dl, K)
        S_all.append(S); C_all.append(C); ydr_all.append(Y_dr); et_all.append(e_t)

    # Concatenate folds
    S, C   = torch.cat(S_all), torch.cat(C_all)
    Y_dr, e_t = torch.cat(ydr_all), torch.cat(et_all)

    # TRIMMING
    keep = e_t >= trim
    S, C, Y_dr = S[keep], C[keep], Y_dr[keep]

    # WINSORIZATION
    lo = torch.quantile(Y_dr, wins[0], dim=0)
    hi = torch.quantile(Y_dr, wins[1], dim=0)
    Y_dr = torch.minimum(torch.maximum(Y_dr, lo), hi)
    
    # ITE
    eff = Y_dr - Y_dr[:, t0:t0 + 1]
    return DataLoader(TensorDataset(S, C, eff), batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def evaluate_ite(val_dl, ite, device=DEVICE, eps=CLIPPING_EPS, trim=TRIM):
    """Evaluate the ITE head.
       Uses MSE throughout to match the training loss."""
    ite.eval()
    mse = nn.MSELoss()
    n, mse_sum = 0, 0.0
    for z in val_dl:
        S, C, eff = (b.to(device) for b in z)
        pred = ite(S, C)            # predict ITE
        loss = mse(pred, eff)       # regression loss
        mse_sum += loss.item()
        n += 1
    return mse_sum / n


@torch.no_grad()
def evaluate_denoiser(inp, enc, denoiser, loader, alpha_bars, device=DEVICE, seed=SEED):
    """Denoising MSE on val/test (per-element, same scale as the train log).
       Steps & noise are seeded so the metric is comparable across epochs."""
    for m in (inp, enc, denoiser): m.eval()
    T = len(alpha_bars)
    g = torch.Generator(device=device).manual_seed(seed)
    mse = nn.MSELoss()
    n, mse_sum = 0, 0.0
    for num, cat, t, _ in loader:
        num, cat, t = num.to(device), cat.to(device), t.to(device)
        S, C = enc(inp(num, cat))                              # clean C, no aug
        step = torch.randint(0, T, (C.shape[0],), device=device, generator=g)
        eps  = torch.randn(C.shape, device=device, generator=g)
        ab   = alpha_bars[step].unsqueeze(1)                   # (B, 1)
        C_t  = torch.sqrt(ab) * C + torch.sqrt(1 - ab) * eps   # forward diffusion
        eps_hat = denoiser(C_t, step, S, t)
        mse_sum += mse(eps_hat, eps).item(); n += 1
    return mse_sum / n


def train_representation(data: Dataset, epochs=EPOCHS, patience=PATIENCE, device=DEVICE):
    """Train the representation (Encoder + PropensityHead) with stability/MMD/anti-collapse losses."""

    # Model Initialization
    K = data.K      # number of treatments
    inp  = InputProcessing(data.num_dim, data.cat_dim).to(device)
    enc  = Encoder(inp.din).to(device)
    prop = PropensityHead(K).to(device)
    parts = [inp, enc, prop]

    # Init Optimizer and Loss Functions
    p = [p for m in parts for p in m.parameters()]
    opt = torch.optim.Adam(p, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    
    es = EarlyStopping(patience)
    mods = {"inp": inp, "enc": enc, "prop": prop}

    for epoch in range(epochs):
        for m in parts: m.train()                           # Set to train mode
        running = {"stab": 0.0, "prop": 0.0, "mmd": 0.0}    # cumulative losses for logging
        n_batches = 0
        for num, cat, t, _ in data.train_dl:
            num, cat, t = num.to(device), cat.to(device), t.to(device)
            x = inp(num, cat)                               # Input vector (B, din)

            # two augmented views of same input
            S1, C1 = enc(augment(x))
            S2, C2 = enc(augment(x))

            # Loss 1: Stability Loss
            # i) "stability of the stable representation across the two views"
            loss_stab = mse(S1, S2)

            # Loss 2: Treatment Predictability Loss
            # ii) "treatment predictability from the confounding representation via the propensity head"
            # Average CrossEntropybetween C1 & C2
            loss_prop = (ce(prop(C1), t) + ce(prop(C2), t)) / 2

            # Loss 3: MMD Loss
            # iii) "distributional balance of the confounding representation across treatments using a kernel-based discrepancy"
            loss_mmd1  = MMD(C1, t, K)
            loss_mmd2  = MMD(C2, t, K)
            loss_mmd = (loss_mmd1 + loss_mmd2) / 2
            # loss_mmd = torch.tensor(0.0) 

            # Loss 4: Anti-Collapse Loss
            # "Lightweight anti-collapse regularization is included with small weights."
            loss_ac_S = anti_collapse(S1, S2)
            loss_ac_C = anti_collapse(C1, C2)

            # Total Loss
            loss =  LAMBDA_STAB * loss_stab + \
                    LAMBDA_AC   * (loss_ac_C + loss_ac_S) + \
                    LAMBDA_MMD  * loss_mmd + \
                    LAMBDA_PROP * loss_prop

            # Backpropagation
            opt.zero_grad(); loss.backward(); opt.step()

            # Loss sum
            running["stab"] += loss_stab.item()
            running["mmd"] += loss_mmd.item()
            running["prop"] += loss_prop.item()
            n_batches += 1

        # average losses for train/val
        avg = {k: v / n_batches for k, v in running.items()}
        val_ce, val_acc = evaluate_propensity(inp, enc, prop, data.val_dl)

        print(f"[Representation Learning] epoch {epoch+1:2d} | CE {val_ce:.3f} | acc {val_acc:.3f} "
              f"| stab {avg['stab']:.3f} | prop {avg['prop']:.3f} | MMD {avg['mmd']:.4f}") # type: ignore
        
        if es.step(val_ce, mods):  # Early stopping on val CE
            print(f"Early stopping triggered at epoch {epoch+1:2d}")
            break

    es.restore(mods)
    return mods["inp"], mods["enc"], mods["prop"]


def train_outcome(data, inp, enc, epochs=EPOCHS, patience=PATIENCE, device=DEVICE):
    """Train the outcome model."""
    # Model Initialization
    K = data.K      # number of treatments
    out  = OutcomeModel(K).to(device)

    # Init Optimizer and Loss Functions
    opt = torch.optim.Adam(out.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    
    es = EarlyStopping(patience); mods = {"out": out}

    for epoch in range(epochs):
        out.train(); enc.eval();
        mse_sum, n_batches = 0.0, 0 # Set to train mode
        for z in data.train_dl:
            num, cat, t, y = (b.to(device) for b in z)
            with torch.no_grad(): S, C = enc(inp(num, cat))

            y_hat = out(S, C, t)            # predict outcome
            loss = mse(y_hat, y)            # Loss: Regression Loss
            opt.zero_grad(); loss.backward(); opt.step() # Backprop

            # Loss sum
            mse_sum += loss.item()
            n_batches += 1

        # average losses for train/val
        val_mse = evaluate_outcome(inp, enc, out, data.val_dl)

        print(f"[Outcome Model] epoch {epoch+1:2d} | Train MSE {mse_sum / n_batches:.4f} | Val MSE {val_mse:.4f}")
        
        if es.step(val_mse, mods):  # Early stopping on val MSE
            print(f"Early stopping triggered at epoch {epoch+1:2d}")
            break

    es.restore(mods)
    return mods["out"]


def train_nuisances(train_dl, val_dl, inp, enc, K, epochs=EPOCHS, 
                    patience=PATIENCE, device=DEVICE): 
    """
    train_nuisances(inp, enc, tr_dl, va_dl, K) -> (OutcomeModel, PropensityHead)

    Train the nuisance models (PropensityHead and OutcomeModel) w/ early stopping. 
    """

    prop = PropensityHead(K).to(device)
    out  = OutcomeModel(K).to(device)

    p = [p for m in [prop, out] for p in m.parameters()]
    opt = torch.optim.Adam(p, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    es_prop, es_out = EarlyStopping(patience), EarlyStopping(patience)

    for epoch in range(epochs):
        inp.eval(); enc.eval()
        prop.train(); out.train()
        for z in train_dl:
            num, cat, t, y = (b.to(device) for b in z)
            with torch.no_grad(): S, C = enc(inp(num, cat))
            loss = ce(prop(C), t) + mse(out(S, C, t), y)
            opt.zero_grad(); loss.backward(); opt.step()

        val_ce, _ = evaluate_propensity(inp, enc, prop, val_dl)
        val_mse   = evaluate_outcome(inp, enc, out, val_dl)
        stop_p = es_prop.step(val_ce, {"prop": prop})
        stop_o = es_out.step(val_mse, {"out": out})
        print(f"[Nuisance Training] epoch {epoch+1:2d} | Prop Val CE {val_ce:.3f} | Out Val MSE {val_mse:.4f}")
        if stop_p and stop_o: 
            break

    es_prop.restore({"prop": prop})
    es_out.restore({"out": out})
    return prop, out


def train_ite(train_dl, val_dl, K, device=DEVICE, epochs=EPOCHS) -> ITEHead:
    """Train the ITE head on the DR targets."""
    ite = ITEHead(K).to(device)
    opt = torch.optim.Adam(ite.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    es  = EarlyStopping(PATIENCE)

    for epoch in range(epochs):
        ite.train()
        mse_sum, n_batches = 0.0, 0
        for z in train_dl:
            S, C, eff = (b.to(device) for b in z)
            pred = ite(S, C)            # predict ITE
            loss = mse(pred, eff)       # regression loss

            opt.zero_grad(); loss.backward(); opt.step()

            # Loss sum
            mse_sum += loss.item()
            n_batches += 1

        val_mse = evaluate_ite(val_dl, ite)
        print(f"[ITE Head] epoch {epoch+1:2d} | Train MSE {mse_sum / n_batches:.4f} | Val MSE {val_mse:.4f}")
        if es.step(val_mse, {"ite": ite}):
            print(f"ITE early stopping at epoch {epoch+1:2d}.")
            break
    es.restore({"ite": ite})
    return ite


def train_denoiser(data, inp, enc, epochs=EPOCHS, patience=PATIENCE, 
                    device=DEVICE, T=T_STEPS) -> Denoiser:
    """Forth Stage - conditional diffusion in the confounding space.
       Standard DDPM eps-prediction: noise the clean C from the (frozen) encoder,
       condition the denoiser on (step, S, treatment t), regress the added noise."""
    K = data.K
    denoiser = Denoiser(K).to(device)
    schedule = make_schedule(T)
    alpha_bars = schedule[2].to(device)

    opt = torch.optim.Adam(denoiser.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    es  = EarlyStopping(patience); mods = {"denoiser": denoiser}

    for epoch in range(epochs):
        inp.eval(); enc.eval(); denoiser.train()
        mse_sum, n_batches = 0.0, 0
        for z in data.train_dl:
            num, cat, t, _ = (b.to(device) for b in z)
            with torch.no_grad(): S, C = enc(inp(num, cat))     # clean C

            step = torch.randint(0, T, (C.shape[0],), device=device)
            ab   = alpha_bars[step].unsqueeze(1)

            eps  = torch.randn_like(C)                          # Create random noise
            C_t  = torch.sqrt(ab) * C + torch.sqrt(1 - ab) * eps
            eps_hat = denoiser(C_t, step, S, t)                    # predict the added noise
            
            loss = mse(eps_hat, eps)                                 
            opt.zero_grad(); loss.backward(); opt.step()
            mse_sum += loss.item(); n_batches += 1

        val_mse = evaluate_denoiser(inp, enc, denoiser, data.val_dl, alpha_bars)
        print(f"[Diffusion] epoch {epoch+1:2d} | Train MSE {mse_sum / n_batches:.4f} | Val MSE {val_mse:.4f}")
        if es.step(val_mse, mods):
            print(f"Diffusion early stopping at epoch {epoch+1:2d}.")
            break

    es.restore(mods)
    return mods["denoiser"]


def train_all(data: Dataset, path = MODEL_SAVE_PATH, seed=SEED, load_existing=False, device=DEVICE) -> Models:
    
    # Init 
    os.makedirs("models", exist_ok=True)
    set_seed(seed)

    if load_existing:
        inp = InputProcessing(data.num_dim, data.cat_dim).to(device)
        enc = Encoder(inp.din).to(device)
        prop = PropensityHead(data.K).to(device)
        out = OutcomeModel(data.K).to(device)
        ite = ITEHead(data.K).to(device)
        denoiser = Denoiser(data.K).to(device)

        inp.load_state_dict(torch.load(os.path.join(path, "inp.pt")))
        enc.load_state_dict(torch.load(os.path.join(path, "enc.pt")))
        prop.load_state_dict(torch.load(os.path.join(path, "prop.pt")))
        out.load_state_dict(torch.load(os.path.join(path, "out.pt")))
        ite.load_state_dict(torch.load(os.path.join(path, "ite.pt")))
        denoiser.load_state_dict(torch.load(os.path.join(path, "denoiser.pt")))

        return Models(
            inp=inp,
            enc=enc,
            prop=prop,
            out=out,
            ite=ite,
            denoiser=denoiser,
        )

    # First Stage: Train Encoder + PropensityHead
    inp, enc, prop = train_representation(data)
    # Test the representation learning performance on the test set
    for p in enc.parameters(): p.requires_grad = False      # Freeze Encoder

    # Second Stage: Train OutcomeModel
    out = train_outcome(data, inp, enc)

    # Test the outcome model performance on the test set
    # Third Stage: Train ITEHead
    n_prop_A, n_out_A = train_nuisances(data.train_dl_A, data.val_dl, inp, enc, data.K)
    n_prop_B, n_out_B = train_nuisances(data.train_dl_B, data.val_dl, inp, enc, data.K)
    train_data  = _build_ite_data([data.train_dl_B, data.train_dl_A], inp, enc, 
                                 [n_prop_A, n_prop_B], [n_out_A, n_out_B], 
                                 data.K, data.params["t0"], shuffle=True)
    val_data    = _build_ite_data([data.val_dl], inp, enc, [prop], [out], data.K, 
                                 data.params["t0"], shuffle=False)
    ite = train_ite(train_data, val_data, data.K)

    # Forth Stage: Train Diffusion Model
    denoiser = train_denoiser(data, inp, enc)

    # Save the trained models
    torch.save(inp.state_dict(),     os.path.join(path, "inp.pt"))
    torch.save(enc.state_dict(),     os.path.join(path, "enc.pt"))
    torch.save(prop.state_dict(),    os.path.join(path, "prop.pt"))
    torch.save(out.state_dict(),     os.path.join(path, "out.pt"))
    torch.save(ite.state_dict(),     os.path.join(path, "ite.pt"))
    torch.save(denoiser.state_dict(),os.path.join(path, "denoiser.pt"))

    return Models(
        inp=inp,
        enc=enc,
        prop=prop,
        out=out,
        ite=ite,
        denoiser=denoiser,
    )

"""
def main():

    # Init 
    os.makedirs("models", exist_ok=True)
    set_seed()
    data = data_preprocessing()

    # # First Stage: Train Encoder + PropensityHead
    # inp, enc, prop = train_representation(data)
    # # Test the representation learning performance on the test set
    # test_ce, test_acc = evaluate_propensity(inp, enc, prop, data.test_dl)
    # print(f"[Representation Learning] Final Test CE: {test_ce:.3f} | Test Accuracy: {test_acc:.3f}")
    # for p in enc.parameters(): p.requires_grad = False      # Freeze Encoder

    # # Second Stage: Train OutcomeModel
    # out = train_outcome(data, inp, enc)
    # test_mse = evaluate_outcome(inp, enc, out, data.test_dl)
    # print(f"[Outcome Model] Final Test MSE: {test_mse:.4f}")

    
    # Third Stage: Train ITEHead
    # n_prop_A, n_out_A = train_nuisances(data.train_dl_A, data.val_dl, inp, enc, data.K)
    # n_prop_B, n_out_B = train_nuisances(data.train_dl_B, data.val_dl, inp, enc, data.K)
    # train_data  = build_ite_data([data.train_dl_B, data.train_dl_A], inp, enc, 
    #                              [n_prop_A, n_prop_B], [n_out_A, n_out_B], 
    #                              data.K, data.params["t0"], shuffle=True)
    # val_data    = build_ite_data([data.val_dl], inp, enc, [prop], [out], data.K, 
    #                              data.params["t0"], shuffle=False)
    # ite = train_ite(train_data, val_data, data.K)
    
    # TEST
    inp = InputProcessing(data.num_dim, data.cat_dim).to(DEVICE)
    enc = Encoder(inp.din).to(DEVICE)
    prop = PropensityHead(data.K).to(DEVICE)
    out = OutcomeModel(data.K).to(DEVICE)
    # TEST

    # Forth Stage: Train Diffusion Model
    denoiser = train_denoiser(data, inp, enc)

    # test_ce, test_acc = evaluate_propensity(inp, enc, prop, data.test_dl)
    # print(f"[Representation Learning] Final Test CE: {test_ce:.3f} | Test Accuracy: {test_acc:.3f}")
    # test_mse = evaluate_outcome(inp, enc, out, data.test_dl)
    # print(f"[Outcome Model] Final Test MSE: {test_mse:.4f}")


if __name__ == "__main__":
    main()

"""