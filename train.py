import os
import torch
import torch.nn as nn
from copy import deepcopy

from config import (set_seed, EPOCHS, LEARNING_RATE, PATIENCE, WEIGHT_DECAY, LAMBDA_STAB, LAMBDA_MMD, LAMBDA_AC)
from dataset import Dataset, data_preprocessing
from model import (InputProcessing, Encoder, PropensityHead, OutcomeModel, ITEHead,
                   Denoiser, make_schedule, sample_C,
                   augment, MMD, anti_collapse, dr_pseudo_targets)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class EarlyStopping:
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
def evaluate_representation(inp, enc, prop, loader) -> tuple[float, float]: 
    """Evaluate the model performance on the val/test set."""
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


def train_representation(data: Dataset, epochs=EPOCHS):
    """Train the representation (Encoder + PropensityHead) with stability/MMD/anti-collapse losses."""

    # Model Initialization
    K = data.K      # number of treatments
    inp  = InputProcessing(data.num_dim, data.cat_dim).to(device)
    enc  = Encoder(inp.din).to(device)
    prop = PropensityHead(K).to(device)
    parts = [inp, enc, prop]

    # Init Optimizer and Loss Functions
    opt = torch.optim.Adam([p for m in parts for p in m.parameters()],
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    
    es = EarlyStopping(PATIENCE)
    mods = {"inp": inp, "enc": enc, "prop": prop}

    for epoch in range(epochs):
        for m in parts: m.train()                           # Set to train mode
        running = {"stab": 0.0, "prop": 0.0, "mmd": 0.0}    # cumulative losses for logging
        n_batches = 0
        for num, cat, t, _ in data.train_loader:
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
                    LAMBDA_AC * (loss_ac_C + loss_ac_S) + \
                    LAMBDA_MMD * loss_mmd + loss_prop

            # Backpropagation
            opt.zero_grad(); loss.backward(); opt.step()

            # Loss sum
            running["stab"] += loss_stab.item()
            running["mmd"] += loss_mmd.item()
            running["prop"] += loss_prop.item()
            n_batches += 1

        # average losses for train/val
        avg = {k: v / n_batches for k, v in running.items()}
        val_ce, val_acc = evaluate_representation(inp, enc, prop, data.val_loader)

        print(f"[Representation Learning] epoch {epoch:2d} | CE {val_ce:.3f} | acc {val_acc:.3f} "
              f"| stab {avg['stab']:.3f} | prop {avg['prop']:.3f} | MMD {avg['mmd']:.4f}") # type: ignore
        
        if es.step(val_ce, mods):  # Early stopping on val CE
            print(f"Early stopping triggered at epoch {epoch:2d}")
            break

    es.restore(mods)
    return mods["inp"], mods["enc"], mods["prop"]

@torch.no_grad()
def evaluate_outcome(inp, enc, out, loader):
    """Evaluate the outcome prediction RMSE on the val/test set."""
    for m in [inp, enc, out]: m.eval()      # Set to eval mode
    mse = nn.MSELoss(reduction="sum")
    n, mse_sum = 0, 0.0
    for num, cat, t, y in loader:
        S, C = enc(inp(num, cat))           # Enchode C, no aug
        y_pred = out(S, C, t)               # predict outcome
        mse_sum += mse(y, y_pred).item()    # Accumulate MSE
        n += y.shape[0]
    return mse_sum / n                      # Return average MSE

def train_outcome(data, inp, enc, epochs=EPOCHS):
    """Train the outcome model."""
    # Model Initialization
    K = data.K      # number of treatments
    out  = OutcomeModel(K).to(device)
    parts = [inp, enc, out]

    # Init Optimizer and Loss Functions
    opt = torch.optim.Adam(out.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    
    es = EarlyStopping(PATIENCE)
    mods = {"out": out}

    for epoch in range(epochs):
        out.train(); MSE_sum, n_batches = 0.0, 0 # Set to train mode
        for num, cat, t, y in data.train_loader:
            num, cat, t, y = num.to(device), cat.to(device), t.to(device), y.to(device)
            S, C = enc(inp(num, cat))

            y_hat = out(S, C, t)
            # Loss: Regression Loss
            loss = mse(y, y_hat)
            # Backpropagation
            opt.zero_grad(); loss.backward(); opt.step()

            # Loss sum
            MSE_sum += loss.item()
            n_batches += 1

        # average losses for train/val
        val_mse = evaluate_outcome(inp, enc, out, data.val_loader)

        print(f"[Outcome Model] epoch {epoch:2d} | MSE {MSE_sum / n_batches:.4f}")
        
        if es.step(val_mse, mods):  # Early stopping on val MSE
            print(f"Early stopping triggered at epoch {epoch:2d}")
            break

    es.restore(mods)
    return mods["out"]


def main():

    # Init 
    os.makedirs("models", exist_ok=True)
    set_seed()
    data = data_preprocessing()

    # First Stage: Train Encoder + PropensityHead
    inp, enc, prop = train_representation(data)
    # Test the representation learning performance on the test set
    test_ce, test_acc = evaluate_representation(inp, enc, prop, data.test_loader)
    print(f"[Representation Learning] Final Test CE: {test_ce:.3f} | Test Accuracy: {test_acc:.3f}")
    for p in enc.parameters(): p.requires_grad = False      # Freeze Encoder

    # Second Stage: Train OutcomeModel
    out = train_outcome(data, inp, enc)
    test_mse = evaluate_outcome(inp, enc, out, data.test_loader)
    print(f"[Outcome Model] Final Test MSE: {test_mse:.4f}")

    # Third Stage: Train ITEHead
    ITE_head = train_ITE(data, inp, enc, prop, out)

    # Forth Stage: Train Diffusion Model

    # Fifth Stage: 


if __name__ == "__main__":
    main()