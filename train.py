import torch
import torch.nn as nn

from config import (EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LAMBDA_STAB, LAMBDA_MMD, LAMBDA_AC)
from dataset import Dataset, data_preprocessing
from model import (InputProcessing, Encoder, PropensityHead, OutcomeModel, ITEHead,
                   Denoiser, make_schedule, sample_C,
                   augment, MMD, anti_collapse, dr_pseudo_targets)

def train_representation(data: Dataset, epochs=EPOCHS):
    """Train the representation (Encoder + PropensityHead) with stability/MMD/anti-collapse losses."""
    K = data.K                                      # number of treatments
    inp  = InputProcessing(data.num_dim, data.cat_dim)
    enc  = Encoder(inp.din)
    prop = PropensityHead(K)
    parts = [inp, enc, prop]

    # Init Optimizer and Loss Functions
    opt = torch.optim.Adam([p for m in parts for p in m.parameters()],
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for m in parts: m.train()
        for num, cat, t, y in data.train_loader:
            x = inp(num, cat) # Input vector (B, din)

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

        val_rmse, val_acc = evaluate(inp, enc, prop, data.val_loader)
        print(f"[Representation Learning] epoch {epoch:2d} | RMSE {val_rmse:.3f} | acc {val_acc:.3f} "
              f"| stab {loss_stab.values:.3f} | mmd {loss_mmd.item():.4f}")
    return inp, enc, prop

def main():
    # Preprocess datasheet, creates Train/Val/Test DataLoaders
    data = data_preprocessing()

    # First Stage: Train Encoder + PropensityHead
    inp, enc, prop = train_representation(data)
    for p in enc.parameters(): p.requires_grad = False      # Freeze Encoder

    # Second Stage: Train OutcomeModel
    out = train_outcome(data, inp, enc, prop)

    # Third Stage: Train ITEHead
    ite = train_ITE(data, inp, enc, prop, out)

    # Forth Stage: Train Diffusion Model

    # Fifth Stage: 


if __name__ == "__main__":
    main()