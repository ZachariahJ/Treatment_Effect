import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import data_preprocessing
from config import (NOISE_STD, DROP_P, LAMBDA_STAB, LAMBDA_MMD, LAMBDA_AC, LEARNING_RATE, WEIGHT_DECAY)

# Dimensions
CAT_EMB_DIM  = 8
THER_EMB_DIM = 16
S_DIM, C_DIM = 32,32

class InputProcessing(nn.Module):
    """
    Input processing
    Concatenate numerical and categorical features (Dim 8)
    Output Vector x
    """
    def __init__(self, num_dim: int, cat_dim: list[int]):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(c, CAT_EMB_DIM) for c in cat_dim])
        self.din = num_dim + len(cat_dim) * CAT_EMB_DIM     # Din for Encoder
        # print dimensions for debugging
        # print(f"InputProcessing: num_dim={num_dim}, cat_dim={cat_dim}, din={self.din}")

    def forward(self, num, cat):
        cat_vecs = [self.cat_embs[i](cat[:, i]) for i in range(len(self.cat_embs))]
        return torch.cat([num] + cat_vecs, dim=1)   # (B, din)


class Encoder(nn.Module):
    """
    2-Layer MLP(256, 128) + BatchNorm + ReLU + dropout 0.1
    Output: [S, C]
    """
    def __init__(self, din: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1), # Hidden Layer 1, 256
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1), # Hidden Layer 2, 128
            nn.Linear(128, S_DIM + C_DIM), # Output
        )

    def forward(self, x):
        h = self.net(x)
        return h[:, :S_DIM], h[:, S_DIM:]   # [S, C]

class PropensityHead(nn.Module):
    """
    Output: Logits over K
    
    1 Layer MLP (64) + BatchNorm + ReLU + dropout 0.1
    """
    def __init__(self, K: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )

    def forward(self, C):
        return self.net(C)  # (B, K)

class OutcomeModel(nn.Module):
    """
    f_theta(S, C, t) -> y, next_visit_HAMD

    2-Layer MLP(128, 64) + BatchNorm + ReLU + dropout 0.1
    """
    def __init__(self, K: int):
        super().__init__()
        self.treatment_embedding = nn.Embedding(K, THER_EMB_DIM) # DIM (K, 16)
        din = S_DIM + C_DIM + THER_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, S, C, t):
        te = self.treatment_embedding(t)    # (B, 16)
        return self.net(torch.cat([S, C, te], dim=1))   # (B, 1)


class ITEHead(nn.Module):
    """
    g(S, C) -> K-dim vector
    """
    def __init__(self, K: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S_DIM + C_DIM, 128), 
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, K),
        )

    def forward(self, S, C):
        return self.net(torch.cat([S, C], dim=1))      # (B, K)
    







def augment(x, noise_std=NOISE_STD, drop_p=DROP_P):
    """
    Data augmentation for stable representation learning
    Guassian Noise + Feature Dropout
    """
    gauss = torch.randn_like(x) * noise_std
    mask = (torch.rand_like(x) > drop_p).float()
    return (x + gauss) * mask


def anti_collapse(z, gamma=1.0, eps=1e-4):
    """
    z:(B,d) -> var_loss + cov_loss
    Loss Term for Anti-Collapse:
    Decorrelation & Variance Control
    """
    zc  = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)        # Matrix (d,d)

    # Variance
    diag = torch.diagonal(cov) 
    std  = torch.sqrt(diag + eps)
    var_loss = torch.relu(gamma - std).mean()

    # Covariance
    off = cov - torch.diag(diag)
    cov_loss = (off ** 2).sum() / z.shape[1]

    return var_loss + cov_loss

