import torch
import torch.nn as nn
import torch.nn.functional as F


from config import LEARNING_RATE, WEIGHT_DECAY, PATIENCE
from dataset import data_preprocessing

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
        self.mlp = nn.Sequential(
            nn.Linear(din, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1), # Hidden Layer 1, 256
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1), # Hidden Layer 2, 128
            nn.Linear(128, S_DIM + C_DIM), # Output
        )

    def forward(self, x):
        h = self.mlp(x)
        return h[:, :S_DIM], h[:, S_DIM:]   # [S, C]

class PropensityHead(nn.Module):
    """
    1 Layer MLP (64) + BatchNorm + ReLU + dropout 0.1
    Output: Logits over K
    """
    def __init__(self, K: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(C_DIM, 64), 
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, K),
        )

    def forward(self, C):
        return self.mlp(C)  # (B, K)

class OutcomeModel(nn.Module):
    """
    2-Layer MLP(128, 64) + BatchNorm + ReLU + dropout 0.1
    f_theta(S, C, t)
    Output: y, next_visit_HAMD
    """
    def __init__(self, K: int):
        super().__init__()
        self.treatment_embedding = nn.Embedding(K, THER_EMB_DIM) # DIM (K, 16)
        din = S_DIM + C_DIM + THER_EMB_DIM
        self.mlp = nn.Sequential(
            nn.Linear(din, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, S, C, t):
        te = self.treatment_embedding(t)    # (B, 16)
        return self.mlp(torch.cat([S, C, te], dim=1))   # (B, 1)


class ITEHead(nn.Module):
    """
    g(S, C)
    Output: K-dim vector
    """
    def __init__(self, K: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(S_DIM + C_DIM, 128), 
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, K),
        )

    def forward(self, S, C):
        return self.mlp(torch.cat([S, C], dim=1))      # (B, K)

