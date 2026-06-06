import torch
import torch.nn as nn

from dataset import data_preprocessing   # ← 模块名按你的文件名改

# 维度选择（论文没钉死 S/C 的大小，这是我们的合理取值）
CAT_EMB_DIM  = 8     # 每个类别列 -> 8 维向量
THER_EMB_DIM = 16    # 治疗 -> 16 维向量
S_DIM, C_DIM = 64, 64

class Encoder(nn.Module):
    """
    2-Layer MLP (256,128), ReLU, BatchNorm, dropout 0.1
    Produce [S,C]
    """
    def __init__(self, num_dim: int, cat_dim: list):   # cat_dim 是每列取值数的列表，如 [3,7,3,10]
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(size, CAT_EMB_DIM) for size in cat_dim])
        din = num_dim + len(cat_dim) * CAT_EMB_DIM     # 18 + 4*8 = 50
        self.net = nn.Sequential(
            nn.Linear(din, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, S_DIM + C_DIM),             # 输出层不加 BN/激活
        )

    def forward(self, num, cat):
        # num: (B, num_dim) 浮点   cat: (B, 列数) 整数索引
        # TODO 1: 把每个类别列过对应 embedding 得到若干 (B,8)，再和 num 在最后一维拼成 x:(B, din)
        #   提示: self.cat_embs[i](cat[:, i]) 得到第 i 列的 (B,8)；用 torch.cat([...], dim=1) 拼接
        x = torch.cat([num] + [self.cat_embs[i](cat[:, i]) for i in range(len(self.cat_embs))], dim=1)
        h = self.net(x)
        S, C = h[:, :S_DIM], h[:, S_DIM:]
        return S, C


class Outcome(nn.Module):
    """f_theta(S, C, t)  ->  预测 y（标量）"""
    def __init__(self, K: int):
        super().__init__()
        self.t_emb = nn.Embedding(K, THER_EMB_DIM)
        din = S_DIM + C_DIM + THER_EMB_DIM             # 64 + 64 + 16 = 144
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, S, C, t):
        te = self.t_emb(t)                  # (B, 16)
        z = torch.cat([S, C, te], dim=1)    # (B, 144)
        return self.net(z)                  # (B, 1)


@torch.no_grad()
def evaluate(enc, out, loader):
    enc.eval(); out.eval()
    se, n = 0.0, 0
    for num, cat, t, y in loader:
        S, C = enc(num, cat)
        pred = out(S, C, t)
        se += ((pred - y) ** 2).sum().item()
        n  += len(y)
    return (se / n) ** 0.5                   # RMSE


def main():
    data = data_preprocessing()
    enc = Encoder(data.num_dim, data.cat_dim)
    out = Outcome(data.K)

    opt = torch.optim.Adam(list(enc.parameters()) + list(out.parameters()),
                           lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val, patience, wait = float("inf"), 15, 0
    for epoch in range(100):
        enc.train(); out.train()
        for num, cat, t, y in data.train_loader:    # 解包顺序固定 (num, cat, t, y)
            S, C = enc(num, cat)
            pred = out(S, C, t)
            loss = loss_fn(pred, y)
            
            # TODO 2: PyTorch 标准训练三步——清梯度、反向求梯度、更新参数
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_rmse = evaluate(enc, out, data.val_loader)
        print(f"epoch {epoch:3d} | val RMSE {val_rmse:.3f}")
        if val_rmse < best_val:
            best_val, wait = val_rmse, 0
        else:
            wait += 1
            if wait >= patience:                     # 早停
                break

    print("test RMSE:", evaluate(enc, out, data.test_loader))


if __name__ == "__main__":
    main()