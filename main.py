import os

import pandas as pd
import torch

from train import Models, DEVICE, train_all
from model import make_schedule, sample_C
from dataset import Dataset, data_preprocessing
from config import set_seed

COL_GEN = "pi_beta(S, Ccf, tcf)"   # generated counterfactual (diffusion)
COL_FIX = "pi_beta(S, C, tcf)"     # treatment-only counterfactual (C fixed)

RESULTS_DIR = "results"


@torch.no_grad()
def run_inference(models: Models, sample_dl, K, therapy_mapping, 
                  n_mc: int = 10, n_show: int = 5, device=DEVICE
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs inference on the given dataset 
    """
    inp, enc, out, denoiser = models.inp, models.enc, models.out, models.denoiser
    for m in (inp, enc, out, denoiser):
        m.eval()
    schedule = make_schedule()

    # ------------------------- single compute pass ------------------------- #
    t_obs_l, y_l, gen_l, fix_l = [], [], [], []
    for z in sample_dl:
        num, cat, t, y = [b.to(device) for b in z]
        S, C = enc(inp(num, cat))
        B = S.shape[0]
        pi_gen = torch.empty(B, K, device=device)
        pi_fix = torch.empty(B, K, device=device)
        for k in range(K):
            t_k = torch.full_like(t, k)
            mc = torch.stack([
                out(S, sample_C(denoiser, S, t_k, schedule), t_k).squeeze(1)
                for _ in range(n_mc)
            ])
            pi_gen[:, k] = mc.mean(dim=0)
            pi_fix[:, k] = out(S, C, t_k).squeeze(1)
        t_obs_l.append(t)
        y_l.append(y.squeeze(1))
        gen_l.append(pi_gen)
        fix_l.append(pi_fix)

    t_obs  = torch.cat(t_obs_l)
    y      = torch.cat(y_l)
    pi_gen = torch.cat(gen_l)
    pi_fix = torch.cat(fix_l)
    N = len(t_obs)
    y_hat = pi_fix[torch.arange(N), t_obs]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------- (i) factual ----------------------------- #
    df_fact = pd.DataFrame({
        "#":     range(N),
        "t_obs": [therapy_mapping[int(k)] for k in t_obs],
        "y_hat": y_hat.cpu().numpy(),
        "y":     y.cpu().numpy(),
    })
    df_fact["SE"] = (df_fact["y_hat"] - df_fact["y"]) ** 2
    fact_csv = os.path.join(RESULTS_DIR, "factual.csv")
    df_fact.to_csv(fact_csv, index=False)

    mse = df_fact["SE"].mean()
    W = 60
    print("\n" + "=" * W)
    print(f"(i) FACTUAL PREDICTION  |  test samples: {N}")
    print("=" * W)
    print(df_fact.head(n_show).to_string(index=False, float_format="%.2f"))
    if N > n_show:
        print(f"... ({N - n_show} more rows)")
    print("-" * W)
    print(f"MSE: {mse:.4f}   RMSE: {mse ** 0.5:.4f}")
    print(f"saved -> {fact_csv}")
    print("=" * W)

    # ------------------------- (ii) counterfactual ------------------------ #
    rows = []
    for i in range(N):
        for k in range(K):
            if k == int(t_obs[i]):                      # keep only tcf != t_obs
                continue
            rows.append({
                "#":     i,
                "t_obs": therapy_mapping[int(t_obs[i])],
                "tcf":   therapy_mapping[k],
                COL_GEN: float(pi_gen[i, k]),
                COL_FIX: float(pi_fix[i, k]),
            })
    df_cf = pd.DataFrame(rows)
    cf_csv = os.path.join(RESULTS_DIR, "counterfactual.csv")
    df_cf.to_csv(cf_csv, index=False)

    W = 80
    print("\n" + "=" * W)
    print(f"(ii) COUNTERFACTUAL PREDICTIONS  |  patients: {N}  |  "
          f"K = {K}  ->  {K - 1} tcf per patient  |  n_mc = {n_mc}")
    print("=" * W)
    shown = df_cf[df_cf["#"] < n_show].set_index(["#", "t_obs"])
    print(shown.to_string(float_format="%.2f"))
    if N > n_show:
        print(f"... ({N - n_show} more patients)")
    print("-" * W)
    print(f"mean | {COL_GEN}: {df_cf[COL_GEN].mean():.2f} | "
          f"{COL_FIX}: {df_cf[COL_FIX].mean():.2f}")
    print(f"mean |gen - fixed| (diffusion contribution): "
          f"{(df_cf[COL_GEN] - df_cf[COL_FIX]).abs().mean():.3f}")
    print(f"saved -> {cf_csv}")
    print("=" * W + "\n")

    return df_fact, df_cf


def main():
    set_seed()
    data = data_preprocessing()
    models = train_all(data, load_existing=True)

    run_inference(models, data.test_dl, K=data.K, 
                  therapy_mapping=data.inverse_therapy_map)


if __name__ == "__main__":
    main()