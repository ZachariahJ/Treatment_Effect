from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from config import BATCH_SIZE, SEED, TRAIN_SPLIT, VAL_SPLIT

UNIQUE_ID = "UNIQUEID"
ORDER_BY = "RAW_ID"
THERAPY_COL = "THERAPY"
HAMDS_COL = [f"HAMD{i:02d}" for i in range(1, 18)]
NUM_COLS = ["AGE"] + HAMDS_COL
CAT_COLS = ["PROTOCOL", "ORIGIN", "GENDER", "GEOCODE"]


@dataclass
class Dataset:
    train_dl    : DataLoader
    train_dl_A  : DataLoader    # For 2-fold cross-fitting
    train_dl_B  : DataLoader    # For 2-fold cross-fitting
    val_dl      : DataLoader
    test_dl     : DataLoader
    K       : int
    num_dim : int
    cat_dim : list
    params  : dict
    inverse_therapy_map: dict[int, str]


def _fit_preprocessor(train_df):
    """
    Fit the preprocessor parameters using the training data.
    """
    return {
        # Numerical Features
        "mean": train_df[NUM_COLS].mean(),
        "std": train_df[NUM_COLS].std().replace(0, 1), # Avoid division by zero
        "cat_maps": {c: {v: i + 1 for i, v in enumerate(train_df[c].dropna().unique())} for c in CAT_COLS}, # '0' UNKNOWN
        "therapy_map": {v: i for i, v in enumerate(sorted(train_df[THERAPY_COL].unique()))},
        "t0": train_df[THERAPY_COL].value_counts().idxmax(), # The most frequent t in training set as baseline (t0)
        "y_mean": train_df["y"].mean(),
        "y_std": train_df["y"].std(),
    }


def _transform(d, p):
    """
    Transform the data using the preprocessor parameters.
    """
    num = ((d[NUM_COLS] - p["mean"]) / p["std"]).fillna(0).astype("float32").values
    cat = np.stack([d[c].map(p["cat_maps"][c]).fillna(0).astype("int64").values for c in CAT_COLS], axis=1)
    t   = d[THERAPY_COL].map(p["therapy_map"]).fillna(0).astype("int64").values
    # y   = d["y"].values.astype("float32")
    y   = ((d["y"] - p["y_mean"]) / p["y_std"]).fillna(0).astype("float32").values
    return num, cat, t, y


def data_preprocessing( csv_path: str = "data_generated.csv",
                        seed: int = SEED,
                        train_split: float = TRAIN_SPLIT,
                        val_split: float = VAL_SPLIT,
                        batch_size: int = BATCH_SIZE
                        ) -> Dataset:
    # Read file as Data Frame
    df: pd.DataFrame = pd.read_csv(csv_path)

    # Calc next_visit_outcome for EACH PATIENT
    df = df.sort_values([UNIQUE_ID, ORDER_BY]).reset_index(drop=True)
    df["HAMD_total"] = df[HAMDS_COL].sum(axis=1).astype("float32")

    df["y"] = df.groupby(UNIQUE_ID)["HAMD_total"].shift(-1)
    df = df.dropna(subset=["y"]).reset_index(drop=True) # Drop last visit

    # Get unique_ids and shuffle
    rng = np.random.default_rng(seed)
    unique_ids = rng.permutation(df[UNIQUE_ID].unique())

    # Split Dataset by patient
    n_unique_ids = len(unique_ids)
    n_train = int(n_unique_ids * train_split)
    n_val   = int(n_unique_ids * val_split)
    n_test  = n_unique_ids - n_train - n_val

    train_ids = unique_ids[:n_train]
    val_ids   = unique_ids[n_train:n_train+n_val]
    test_ids  = unique_ids[n_train+n_val:n_train+n_val+n_test]

    # For 2-fold cross-fitting
    n_two_fold  = n_train // 2
    train_A_ids = unique_ids[:n_two_fold]
    train_B_ids = unique_ids[n_two_fold:n_train]

    # Get indices for train, val, and test splits
    u = df[UNIQUE_ID].values
    train_idx   = np.where(np.isin(u, train_ids))[0]  # type: ignore
    val_idx     = np.where(np.isin(u, val_ids))[0]    # type: ignore
    test_idx    = np.where(np.isin(u, test_ids))[0]   # type: ignore
    train_A_idx = np.where(np.isin(u, train_A_ids))[0]  # type: ignore
    train_B_idx = np.where(np.isin(u, train_B_ids))[0]  # type: ignore

    print(f"### SPLIT DONE ###")
    print(f"Total Patients: {n_unique_ids}")
    print(f"Train/Val/Test Patients: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    # Extract Standardization from TRAINING DATA ONLY
    params = _fit_preprocessor(df.iloc[train_idx])
    num, cat, t, y = _transform(df, params)
    
    # Add inverse map to params 
    params["inverse_therapy_map"] = {i: v for v, i in params["therapy_map"].items()}
    params["t0"] = params["therapy_map"][params["t0"]]

    # Conv to Tensors
    full = TensorDataset(torch.from_numpy(num.copy()),
                         torch.from_numpy(cat.copy()),
                         torch.from_numpy(t.copy()),
                         torch.from_numpy(y.copy()).view(-1, 1))

    def make_dl(idx, shuffle):
        return DataLoader(Subset(full, idx), batch_size=batch_size, shuffle=shuffle)

    return Dataset(
        train_dl = make_dl(train_idx, shuffle=True),
        train_dl_A = make_dl(train_A_idx, shuffle=True),
        train_dl_B = make_dl(train_B_idx, shuffle=True),
        val_dl = make_dl(val_idx, shuffle=False),
        test_dl = make_dl(test_idx, shuffle=False),
        K = len(params["therapy_map"]),
        num_dim = len(NUM_COLS),
        cat_dim = [len(params["cat_maps"][c]) + 1 for c in CAT_COLS], # +1 for unknown category (0)
        inverse_therapy_map = params["inverse_therapy_map"],
        params = params
    )

if __name__ == "__main__":
    data = data_preprocessing()
    # Print Dataset Summary
    print(f"### DATASET SUMMARY ###")
    print(f"Numerical Feature Dimension: {data.num_dim}")
    print(f"Categorical Feature Dimensions: {data.cat_dim}")
    # print the inverse therapy map in separate lines for better readability
    print(f"Number of Therapies (K): {data.K}")
    print(f"Max Therapy Value in Training(t0): {data.params['t0']}")
