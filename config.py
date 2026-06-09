# Dataset Split Ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.1

# Stable Representation Augment
NOISE_STD   = 0.1    # Gaussian noise strength
DROP_P      = 0.1    # Feature dropout

# Stabilization for ITE Estimation
CLIPPING_EPS = 0.05     # Propensity score clipping epsilon
TRIM = 0.05             # Propensity score trimming threshold
WINSORIZATION_QUANTILES = (0.01, 0.99)

# Loss weights
LAMBDA_STAB = 1.0
LAMBDA_MMD = 1.0
LAMBDA_PROP = 1.0
LAMBDA_AC = 0.1



# Configurations and Hyperparameters
SEED: int   = 42
EPOCHS: int = 100
BATCH_SIZE  = 4096
PATIENCE    = 15

# Optimizer Hyperparameters
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5

# Model Save Path
MODEL_SAVE_PATH = "models/"

def set_seed(seed=SEED):
    """Set random seed for reproducibility"""
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed