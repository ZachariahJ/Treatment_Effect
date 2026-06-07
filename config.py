
# Loss Weights
LAMBDA_STAB = 1.0
LAMBDA_MMD = 1.0
LAMBDA_AC = 0.1


# Configurations and Hyperparameters
SEED        = 42
EPOCHS      = 100
BATCH_SIZE  = 16
PATIENCE    = 15

# Dataset Split Ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.1

# Stable Representation Augment
NOISE_STD   = 0.1    # Gaussian noise strength
DROP_P      = 0.1    # Feature dropout

# Model Save Path
MODEL_SAVE_PATH = "model.pt"

# Optimizer Hyperparameters
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5