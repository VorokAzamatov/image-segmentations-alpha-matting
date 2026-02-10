import torch

from utils.seed import set_seed



# ======================
# Central config dictionary
# ======================
config = {
    # ========= Runtime =========
    "RUNNING_TRAIN": False,
    "FINETUNE": False,

    # ========= Paths =========
    "DUTS_DATA_PATH": "data/duts-dataset",
    "AIM500_DATA_PATH": "data/AIM-500-dataset",
    "BEST_MODEL_SAVE_DIR": "best_models/train_best_models",
    "FT_BEST_MODEL_SAVE_DIR": "best_models/FT_best_models",
    "METRICS_SAVE_DIR": "metrics/train_metrics",
    "FT_METRICS_SAVE_DIR": "metrics/FT_metrics",
    "BEST_MODEL_LOAD_PATH": "best_models/best_model_38epoch.pt",
    "METRICS_PATH": "metrics/train_metrics/metrics.pt",

    # ========= Hyperparameters =========
    # General
    'SEED': 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # General model parameters
    "IMAGE_SIZE": 512,
    "BATCH_SIZE": 16,
    "BASE_CH": 32,
    "IN_CH": 3,
    "NUM_CL": 1,

    # Data parameters
    "DUTS_SUBSET_SIZE": 2500,
    "AIM_SUBSET_SIZE": None,

    # === Training parameters ===
    # Main training params
    "EPOCHS": 100,
    "LR": 1e-3,

    # Scheduler and early stopping
    "LR_SCHEDULER_PATIENCE": 4,
    "FACTOR": 0.5,
    "EARLYSTOPPING_PATIENCE": 9,
    "MIN_DELTA": 1e-3,

    # === Finetune (FT) parameters ===
    # Main training params
    "FT_EPOCHS": 60,
    "FT_LR": 1e-4,

    "EVERY_N_EP": 5,

    # Scheduler and early stopping
    "FT_LR_SCHEDULER_PATIENCE": 2,
    "FT_FACTOR": 0.6,
    "FT_MINLR": 1e-6,
    "FT_EARLYSTOPPING_PATIENCE": 7,
    "FT_MIN_DELTA": 1e-4,
}

# ======================
# Unpack config into variables
# ======================

# ======================
# Runtime
# ======================
RUNNING_TRAIN = config["RUNNING_TRAIN"]
FINETUNE = config["FINETUNE"]

# ======================
# Paths
# ======================
DUTS_DATA_PATH = config["DUTS_DATA_PATH"]
AIM500_DATA_PATH = config["AIM500_DATA_PATH"]
BEST_MODEL_SAVE_DIR = config["BEST_MODEL_SAVE_DIR"]
FT_BEST_MODEL_SAVE_DIR = config["FT_BEST_MODEL_SAVE_DIR"]
METRICS_SAVE_DIR = config["METRICS_SAVE_DIR"]
FT_METRICS_SAVE_DIR = config["FT_METRICS_SAVE_DIR"]
BEST_MODEL_LOAD_PATH = config["BEST_MODEL_LOAD_PATH"]
METRICS_PATH = config["METRICS_PATH"]

# ======================
# Hyperparameters
# ======================
# General
SEED = config['SEED']
set_seed(SEED)
DEVICE = config["DEVICE"]

# General model parameters
IMAGE_SIZE = config["IMAGE_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
BASE_CH = config["BASE_CH"]
IN_CH = config["IN_CH"]
NUM_CL = config["NUM_CL"]

# Data parameters 
DUTS_SUBSET_SIZE = config["DUTS_SUBSET_SIZE"]
AIM_SUBSET_SIZE = config["AIM_SUBSET_SIZE"]

# === Training parameters ===
# Main training params
EPOCHS = config["EPOCHS"]
LR = config["LR"]

EVERY_N_EP = config['EVERY_N_EP']

# Scheduler and early stopping
LR_SCHEDULER_PATIENCE = config["LR_SCHEDULER_PATIENCE"]
FACTOR = config["FACTOR"]
EARLYSTOPPING_PATIENCE = config["EARLYSTOPPING_PATIENCE"]
MIN_DELTA = config["MIN_DELTA"]

# === Finetune (FT) parameters ===
# Main training params
FT_EPOCHS = config["FT_EPOCHS"]
FT_LR = config["FT_LR"]

# Scheduler and early stopping
FT_LR_SCHEDULER_PATIENCE = config["FT_LR_SCHEDULER_PATIENCE"]
FT_FACTOR = config["FT_FACTOR"]
FT_MINLR = config["FT_MINLR"]
FT_EARLYSTOPPING_PATIENCE = config["FT_EARLYSTOPPING_PATIENCE"]
FT_MIN_DELTA = config["FT_MIN_DELTA"]
