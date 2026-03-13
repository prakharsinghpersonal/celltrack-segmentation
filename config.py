"""
CellTrack Configuration
Hyperparameters, paths, and training settings for microscopy instance segmentation.
"""

import os

# ─── Paths ──────────────────────────────────────────────
DATA_DIR = os.environ.get("CELLTRACK_DATA", "./data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# ─── Model ──────────────────────────────────────────────
NUM_CLASSES = 2  # background + cell
BACKBONE = "resnet50"  # Options: resnet50, resnet101
PRETRAINED = True  # Use ImageNet pretrained backbone
MIN_SIZE = 512  # Minimum image dimension for Mask R-CNN
MAX_SIZE = 1024  # Maximum image dimension

# ─── Training ───────────────────────────────────────────
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"  # Options: cosine, step
LR_STEP_SIZE = 15  # For step scheduler
LR_GAMMA = 0.1  # For step scheduler
NUM_WORKERS = 4
PIN_MEMORY = True

# ─── Augmentation ───────────────────────────────────────
AUG_RANDOM_CROP_SIZE = 256
AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.3
AUG_NOISE_VAR_LIMIT = (10, 50)
AUG_ELASTIC_ALPHA = 120
AUG_ELASTIC_SIGMA = 6

# ─── Preprocessing ──────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
MEDIAN_BLUR_KERNEL = 3
TARGET_SIZE = (512, 512)

# ─── Evaluation ─────────────────────────────────────────
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# ─── Device ─────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
