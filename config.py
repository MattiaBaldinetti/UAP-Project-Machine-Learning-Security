from pathlib import Path

SEED = 42
NUM_CLASSES = 10
BATCH_SIZE = 128
NUM_EPOCHS = 20

LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

LR_STEP_SIZE = 10
LR_GAMMA = 0.1

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Root = cartella "code/" (dove stanno questi file .py)
PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints_compare"

# -------------------------
# UAP / Adversarial settings
# -------------------------

# Pixel-space L∞ budget
EPS_PIX = 16 / 255

# UAP optimization
UAP_EPOCHS = 10
STEP_DECAY = 0.8

# Clamped loss
BETA = 10.0

# Targeted attack (None = untargeted)
Y_TARGET = None
