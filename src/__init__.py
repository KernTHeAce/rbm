from pathlib import Path

import torch

torch.manual_seed(0)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")
ADAPTIVE_LRS = [None, False]
INITIALIZER_EPOCHS = [1, 5, 10]
GRAD_MIN_MAX = (-15, 15)
ADAM_EPOCHS = 50
BATCH_SIZE = 32

BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = str(Path(BASE_DIR, "data"))
MLRUNS_DIR = str(Path(BASE_DIR, "mlruns"))
