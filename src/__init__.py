from pathlib import Path

from src.common.logging import set_project_logging
import torch
torch.manual_seed(0)

DEFAULT_MAX_EPOCH = 20

BASE_DIR = str(Path(__file__).resolve().parent.parent)
EXPERIMENTS_DIR = str(Path(BASE_DIR, "experiments"))
DATA_DIR = str(Path(BASE_DIR, "data"))
MLRUNS_DIR = str(Path(BASE_DIR, "mlruns"))

set_project_logging("rbm", Path(BASE_DIR, "logs"))
