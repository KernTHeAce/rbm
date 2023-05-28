from pathlib import Path

import torch

from src.common.logging import set_project_logging

torch.manual_seed(0)

DEFAULT_MAX_EPOCH = 20

BASE_DIR = str(Path(__file__).resolve().parent.parent)
EXPERIMENTS_DIR = str(Path(BASE_DIR, "experiments"))
DATA_DIR = str(Path(BASE_DIR, "data"))
MLRUNS_DIR = str(Path(BASE_DIR, "mlruns"))

set_project_logging("rbm", Path(BASE_DIR, "logs"))
