from pathlib import Path

from src.common.logging import set_project_logging

BASE_DIR = str(Path(__file__).resolve().parent.parent)
EXPERIMENTS_DIR = str(Path(BASE_DIR, "experiments"))
DATA_DIR = str(Path(BASE_DIR, "data"))

set_project_logging("rbm", Path(BASE_DIR, "logs"))
