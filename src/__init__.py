from pathlib import Path
from src.common.logging import set_project_logging

BASE_DIR = str(Path(__file__).resolve().parent.parent)
set_project_logging("rbm", Path(BASE_DIR, "logs"))
