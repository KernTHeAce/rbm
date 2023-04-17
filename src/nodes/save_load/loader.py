from src.common.const import SaverLoaderConst as slc
from pathlib import Path

from src.common.const import CommonConst as cc

import torch


def load_state_dict(experiment_dir_path, checkpoint=slc.LAST, new_experiment: bool = True):
    model_path = Path(experiment_dir_path, f"{checkpoint}.pt")
    if not model_path.exists():
        if new_experiment:
            print(
                f"It is a new experiment and there is no checkpoint {checkpoint}"
                f"Loader will be mot work"
                )
            return cc.NONE, cc.NONE
        else:
            raise Exception(f"There is no such file: {str(model_path)}")
    else:
        state_dict = torch.load(model_path)
        model = state_dict[slc.MODEL_STATE_DICT] if slc.MODEL_STATE_DICT in state_dict else None
        optimizer = state_dict[slc.OPTIMIZER_STATE_DICT] if slc.OPTIMIZER_STATE_DICT in state_dict else None
        return model, optimizer
