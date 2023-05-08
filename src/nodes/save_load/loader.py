from pathlib import Path

import torch

from src import EXPERIMENTS_DIR
from src.common.const import CommonConst as cc
from src.common.const import SaverLoaderConst as slc


def load_state_dict(experiment_name, checkpoint=slc.LAST, new_experiment: bool = True):
    model_path = Path(EXPERIMENTS_DIR, experiment_name, f"{checkpoint}.pt")
    if not model_path.exists():
        if new_experiment:
            print(f"It is a new experiment and there is no checkpoint {checkpoint}" f"Loader will be mot work")
            return cc.NONE, cc.NONE, cc.NONE, False
        else:
            raise Exception(f"There is no such file: {str(model_path)}")
    else:
        state_dict = torch.load(model_path)
        model = state_dict[slc.MODEL_STATE_DICT] if slc.MODEL_STATE_DICT in state_dict else cc.NONE
        optimizer = state_dict[slc.OPTIMIZER_STATE_DICT] if slc.OPTIMIZER_STATE_DICT in state_dict else cc.NONE
        epoch = state_dict[slc.EPOCH] if slc.EPOCH in state_dict else cc.NONE
        return model, optimizer, epoch, True
