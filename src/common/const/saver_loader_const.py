from dataclasses import dataclass


@dataclass(frozen=True)
class SaverLoaderConst:
    PATH: str = "path"
    VALUE: str = "value"
    MODEL: str = "model"

    CURRENT: str = "current"
    BEST: str = "best"
    LAST: str = "last"

    MODEL_STATE_DICT: str = "model_state_dict"
    OPTIMIZER_STATE_DICT: str = "optimizer_state_dict"
