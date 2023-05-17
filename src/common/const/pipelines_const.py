from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfigConst:
    DATA: str = "data"
    PREPROCESSING: str = "preprocessing"
    EPOCH: str = "epoch"
    POSTPROCESSING: str = "postprocessing"
