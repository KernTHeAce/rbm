from dataclasses import dataclass


@dataclass
class RBMType:
    ADAPT_LR: str = "adaptive_lr"
    CONST_LR: str = "const_lr"
    NO_RBM: str = "no_pretraining"


@dataclass(frozen=True)
class ParserConst:
    LAYER: str = "layer"
    FUNC: str = "func"
    LAYER_FN: str = "layer_fn"
