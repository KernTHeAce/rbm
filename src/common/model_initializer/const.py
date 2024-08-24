from dataclasses import dataclass


@dataclass(frozen=True)
class ParserConst:
    LAYER: str = "layer"
    FUNC: str = "func"
    LAYER_FN: str = "layer_fn"
