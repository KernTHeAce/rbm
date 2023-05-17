from dataclasses import dataclass


@dataclass(frozen=True)
class RBMInitTypes:
    IN_LAYER_ORDER: str = "in_layer_order"  # 1st batch learn avery layer, then 2nd batch learn every layer, etc.
    IN_DATA_ORDER: str = "in_data_order"  # 1st layer learned by all data, then 2nd layer learned by all data, etc.


@dataclass(frozen=True)
class RBMTypes:
    RBM: str = "rbm"
    NO_RBM: str = "no_rbm"
    RRRBM: str = "rrrbm"  # restricted rule rbm
