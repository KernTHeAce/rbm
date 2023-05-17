from src.common.pipelines import run_experiment
from src.pipelines import parse
from .config import config
from src.pipelines import common_data as cd


if __name__ == "__main__":
    args = parse()
    run_experiment(
        config=config,
        max_epoch=args.max_epoch,
        rbm_epochs=cd.RBM_EPOCHS,
        rbm_types=cd.RBM_TYPES,
        rbm_init_types=cd.RBM_INIT_TYPES,
        prefix="mnist_ae",
        postfix=""
    )
