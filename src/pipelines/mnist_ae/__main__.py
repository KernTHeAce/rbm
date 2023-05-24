from src.common.pipelines import run_rbm_experiment
from src.pipelines import common_data as cd
from src.pipelines import parse

from src.pipelines.mnist_ae.config import config

if __name__ == "__main__":
    # args = parse()
    max_epoch = 100
    for i in range(3):
        run_rbm_experiment(
            config=config,
            max_epoch=max_epoch,
            rbm_epochs=cd.RBM_EPOCHS,
            rbm_types=cd.RBM_TYPES,
            rbm_init_types=cd.RBM_INIT_TYPES,
            prefix="mnist_ae",
            postfix=str(i),
        )
