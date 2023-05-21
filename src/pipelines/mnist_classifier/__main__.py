from src.common.pipelines import run_rbm_experiment
from src.pipelines import common_data as cd
from src.pipelines import parse
from src.pipelines.mnist_classifier.config import config

if __name__ == "__main__":
    args = parse()
    run_rbm_experiment(
        config=config,
        max_epoch=args.max_epoch,
        rbm_epochs=[1, 2, 3],
        rbm_types=cd.RBM_TYPES,
        rbm_init_types=cd.RBM_INIT_TYPES,
        prefix=args.prefix,
        postfix=args.postfix,
    )
