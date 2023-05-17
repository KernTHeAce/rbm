import argparse

from src.common.const import RBMInitTypes as rit
from src.common.const import RBMTypes as rt
from src.common.pipelines import run_pipeline
from src.pipelines import mnist_ae

print(mnist_ae.__name__)


def get_name(prefix, new_data, postfix=""):
    data = [str(value) for value in new_data.values()]
    return f"{prefix}_{'_'.join(data)}_{postfix}"


RBM_EPOCHS = [1, 10, 20]
RBM_INIT_TYPES = [rit.IN_LAYER_ORDER, rit.IN_DATA_ORDER]
RBM_TYPES = [rt.RBM, rt.NO_RBM]


def run_experiments(max_epoch):
    for rbm_epoch in RBM_EPOCHS:
        for rbm_init_type in RBM_INIT_TYPES:
            for rbm_type in RBM_TYPES:
                new_config = {"rbm_epoch": rbm_epoch, "rbm_init_type": rbm_init_type, "rbm_type": rbm_type}
                new_config["experiment_name"] = get_name("mnist_ae", new_config)
                run_pipeline(mnist_ae.config, new_config, max_epoch=max_epoch)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", help="num of experiment epochs", default=5, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    run_experiments(args.max_epoch)
