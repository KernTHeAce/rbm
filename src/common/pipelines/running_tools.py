from kedro.io import DataCatalog, MemoryDataSet

from src.common.const import PipelineConfigConst as pcc
from src.common.const import RBMTypes as rt
from src.common.pipelines import common_pipeline


def run_pipeline(config, updated_data=None, max_epoch=20):
    data = config[pcc.DATA]
    if updated_data is not None:
        for key, item in updated_data.items():
            data[key] = MemoryDataSet(item, copy_mode="assign")
    data = DataCatalog(data)
    common_pipeline(
        preprocessing_pipeline=config[pcc.PREPROCESSING],
        epoch_pipeline=config[pcc.EPOCH],
        postrocessing_pipeline=config[pcc.POSTPROCESSING],
        data=data,
        max_epoch=max_epoch,
    )


def _get_experiment_name(prefix, new_data, postfix=""):
    data = [str(value) for value in new_data.values()]
    return f"{prefix}_{'_'.join(data)}_{postfix}"


def run_experiment(config, max_epoch, rbm_epochs, rbm_init_types, rbm_types, prefix="", postfix=""):
    for rbm_type in rbm_types:
        if rbm_type == rt.NO_RBM:
            new_config = {"rbm_type": rbm_type}
            new_config["experiment_name"] = _get_experiment_name(prefix, new_config, postfix)
            run_pipeline(config, new_config, max_epoch=max_epoch)
        else:
            for rbm_epoch in rbm_epochs:
                for rbm_init_type in rbm_init_types:
                    new_config = {"rbm_epoch": rbm_epoch, "rbm_init_type": rbm_init_type, "rbm_type": rbm_type}
                    new_config["experiment_name"] = _get_experiment_name(prefix, new_config, postfix)
                    run_pipeline(config, new_config, max_epoch=max_epoch)
