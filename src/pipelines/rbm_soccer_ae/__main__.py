from kedro.runner import SequentialRunner

from common.const import CommonConst as cc
from src.common.const import SaverLoaderConst as slc
from src.common.utils import update_datacatalog
from src.pipelines.rbm_soccer_ae.epoch import epoch_data, epoch_pipeline
from src.pipelines.rbm_soccer_ae.preprocessing import preprocessing_data, preprocessing_pipeline

MAX_EPOCH = 20
runner = SequentialRunner()
if __name__ == "__main__":
    preprocessing_output = runner.run(preprocessing_pipeline, preprocessing_data)
    loop_data = update_datacatalog(epoch_data, preprocessing_output["results"])

    current_epoch = (
        preprocessing_output["results"]["epoch"] if preprocessing_output["results"]["epoch"] != cc.NONE else 0
    )

    for i in range(current_epoch + 1, MAX_EPOCH + current_epoch + 1):
        loop_data = update_datacatalog(loop_data, {slc.EPOCH: i}, replace=True)
        epoch_output = runner.run(epoch_pipeline, loop_data)
        loop_data = update_datacatalog(loop_data, epoch_output["results"], replace=True)
