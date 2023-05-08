from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from common.const import CommonConst as cc
from common.const import SaverLoaderConst as slc
from common.utils import update_datacatalog

runner = SequentialRunner()


def common_pipeline(epoch_pipeline, data, preprocessing_pipeline=None, postrocessing_pipeline=None, max_epoch=20):
    if preprocessing_pipeline:
        preprocessing_output = runner.run(preprocessing_pipeline, data)
        loop_data = update_datacatalog(DataCatalog({}), preprocessing_output["results"])
        current_epoch = (
            preprocessing_output["results"]["epoch"] if preprocessing_output["results"]["epoch"] != cc.NONE else 0
        )
    else:
        loop_data = data
        current_epoch = 0
    for i in range(current_epoch + 1, max_epoch + current_epoch + 1):
        loop_data = update_datacatalog(loop_data, {slc.EPOCH: i}, replace=True)
        epoch_output = runner.run(epoch_pipeline, loop_data)
        loop_data = update_datacatalog(loop_data, epoch_output["results"], replace=True)
    if postrocessing_pipeline:
        runner.run(postrocessing_pipeline, loop_data)
