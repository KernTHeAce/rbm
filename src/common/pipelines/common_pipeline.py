from typing import Any, Dict

from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner

from src.common.const import CommonConst as cc
from src.common.const import SaverLoaderConst as slc

runner = SequentialRunner()


def update_datacatalog(datacatalog: DataCatalog, new_data: Dict[str, Any], replace=False):
    for key, value in new_data.items():
        datacatalog.add(key, MemoryDataSet(value, copy_mode="assign"), replace=replace)
    return datacatalog


def common_pipeline(epoch_pipeline, data, preprocessing_pipeline=None, postprocessing_pipeline=None, max_epoch=20):
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
    if postprocessing_pipeline:
        runner.run(postprocessing_pipeline, loop_data)
