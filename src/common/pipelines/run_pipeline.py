from kedro.io import DataCatalog, MemoryDataSet

from src.common.const import PipelineConfigConst as pcc
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
