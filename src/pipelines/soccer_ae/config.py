from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import MemoryDataSet

from src import DATA_DIR
from src.common.const import PipelineConfigConst as pcc
from src.common.const import SaverLoaderConst as slc

from .epoch import epoch_pipeline
from .preprocessing import preprocessing_pipeline

config = {
    pcc.PREPROCESSING: preprocessing_pipeline,
    pcc.EPOCH: epoch_pipeline,
    pcc.POSTPROCESSING: None,
    pcc.DATA: {
        "soccer_train_dataset": CSVDataSet(filepath=f"{DATA_DIR}/soccer/01_raw/wiscout/train_x_sigm_1221.csv"),
        "soccer_test_dataset": CSVDataSet(filepath=f"{DATA_DIR}/soccer/01_raw/wiscout/test_x_sigm_136.csv"),
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "model": MemoryDataSet(copy_mode="assign"),
        "initialized_model": MemoryDataSet(copy_mode="assign"),
        "initialized_optimizer": MemoryDataSet(copy_mode="assign"),
        "loaded_model": MemoryDataSet(copy_mode="assign"),
        "loaded_optimizer": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(16),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([39, 34, 29, 24, 19]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(1e-3),
        "experiment_name": MemoryDataSet("test_1"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        "preprocessing": MemoryDataSet(lambda x: x),
    },
}
