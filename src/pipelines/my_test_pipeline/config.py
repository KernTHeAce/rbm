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
        "mnist_train_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/train"),
        "mnist_test_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/test"),
        "batch_size": MemoryDataSet(32),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([28 * 28, 500, 300, 100, 10]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(1e-3),
        "experiment_name": MemoryDataSet("my_test_experiment"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        "preprocessing": MemoryDataSet(lambda img: img.view(-1, 28 * 28)),
    },
}
