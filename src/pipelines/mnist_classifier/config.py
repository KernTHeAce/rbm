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
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "model": MemoryDataSet(copy_mode="assign"),
        "initialized_model": MemoryDataSet(copy_mode="assign"),
        "initialized_optimizer": MemoryDataSet(copy_mode="assign"),
        "loaded_model": MemoryDataSet(copy_mode="assign"),
        "loaded_optimizer": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(64),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([28 * 28, 100, 50, 10]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(0.001),
        "experiment_name": MemoryDataSet(f"test_2"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        "preprocessing": MemoryDataSet(lambda img: img.view(-1, 28 * 28)),
    },
}
