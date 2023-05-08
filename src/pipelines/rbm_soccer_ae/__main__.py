from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import DataCatalog, MemoryDataSet

from common.pipelines import common_pipeline
from src import DATA_DIR
from src.common.const import SaverLoaderConst as slc
from src.pipelines.rbm_soccer_ae.epoch import epoch_pipeline
from src.pipelines.rbm_soccer_ae.preprocessing import preprocessing_pipeline

data = DataCatalog(
    {
        "soccer_train_dataset": CSVDataSet(filepath=f"{DATA_DIR}/soccer/01_raw/wiscout/train_x_sigm_1221.csv"),
        "soccer_test_dataset": CSVDataSet(filepath=f"{DATA_DIR}/soccer/01_raw/wiscout/test_x_sigm_136.csv"),
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(16),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([39, 34, 29, 24, 19]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(1e-3),
        "experiment_name": MemoryDataSet("test_1"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        "preprocessing": MemoryDataSet(lambda x: x),
    }
)


if __name__ == "__main__":
    common_pipeline(epoch_pipeline, data, preprocessing_pipeline, max_epoch=20)
