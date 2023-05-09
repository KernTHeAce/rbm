from epoch import epoch_pipeline
from kedro.io import DataCatalog, MemoryDataSet
from preprocessing import preprocessing_pipeline

from common.const import SaverLoaderConst as slc
from common.pipelines import common_pipeline
from src import DATA_DIR

data = DataCatalog(
    {
        "mnist_train_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/train"),
        "mnist_test_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/test"),
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(16),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([28 * 28, 32]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(1e-3),
        "experiment_name": MemoryDataSet("test_3"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        "preprocessing": MemoryDataSet(lambda images: images[0].view(images[0].size()[0], -1)),
    }
)


if __name__ == "__main__":
    common_pipeline(epoch_pipeline, data, preprocessing_pipeline, max_epoch=20)