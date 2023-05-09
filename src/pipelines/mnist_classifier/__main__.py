from kedro.io import DataCatalog, MemoryDataSet
from preprocessing import preprocessing_pipeline

from common.const import SaverLoaderConst as slc
from common.pipelines import common_pipeline
from pipelines.mnist_classifier.epoch import epoch_pipeline
from src import DATA_DIR

data = DataCatalog(
    {
        "mnist_train_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/train"),
        "mnist_test_dataset_path": MemoryDataSet(f"{DATA_DIR}/mnist/test"),
        "train_data_loader": MemoryDataSet(copy_mode="assign"),
        "batch_size": MemoryDataSet(16),
        "shuffle": MemoryDataSet(True),
        "features": MemoryDataSet([28 * 28, 100, 50, 10]),
        "is_cuda": MemoryDataSet(False),
        "lr": MemoryDataSet(1e-3),
        "experiment_name": MemoryDataSet(f"test_2"),
        "checkpoint": MemoryDataSet(slc.LAST),
        "new_experiment": MemoryDataSet(True),
        # "preprocessing": MemoryDataSet(lambda images: images[0].view(images[0].size()[0], -1)),
        "preprocessing": MemoryDataSet(lambda img: img.view(-1, 28*28)),
    }
)


if __name__ == "__main__":
    common_pipeline(epoch_pipeline, data, preprocessing_pipeline, max_epoch=20)
