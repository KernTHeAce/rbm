import torchvision
from kedro.runner import SequentialRunner
from preprocessing import preprocessing_data, preprocessing_pipeline
from torchvision import datasets

from common.const import CommonConst as cc
from common.const import SaverLoaderConst as slc
from common.utils import update_datacatalog
from epoch import epoch_data, epoch_pipeline
from src import DATA_DIR, EXPERIMENTS_DIR

MAX_EPOCH = 20
runner = SequentialRunner()

if __name__ == "__main__":
    datasets.MNIST(root=f"{DATA_DIR}/mnist/test", train=False, download=True, transform=None)
    preprocessing_output = runner.run(preprocessing_pipeline, preprocessing_data)
    # print(preprocessing_output)

    loop_data = update_datacatalog(epoch_data, preprocessing_output["results"])
    #
    current_epoch = (
        preprocessing_output["results"]["epoch"] if preprocessing_output["results"]["epoch"] != cc.NONE else 0
    )
    #
    for i in range(current_epoch + 1, MAX_EPOCH + current_epoch + 1):
        loop_data = update_datacatalog(loop_data, {slc.EPOCH: i}, replace=True)
        epoch_output = runner.run(epoch_pipeline, loop_data)
        loop_data = update_datacatalog(loop_data, epoch_output["results"], replace=True)
