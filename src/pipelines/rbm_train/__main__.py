from kedro.runner import SequentialRunner

from src.common.utils import update_datacatalog
from src.pipelines.rbm_train.epoch import epoch_data, epoch_pipeline
from src.pipelines.rbm_train.preprocessing import preprocessing_data, preprocessing_pipeline

from src.common.const import SaverLoaderConst as slc

MAX_EPOCH = 300
runner = SequentialRunner()
if __name__ == "__main__":
    preprocessing_output = runner.run(preprocessing_pipeline, preprocessing_data)
    loop_data = update_datacatalog(epoch_data, preprocessing_output["results"])

    for i in range(MAX_EPOCH):
        loop_data = update_datacatalog(loop_data, {slc.EPOCH: i}, replace=True)
        epoch_output = runner.run(epoch_pipeline, loop_data)
        print(f"epoch: {i+1}  {epoch_output['metrics_report']}")
        loop_data = update_datacatalog(loop_data, epoch_output["results"], replace=True)
