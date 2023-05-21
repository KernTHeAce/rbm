from src.common.pipelines.running_tools import run_pipeline
from src.pipelines import parse
from src.pipelines.my_test_pipeline.config import config

if __name__ == "__main__":
    args = parse()
    run_pipeline(config, max_epoch=args.max_epoch)
