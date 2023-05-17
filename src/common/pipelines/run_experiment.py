from src.common.pipelines import run_pipeline


def get_experiment_name(prefix, new_data, postfix=""):
    data = [str(value) for value in new_data.values()]
    return f"{prefix}_{'_'.join(data)}_{postfix}"


def run_experiment(config, max_epoch, rbm_epochs, rbm_init_types, rbm_types, prefix="", postfix=""):
    for rbm_epoch in rbm_epochs:
        for rbm_init_type in rbm_init_types:
            for rbm_type in rbm_types:
                new_config = {"rbm_epoch": rbm_epoch, "rbm_init_type": rbm_init_type, "rbm_type": rbm_type}
                new_config["experiment_name"] = get_experiment_name(prefix, new_config, postfix)
                run_pipeline(config, new_config, max_epoch=max_epoch)
