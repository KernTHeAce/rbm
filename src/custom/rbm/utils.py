from itertools import product


def generate_combinations(parameters):
    keys = list(parameters.keys())
    values = list(parameters.values())
    combinations = [dict(zip(keys, values_tuple)) for values_tuple in product(*values)]
    return combinations


def get_name_by_params(initializer_type, params):
    if not params:
        return "reference"
    if initializer_type == "rbm":
        if params["adaptive_lr"]:
            return f"rbm_adapt_{params['epochs']}_ssl_{params['semisupervised_learning']}"
        return f"rbm_{params['epochs']}_ssl_{params['semisupervised_learning']}"
    return f"cr_{params['epochs']}"
