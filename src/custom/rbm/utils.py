from itertools import product


def generate_combinations(parameters):
    keys = list(parameters.keys())
    values = list(parameters.values())
    combinations = [dict(zip(keys, values_tuple)) for values_tuple in product(*values)]
    return combinations


def get_name_by_params(params):
    if params["adaptive_lr"] is None:
        return "reference"
    if params["adaptive_lr"]:
        return f"rbm_adapt_{params['epochs']}_{params['grad_clipping']}"
    return f"rbm_{params['epochs']}"
