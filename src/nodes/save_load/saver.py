import glob
import torch
import os
from src.common.const import SaverLoaderConst as slc
from src.common.const import MetricConst as mc


def parse_name(name: str):
    params = name.split("_")[1:]
    data = torch.load(name)
    if params:
        result = {}
        for param in params:
            result[param] = {
                slc.PATH: name,
                slc.VALUE: data[param],
            }
        return result
    return {}


def is_better(item1, item2, param):
    return item1 > item2 if param == mc.UP else item1 < item2


def update_best_results(old_data, metrics):
    for key, item in old_data.items():
        if is_better(item[slc.VALUE], metrics[key][slc.VALUE], metrics[key][mc.BEST]):
            old_data[key] = {
                slc.VALUE: metrics[key][slc.VALUE],
                slc.PATH: slc.CURRENT,
            }
    return old_data


def remove_old(old_checkpoints, new_checkpoints):
    checkpoints_for_removing = set(old_checkpoints) - new_checkpoints
    for file in list(checkpoints_for_removing):
        os.remove(file)


def save(experiment_dir_path, new_checkpoints, new_data, params):
    save_manager = {}
    for file in new_checkpoints:
        save_manager[file] = []
    for old_name, item in new_data.items():
        save_manager[item[slc.PATH]].append(old_name)
    for old_name, item in save_manager.items():
        new_name = f"{slc.BEST}_{'_'.join(item)}"
        if old_name != slc.CURRENT:
            os.rename(old_name, new_name)
        else:
            torch.save(params, f"{experiment_dir_path}/{new_name}.pt")
            torch.save(params, f"{experiment_dir_path}/{slc.LAST}.pt")


def first_save(experiment_dir_path, params, metrics):
    name = f"{experiment_dir_path}/{mc.BEST}_{'_'.join(list(metrics.keys()))}.pt"
    last = f"{experiment_dir_path}/{slc.LAST}.pt"
    torch.save(params, name)
    torch.save(params, last)


def save_state_dict(experiment_dir_path, metrics, model, optimizer):
    old_data = {}
    checkpoints = [file for file in glob.glob(experiment_dir_path + "\\*.pt")]
    params = {slc.MODEL_STATE_DICT: model.state_dict(), slc.OPTIMIZER_STATE_DICT: optimizer.state_dict()}
    params.update(metrics)
    if not checkpoints:
        first_save(experiment_dir_path, params, metrics)
    for file in checkpoints:
        old_data.update(parse_name(file))
        new_data = update_best_results(old_data, metrics)
        new_checkpoints = {item[slc.PATH] for key, item in new_data.items()}
        remove_old(checkpoints, new_checkpoints)
        save(experiment_dir_path, new_checkpoints, new_data, params)
    return 1
