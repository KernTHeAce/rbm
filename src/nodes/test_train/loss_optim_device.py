from collections import OrderedDict

import torch

from src.common.const import CommonConst as cc


def get_device(is_cuda: bool = False):
    print(torch.device("cpu") if not is_cuda else torch.device("cuda:0"))
    return torch.device("cpu") if not is_cuda else torch.device("cuda:0")


def get_mse_loss():
    return torch.nn.MSELoss()


def get_cross_entropy_loss():
    return torch.nn.CrossEntropyLoss()


def get_adam_optimizer(model, lr, loaded_optim: OrderedDict = cc.NONE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not loaded_optim == cc.NONE:
        optimizer.load_state_dict(loaded_optim)
    return optimizer
