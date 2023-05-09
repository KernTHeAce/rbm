import torch


def data_preprocess(data: list):
    result = None
    for item in data:
        if len(item.shape) == 2:
            for i in item:
                if result is None:
                    result = torch.clone(i)
                else:
                    result = torch.cat((result, i), 0)
        elif len(item.shape) == 1:
            if result is None:
                result = torch.clone(item)
            else:
                result = torch.cat((result, item), 0)

    return result
