from torch import nn

from .base import BaseModel


class Classifier(BaseModel):
    @staticmethod
    def postprocess_modules(modules):
        modules[-1] = nn.Softmax()
        return modules
