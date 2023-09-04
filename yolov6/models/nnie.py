import torch.nn as nn


class NNIE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, im):
        y = self.model(im)
        y = y[0] if isinstance(y, list) else y
        return y
