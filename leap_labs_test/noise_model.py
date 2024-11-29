import torch
import torch.nn as nn


class NoiseModel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.model = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    def forward(self):
        return self.model
