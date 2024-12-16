import torch
import torch.nn as nn
from math import pi as PI
import Angular_Deviation_Diffuser.util as util


def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI



class WrappedSmoothL1Loss(nn.Module):
    def __init__(self, beta=0.1 * PI):
        super().__init__()
        self.beta = beta

    def forward(self, input, target):
        d = util.wrap(target - input)
        cond = d.abs() < self.beta

        return torch.where(cond, 0.5 * d.square() / self.beta, d.abs() - 0.5 * self.beta).mean()