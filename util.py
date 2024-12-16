import torch
from torch import pi as PI


def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI


