from typing import List

from torch import nn


class Branch:
    def __init__(self, units: List[nn.Module]):
        self.units = units

    @property
    def num_layers(self):
        return len(self.units)
