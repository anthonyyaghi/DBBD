from typing import List

from torch import nn


class L2GBranch:
    def __init__(self, units: List[nn.Module]):
        self.units = units


class G2LBranch:
    def __init__(self, units: List[nn.Module]):
        self.units = units
