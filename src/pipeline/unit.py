import torch
from torch import nn


class L2GUnit(nn.Module):
    def __init__(self, encoder, aggregator):
        super(L2GUnit, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator

    def forward(self, x):
        x = self.encoder(x)
        x = self.aggregator(x)
        return x


class G2LUnit(nn.Module):
    def __init__(self, encoder, aggregator, propagator):
        super(G2LUnit, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.propagator = propagator

    def forward(self, x):
        x = self.encoder(x)
        x = self.aggregator(x)
        x = self.propagator(x)
        return x
