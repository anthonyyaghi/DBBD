from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn as nn


class Aggregator(ABC):

    @abstractmethod
    def aggregate(self, tensor: Tensor) -> Tensor:
        pass


class MaxPoolAggregator(Aggregator):

    def __init__(self):
        self.pool = nn.AdaptiveMaxPool1d(1)

    def aggregate(self, tensor: Tensor) -> Tensor:
        """
        Aggregate tensor data by performing max pooling operation.

        :param tensor: input tensor data to be aggregated
        :return: tensor data after performing a max pooling operation
        """
        return self.pool(tensor)


class AvgPoolAggregator(Aggregator):
    def __init__(self):
        self.pool = nn.AdaptiveAvgPool1d(1)

    def aggregate(self, tensor: Tensor) -> Tensor:
        """
        Aggregate tensor data by performing average pooling operation.

        :param tensor: input tensor data to be aggregated
        :return: tensor data after performing an average pooling operation
        """
        return self.pool(tensor)
