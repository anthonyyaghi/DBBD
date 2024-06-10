from abc import ABC, abstractmethod
import torch
from torch import Tensor


class Propagator(ABC):
    """
    Abstract Base Class for all propagators.
    Propagator objects handle the task of propagating global information
    to each point in a tensor.
    """

    @abstractmethod
    def propagate(self, points: Tensor, global_representation: Tensor) -> Tensor:
        """
        Abstract method to propagate global information to each point.

        :param points: tensor of points 
        :param global_representation: global representation tensor
        :return: updated points tensor with propagated global information
        """
        pass


class AppendPropagator(Propagator):
    def propagate(self, points: Tensor, global_representation: Tensor) -> Tensor:
        """
        Propagate global information by appending global_representation to each point in points tensor.

        :param points: input tensor of points where each point is expected to receive propagated information
        :param global_representation: global representation to be propagated
        :return: updated points tensor with appended global representation
        """
        global_representation = global_representation.repeat(points.shape[0], 1)
        return torch.cat((points, global_representation), dim=1)
