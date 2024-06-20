from abc import ABC, abstractmethod
import numpy as np

class PointCloudTransformer(ABC):
    """
    Abstract base class for point cloud transformations.
    """
    
    @abstractmethod
    def transform(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Apply a transformation to a point cloud and return the transformed point cloud.
        
        :param point_cloud: Original point cloud as a NumPy array.
        :return: Transformed point cloud as a NumPy array.
        """
        pass

    @abstractmethod
    def get_mapping(self) -> np.ndarray:
        """
        Return a mapping from the indices of the transformed points back to the indices of the original points.
        
        :return: Array of indices mapping transformed points to original points.
        """
        pass

class RotateTransformer(PointCloudTransformer):
    def __init__(self, rotation_matrix: np.ndarray):
        self.rotation_matrix = rotation_matrix
        self.mapping = None

    def transform(self, point_cloud: np.ndarray) -> np.ndarray:
        self.mapping = np.arange(point_cloud.shape[0])  # Identity mapping since rotation doesn't change point order
        return point_cloud @ self.rotation_matrix.T

    def get_mapping(self) -> np.ndarray:
        return self.mapping

class TranslateTransformer(PointCloudTransformer):
    def __init__(self, translation_vector: np.ndarray):
        self.translation_vector = translation_vector
        self.mapping = None

    def transform(self, point_cloud: np.ndarray) -> np.ndarray:
        self.mapping = np.arange(point_cloud.shape[0])  # Identity mapping since translation doesn't change point order
        return point_cloud + self.translation_vector

    def get_mapping(self) -> np.ndarray:
        return self.mapping
