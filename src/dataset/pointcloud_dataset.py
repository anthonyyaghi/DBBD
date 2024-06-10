import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, directory, colored=False):
        """
        Initialize the dataset by listing all .ply files in the specified directory and its subdirectories.
        :param directory: Path to the root directory containing .ply files.
        :param colored: Boolean indicating if the dataset includes color information.
        """
        self.files = []
        self.colored = colored
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.ply'):
                    self.files.append(os.path.join(root, file))

    def __len__(self):
        """
        Return the number of files in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a point cloud from a .ply file and return it as a tensor.
        If colored, also return the colors as a tensor.
        :param idx: Index of the file to load.
        :return: Point cloud data as a tensor, and optionally color data as a tensor.
        """
        pcd = o3d.io.read_point_cloud(self.files[idx])
        points = np.asarray(pcd.points)
        if self.colored:
            colors = np.asarray(pcd.colors)
            return torch.tensor(points, dtype=torch.float32), torch.tensor(colors, dtype=torch.float32)
        else:
            return torch.tensor(points, dtype=torch.float32)
