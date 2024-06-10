from typing import List

import numpy as np
import open3d as o3d


def open3d_fps(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Perform farthest point sampling on a set of points using Open3D.

    :param points: (N, 3) array of points
    :param num_samples: Number of points to sample
    :return: (num_samples, 3) array of sampled points
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    sampled_pcd = pcd.farthest_point_down_sample(num_samples)
    sampled_points = np.asarray(sampled_pcd.points)
    return sampled_points


def assign_points_to_regions(points: np.ndarray, centers: np.ndarray) -> List[List[int]]:
    """
    Assign each point to the nearest region center.

    :param points: (N, 3) array of points
    :param centers: (M, 3) array of region centers
    :return: List of lists, each containing the indices of points in the corresponding region
    """
    if points.size == 0 or centers.size == 0:
        return [[] for _ in range(centers.shape[0])]

    num_points = points.shape[0]
    num_centers = centers.shape[0]
    regions = [[] for _ in range(num_centers)]

    for i in range(num_points):
        distances = np.linalg.norm(points[i] - centers, axis=1)
        nearest_center = np.argmin(distances)
        regions[nearest_center].append(i)

    return regions


def remove_points_from_point_cloud(pcd: o3d.geometry.PointCloud,
                                   points_to_remove: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Remove specified points from an Open3D point cloud.

    :param pcd: Open3D PointCloud object
    :param points_to_remove: (N, 3) numpy array of points to remove
    :return: Open3D PointCloud object with specified points removed
    """
    # Convert point cloud to numpy array
    pcd_points = np.asarray(pcd.points)

    # Create a mask for points to remove using broadcasting and advanced indexing
    mask = np.ones(len(pcd_points), dtype=bool)

    if len(points_to_remove) > 0:
        # Use numpy broadcasting to create a mask for points to remove
        remove_mask = np.isin(pcd_points, points_to_remove).all(axis=1)
        mask = ~remove_mask

    # Get indices of points to keep
    indices_to_keep = np.where(mask)[0]

    # Select points to keep using select_by_index
    pcd_filtered = pcd.select_by_index(indices_to_keep)

    return pcd_filtered
