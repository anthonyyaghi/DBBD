from region_proposal.utils import assign_points_to_regions, open3d_fps
import numpy as np
from typing import List, Dict, Tuple, Any


def hierarchical_region_proposal(points: np.ndarray, num_samples_per_level: int, max_levels: int) -> List[Dict[str, Any]]:
    """
    Generate hierarchical spherical regions using FPS.

    :param points: (N, 3) array of points
    :param num_samples_per_level: Number of points to sample at each level
    :param max_levels: Maximum depth of the hierarchy
    :return: Hierarchical regions as a list of dictionaries
    """
    def recursive_fps(points: np.ndarray, level: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if level > max_levels or len(points) <= num_samples_per_level:
            return points, []

        sampled_centers = open3d_fps(points, num_samples_per_level)
        regions_pts_indices = assign_points_to_regions(points, sampled_centers)

        hierarchical_regions = []
        for region_indices in regions_pts_indices:
            region_points = points[region_indices]
            _, sub_regions = recursive_fps(region_points, level + 1)
            hierarchical_regions.append({
                'center': sampled_centers,
                'points': region_points,
                'points_indices': region_indices,
                'sub_regions': sub_regions
            })

        return sampled_centers, hierarchical_regions

    _, hierarchical_regions = recursive_fps(points, 0)
    return hierarchical_regions
