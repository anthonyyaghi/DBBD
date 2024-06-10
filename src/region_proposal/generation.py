from region_proposal.utils import assign_points_to_regions, farthest_point_sampling
import numpy as np
from typing import List, Dict, Tuple, Any


def hierarchical_fps(points: np.ndarray, num_samples_per_level: int, max_levels: int) -> List[Dict[str, Any]]:
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

        sampled_indices = farthest_point_sampling(points, num_samples_per_level)
        sampled_points = points[sampled_indices]
        regions_indices = assign_points_to_regions(points, sampled_points)

        hierarchical_regions = []
        for region_indices in regions_indices:
            region_points = points[region_indices]
            _, sub_regions = recursive_fps(region_points, level + 1)
            hierarchical_regions.append({
                'center': sampled_points,
                'points': region_points,
                'sub_regions': sub_regions
            })

        return sampled_points, hierarchical_regions

    _, hierarchical_regions = recursive_fps(points, 0)
    return hierarchical_regions
