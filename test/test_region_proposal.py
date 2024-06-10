import unittest
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from region_proposal.generation import hierarchical_region_proposal
from region_proposal.utils import remove_points_from_point_cloud


class TestHierarchicalFPS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load sample point cloud from open3d.data
        pcd_data = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(pcd_data.path)
        cls.points = np.asarray(pcd.points)
        cls.visualize = False

    def test_hierarchical_fps_basic(self):
        num_samples_per_level = 10
        max_levels = 2
        regions = hierarchical_region_proposal(self.points, num_samples_per_level, max_levels)

        self.assertIsInstance(regions, list)
        self.assertGreater(len(regions), 0)
        for region in regions:
            self.assertIn('center', region)
            self.assertIn('points', region)
            self.assertIn('points_indices', region)
            self.assertIn('sub_regions', region)

    def test_hierarchical_fps_edge_cases(self):
        num_samples_per_level = 1000
        max_levels = 5
        regions = hierarchical_region_proposal(self.points, num_samples_per_level, max_levels)

        self.assertIsInstance(regions, list)
        self.assertGreater(len(regions), 0)
        for region in regions:
            self.assertIn('center', region)
            self.assertIn('points', region)
            self.assertIn('points_indices', region)
            self.assertIn('sub_regions', region)

    def test_visualize_hierarchical_fps(self):
        if self.visualize:
            num_samples_per_level = 4
            max_levels = 3
            regions = hierarchical_region_proposal(self.points, num_samples_per_level, max_levels)
            self.visualize_regions_step_by_step(self.points, regions)

    @staticmethod
    def visualize_regions_step_by_step(points: np.ndarray, regions: list):
        def vis_regions(regions: list):
            if len(regions) == 0:
                return
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            cmap = plt.get_cmap('tab20')
            colors = [cmap(i)[:3] for i in range(len(regions))]

            pc_regions = []

            for i, region in enumerate(regions):
                color = colors[i % len(colors)]
                region_pcd = o3d.geometry.PointCloud()
                region_pcd.points = o3d.utility.Vector3dVector(region['points'])
                region_pcd.paint_uniform_color(color)
                pc_regions.append(region_pcd)
                # Remove the points we already colored
                pcd = remove_points_from_point_cloud(pcd, region['points'])
            # Color the rest of the points in black
            pcd.paint_uniform_color((0, 0, 0))
            pc_regions.append(pcd)
            o3d.visualization.draw_geometries(pc_regions)
            vis_regions(regions[0]['sub_regions'])

        vis_regions(regions)


if __name__ == '__main__':
    unittest.main()
