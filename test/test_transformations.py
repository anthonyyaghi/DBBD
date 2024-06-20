import sys
sys.path.append('../src')

import unittest
import numpy as np
from transformation.transformations import RotateTransformer
from transformation.utils import euler_to_rotation_matrix

class TestRotatePointCloud(unittest.TestCase):
    def setUp(self):
        # Define a simple rotation matrix for 90 degrees around the Z-axis
        self.rotation_matrix = euler_to_rotation_matrix(0, 0, np.pi/2)
        self.rotator = RotateTransformer(self.rotation_matrix)

        # Define a sample point cloud
        self.point_cloud = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0]
        ])

    def test_rotation_transformation(self):
        # Expected results after rotation
        expected_transformed = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [1, 0, 0]
        ])

        # Perform the transformation
        transformed = self.rotator.transform(self.point_cloud)

        # Check if the transformed point cloud matches the expected results
        np.testing.assert_array_almost_equal(transformed, expected_transformed, decimal=6, err_msg="Rotation transformation did not produce expected results.")

    def test_identity_mapping(self):
        # Perform the transformation to initialize the mapping
        self.rotator.transform(self.point_cloud)

        # Retrieve the mapping
        mapping = self.rotator.get_mapping()

        # Check if the mapping is an identity mapping
        expected_mapping = np.arange(self.point_cloud.shape[0])
        np.testing.assert_array_equal(mapping, expected_mapping, err_msg="Mapping is not identity as expected.")

if __name__ == '__main__':
    unittest.main()
