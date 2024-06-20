import os

from dataset.pointcloud_dataset import PointCloudDataset
from pipeline.builder import build_branches
from region_proposal.generation import hierarchical_region_proposal

if __name__ == '__main__':
    root_dir = '/Users/anthonyyaghi/PycharmProjects/DBBD-SSL'
    dataset = PointCloudDataset(os.path.join(root_dir, 'data/pointclouds'), colored=True)
    regions_generator = hierarchical_region_proposal
    branches = build_branches(config_path=os.path.join(root_dir, '/config/pipeline.json'))
