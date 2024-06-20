import json
from typing import Dict

from encoding.encoders import SparseResUNet
from features_aggregation.aggregators import MaxPoolAggregator, AvgPoolAggregator
from features_propagation.propagators import AppendPropagator
from pipeline.branch import Branch
from pipeline.unit import L2GUnit, G2LUnit


def _create_l2g_units(config, components):
    return [
        L2GUnit(
            encoder=components[unit['encoder']],
            aggregator=components[unit['aggregator']]
        )
        for unit in config
    ]


def _create_g2l_units(config, components):
    return [
        G2LUnit(
            encoder=components[unit['encoder']],
            aggregator=components[unit['aggregator']],
            propagator=components[unit['propagator']]
        )
        for unit in config
    ]


def build_branches(config_path: str) -> Dict:
    module_dict = {
        'encoders': {'SparseResUNet': SparseResUNet},
        'aggregators': {'MaxPoolAggregator': MaxPoolAggregator, "AvgPoolAggregator": AvgPoolAggregator},
        'propagators': {'AppendPropagator': AppendPropagator}
    }

    with open(config_path) as f:
        config = json.load(f)

    # Build components from configuration
    components = {}
    for group_name, group in config.items():
        if group_name in ['L2GBranch', 'G2LBranch']:
            continue
        for item in group:
            class_ = module_dict[group_name][item['class']]
            components[item['name']] = class_(*item['args'])

    # Build branches from components
    branches = {
        'L2GBranch': Branch(units=_create_l2g_units(config['L2GBranch'], components)),
        'G2LBranch': Branch(units=_create_g2l_units(config['G2LBranch'], components)),
    }

    return branches


def build_losses(branch: Branch) -> Dict:
    # TODO
    pass
