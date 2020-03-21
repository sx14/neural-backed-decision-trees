"""
For external use as part of nbdt package. This is a model wrapper that
runs inferecne as an NBDT. Note these wrappers make no assumption about the
underlying neural network other than it (1) is a classification model and
(2) returns logits.
"""

import torch.nn as nn
from nbdt.utils import (
    dataset_to_default_path_graph,
    dataset_to_default_path_wnids,
    hierarchy_to_path_graph)
from nbdt.data.custom import dataset_to_dummy_classes
from nbdt.analysis import SoftEmbeddedDecisionRules, HardEmbeddedDecisionRules


class NBDT(nn.Module):

    def __init__(self, path_graph, path_wnids, classes,
            model, Rules=HardEmbeddedDecisionRules):
        super().__init__()
        self.rules = Rules(path_graph, path_wnids, classes)
        self.model = model

    @classmethod
    def with_defaults(cls, dataset, model, hierarchy=None, **kwargs):
        assert 'path_graph' not in kwargs and 'path_wnids' not in kwargs, \
            '`from_dataset` sets both the path_graph and path_wnids'
        path_graph = dataset_to_default_path_graph(dataset)
        path_wnids = dataset_to_default_path_wnids(dataset)
        classes = dataset_to_dummy_classes(dataset)

        if hierarchy:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        return cls(path_graph, path_wnids, classes, model, **kwargs)

    def forward(self, x):
        x = self.model(x)
        x = self.rules.forward(x)
        return x


class HardNBDT(NBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({
            'Rules': HardEmbeddedDecisionRules
        })
        super().__init__(*args, **kwargs)


class SoftNBDT(NBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({
            'Rules': SoftEmbeddedDecisionRules
        })
        super().__init__(*args, **kwargs)