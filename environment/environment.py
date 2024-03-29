import orjson
import numpy as np
from itertools import product
from .node_type import NodeTypeEnum


class Environment(object):
    """
    Environment class. Contains information on scenes and attention.

    Args:
        node_type_list: list(str) : list of types of node, e.g. PEDESTRIAN
        standardization: dict() : information on standardization of scenes (mean and std for x and y)
        scenes: list(Scene) : list of scenes, each composed by duration in seconds and number of nodes (for better understanding, read doc for Scene class)
        attention_radius: int : radius for attention mechanism
        robot_type: str : 
        NodeType: Enum : enum istantiated with node_type_list
        standardize_param_memo: dict() : 
        _scenes_resample_prop: None : 

    Methods:
        get_edge_types(): list() : returns cartesian product of the node types in a list
        get_standardize_params(state,node_type): np.stack(), np.stack() : returns parameters for standardization
        standardize(array,state,node_type,mean=None,std=None): : returns parameters for standardization for each scene
        unstandardize(self,array,state,node_type,mean=None,std=None): : returns parameters for unstandardization for each scene
    """
    def __init__(self, node_type_list, standardization, scenes=None, attention_radius=None, robot_type=None):
        self.scenes = scenes
        self.node_type_list = node_type_list
        self.attention_radius = attention_radius
        self.NodeType = NodeTypeEnum(node_type_list)
        self.robot_type = robot_type

        self.standardization = standardization
        self.standardize_param_memo = dict()

        self._scenes_resample_prop = None

    def get_edge_types(self):
        return list(product(self.NodeType, repeat=2))

    def get_standardize_params(self, state, node_type):
        memo_key = (orjson.dumps(state), node_type)
        if memo_key in self.standardize_param_memo:
            return self.standardize_param_memo[memo_key]

        standardize_mean_list = list()
        standardize_std_list = list()
        for entity, dims in state.items():
            for dim in dims:
                standardize_mean_list.append(self.standardization[node_type][entity][dim]['mean'])
                standardize_std_list.append(self.standardization[node_type][entity][dim]['std'])
        standardize_mean = np.stack(standardize_mean_list)
        standardize_std = np.stack(standardize_std_list)

        self.standardize_param_memo[memo_key] = (standardize_mean, standardize_std)
        return standardize_mean, standardize_std

    def standardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        return np.where(np.isnan(array), np.array(np.nan), (array - mean) / std)

    def unstandardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        print(mean, std)
        return array * std + mean

    @property
    def scenes_resample_prop(self):
        if self._scenes_resample_prop is None:
            self._scenes_resample_prop = np.array([scene.resample_prob for scene in self.scenes])
            self._scenes_resample_prop = self._scenes_resample_prop / np.sum(self._scenes_resample_prop)
        return self._scenes_resample_prop

