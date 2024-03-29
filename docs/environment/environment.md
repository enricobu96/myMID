# Environment

[Mymid Index](../README.md#mymid-index) /
[Environment](./index.md#environment) /
Environment

> Auto-generated documentation for [environment.environment](https://github.com/enricobu96/myMID/blob/main/environment/environment.py) module.

- [Environment](#environment)
  - [Environment](#environment-1)
    - [Environment().get_edge_types](#environment()get_edge_types)
    - [Environment().get_standardize_params](#environment()get_standardize_params)
    - [Environment().scenes_resample_prop](#environment()scenes_resample_prop)
    - [Environment().standardize](#environment()standardize)
    - [Environment().unstandardize](#environment()unstandardize)

## Environment

[Show source in environment.py:7](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L7)

Environment class. Contains information on scenes and attention.

#### Arguments

- `node_type_list` - list(str) : list of types of node, e.g. PEDESTRIAN
- `standardization` - dict() : information on standardization of scenes (mean and std for x and y)
- `scenes` - list(Scene) : list of scenes, each composed by duration in seconds and number of nodes (for better understanding, read doc for Scene class)
- `attention_radius` - int : radius for attention mechanism
- `robot_type` - str :
- `NodeType` - Enum : enum istantiated with node_type_list
- `standardize_param_memo` - dict() :
- `_scenes_resample_prop` - None :

#### Methods

- `get_edge_types()` - list() : returns cartesian product of the node types in a list
- `get_standardize_params(state,node_type)` - np.stack(), np.stack() : returns parameters for standardization
- `standardize(array,state,node_type,mean=None,std=None)` - : returns parameters for standardization for each scene
- `unstandardize(self,array,state,node_type,mean=None,std=None)` - : returns parameters for unstandardization for each scene

#### Signature

```python
class Environment(object):
    def __init__(
        self,
        node_type_list,
        standardization,
        scenes=None,
        attention_radius=None,
        robot_type=None,
    ):
        ...
```

### Environment().get_edge_types

[Show source in environment.py:39](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L39)

#### Signature

```python
def get_edge_types(self):
    ...
```

### Environment().get_standardize_params

[Show source in environment.py:42](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L42)

#### Signature

```python
def get_standardize_params(self, state, node_type):
    ...
```

### Environment().scenes_resample_prop

[Show source in environment.py:78](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L78)

#### Signature

```python
@property
def scenes_resample_prop(self):
    ...
```

### Environment().standardize

[Show source in environment.py:59](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L59)

#### Signature

```python
def standardize(self, array, state, node_type, mean=None, std=None):
    ...
```

### Environment().unstandardize

[Show source in environment.py:68](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L68)

#### Signature

```python
def unstandardize(self, array, state, node_type, mean=None, std=None):
    ...
```


