# Node

[Mymid Index](../README.md#mymid-index) /
[Environment](./index.md#environment) /
Node

> Auto-generated documentation for [environment.node](https://github.com/enricobu96/myMID/blob/main/environment/node.py) module.

- [Node](#node)
  - [MultiNode](#multinode)
    - [MultiNode.find_non_overlapping_nodes](#multinodefind_non_overlapping_nodes)
    - [MultiNode().get](#multinode()get)
    - [MultiNode().get_all](#multinode()get_all)
    - [MultiNode().get_node_at_timesteps](#multinode()get_node_at_timesteps)
    - [MultiNode().history_points_at](#multinode()history_points_at)
    - [MultiNode().scene_ts_to_node_ts](#multinode()scene_ts_to_node_ts)
    - [MultiNode().timesteps](#multinode()timesteps)
  - [Node](#node-1)
    - [Node().get](#node()get)
    - [Node().history_points_at](#node()history_points_at)
    - [Node().last_timestep](#node()last_timestep)
    - [Node().overwrite_data](#node()overwrite_data)
    - [Node().scene_ts_to_node_ts](#node()scene_ts_to_node_ts)
    - [Node().timesteps](#node()timesteps)

## MultiNode

[Show source in node.py:126](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L126)

#### Signature

```python
class MultiNode(Node):
    def __init__(self, node_type, node_id, nodes_list, is_robot=False):
        ...
```

#### See also

- [Node](#node)

### MultiNode.find_non_overlapping_nodes

[Show source in node.py:141](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L141)

Greedily finds a set of non-overlapping nodes in the provided scene.

#### Returns

A list of non-overlapping nodes.

#### Signature

```python
@staticmethod
def find_non_overlapping_nodes(nodes_list, min_timesteps=1) -> list:
    ...
```

### MultiNode().get

[Show source in node.py:191](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L191)

#### Signature

```python
def get(self, tr_scene, state, padding=np.nan) -> np.ndarray:
    ...
```

### MultiNode().get_all

[Show source in node.py:206](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L206)

#### Signature

```python
def get_all(self, tr_scene, state, padding=np.nan) -> np.ndarray:
    ...
```

### MultiNode().get_node_at_timesteps

[Show source in node.py:159](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L159)

#### Signature

```python
def get_node_at_timesteps(self, scene_ts) -> Node:
    ...
```

#### See also

- [Node](#node)

### MultiNode().history_points_at

[Show source in node.py:218](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L218)

Number of history points in trajectory. Timestep is exclusive.

#### Arguments

- `ts` - Scene timestep where the number of history points are queried.

#### Returns

Number of history timesteps.

#### Signature

```python
def history_points_at(self, ts) -> int:
    ...
```

### MultiNode().scene_ts_to_node_ts

[Show source in node.py:170](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L170)

Transforms timestamp from scene into timeframe of node data.

#### Arguments

- `scene_ts` - Scene timesteps

#### Returns

ts: Transformed timesteps, paddingl: Number of timesteps in scene range which are not available in
        node data before data is available. paddingu: Number of timesteps in scene range which are not
        available in node data after data is available.

#### Signature

```python
def scene_ts_to_node_ts(self, scene_ts) -> (Node, np.ndarray, int, int):
    ...
```

#### See also

- [Node](#node)

### MultiNode().timesteps

[Show source in node.py:229](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L229)

Number of available timesteps for node.

#### Returns

Number of available timesteps.

#### Signature

```python
@property
def timesteps(self) -> int:
    ...
```



## Node

[Show source in node.py:8](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L8)

#### Signature

```python
class Node(object):
    def __init__(
        self,
        node_type,
        node_id,
        data,
        length=None,
        width=None,
        height=None,
        first_timestep=0,
        is_robot=False,
        description="",
        frequency_multiplier=1,
        non_aug_node=None,
    ):
        ...
```

### Node().get

[Show source in node.py:87](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L87)

Returns a time range of multiple properties of the node.

#### Arguments

- `tr_scene` - The timestep range (inklusive).
- `state` - The state description for which the properties are returned.
- `padding` - The value which should be used for padding if not enough information is available.

#### Returns

Array of node property values.

#### Signature

```python
def get(self, tr_scene, state, padding=np.nan) -> np.ndarray:
    ...
```

### Node().history_points_at

[Show source in node.py:78](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L78)

Number of history points in trajectory. Timestep is exclusive.

#### Arguments

- `ts` - Scene timestep where the number of history points are queried.

#### Returns

Number of history timesteps.

#### Signature

```python
def history_points_at(self, ts) -> int:
    ...
```

### Node().last_timestep

[Show source in node.py:114](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L114)

Nodes last timestep in the Scene.

#### Returns

Nodes last timestep.

#### Signature

```python
@property
def last_timestep(self) -> int:
    ...
```

### Node().overwrite_data

[Show source in node.py:49](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L49)

This function hard overwrites the data matrix. When using it you have to make sure that the columns
in the new data matrix correspond to the old structure. As well as setting first_timestep.

#### Arguments

- `data` - New data matrix
- `forward_in_time_on_next_overwrite` - On the !!NEXT!! call of overwrite_data first_timestep will be increased.

#### Returns

None

#### Signature

```python
def overwrite_data(self, data, forward_in_time_on_next_overwrite=False):
    ...
```

### Node().scene_ts_to_node_ts

[Show source in node.py:64](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L64)

Transforms timestamp from scene into timeframe of node data.

#### Arguments

- `scene_ts` - Scene timesteps

#### Returns

ts: Transformed timesteps, paddingl: Number of timesteps in scene range which are not available in
        node data before data is available. paddingu: Number of timesteps in scene range which are not
        available in node data after data is available.

#### Signature

```python
def scene_ts_to_node_ts(self, scene_ts) -> (np.ndarray, int, int):
    ...
```

### Node().timesteps

[Show source in node.py:105](https://github.com/enricobu96/myMID/blob/main/environment/node.py#L105)

Number of available timesteps for node.

#### Returns

Number of available timesteps.

#### Signature

```python
@property
def timesteps(self) -> int:
    ...
```


