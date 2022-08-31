# SceneGraph

[Mymid Index](../README.md#mymid-index) /
[Environment](./index.md#environment) /
SceneGraph

> Auto-generated documentation for [environment.scene_graph](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py) module.

#### Attributes

- `B` - # # # # # # # # # # # # # # # #
  Testing edge mask calculation #
  # # # # # # # # # # # # # # # #: `np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]])[:, :, np.newaxis, np.newaxis]`


- [SceneGraph](#scenegraph)
  - [DirectedEdge](#directededge)
    - [DirectedEdge.get_edge_id](#directededgeget_edge_id)
    - [DirectedEdge.get_edge_type](#directededgeget_edge_type)
    - [DirectedEdge.get_str_from_types](#directededgeget_str_from_types)
  - [Edge](#edge)
    - [Edge.get_edge_id](#edgeget_edge_id)
    - [Edge.get_edge_type](#edgeget_edge_type)
    - [Edge.get_str_from_types](#edgeget_str_from_types)
  - [SceneGraph](#scenegraph-1)
    - [SceneGraph().get_connection_mask](#scenegraph()get_connection_mask)
    - [SceneGraph().get_edge_scaling](#scenegraph()get_edge_scaling)
    - [SceneGraph().get_edge_weight](#scenegraph()get_edge_weight)
    - [SceneGraph().get_index](#scenegraph()get_index)
    - [SceneGraph().get_neighbors](#scenegraph()get_neighbors)
    - [SceneGraph().get_num_edges](#scenegraph()get_num_edges)
  - [TemporalSceneGraph](#temporalscenegraph)
    - [TemporalSceneGraph.calculate_edge_scaling](#temporalscenegraphcalculate_edge_scaling)
    - [TemporalSceneGraph().calculate_node_index_lookup](#temporalscenegraph()calculate_node_index_lookup)
    - [TemporalSceneGraph.create_from_temp_scene_dict](#temporalscenegraphcreate_from_temp_scene_dict)
    - [TemporalSceneGraph().get_index](#temporalscenegraph()get_index)
    - [TemporalSceneGraph().get_num_edges](#temporalscenegraph()get_num_edges)
    - [TemporalSceneGraph().to_scene_graph](#temporalscenegraph()to_scene_graph)
  - [UndirectedEdge](#undirectededge)
    - [UndirectedEdge.get_edge_id](#undirectededgeget_edge_id)
    - [UndirectedEdge.get_edge_type](#undirectededgeget_edge_type)
    - [UndirectedEdge.get_str_from_types](#undirectededgeget_str_from_types)

## DirectedEdge

[Show source in scene_graph.py:59](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L59)

#### Signature

```python
class DirectedEdge(Edge):
    def __init__(self, curr_node, other_node):
        ...
```

#### See also

- [Edge](#edge)

### DirectedEdge.get_edge_id

[Show source in scene_graph.py:63](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L63)

#### Signature

```python
@staticmethod
def get_edge_id(n1, n2):
    ...
```

### DirectedEdge.get_edge_type

[Show source in scene_graph.py:71](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L71)

#### Signature

```python
@staticmethod
def get_edge_type(n1, n2):
    ...
```

### DirectedEdge.get_str_from_types

[Show source in scene_graph.py:67](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L67)

#### Signature

```python
@staticmethod
def get_str_from_types(nt1, nt2):
    ...
```



## Edge

[Show source in scene_graph.py:9](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L9)

#### Signature

```python
class Edge(object):
    def __init__(self, curr_node, other_node):
        ...
```

### Edge.get_edge_id

[Show source in scene_graph.py:16](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L16)

#### Signature

```python
@staticmethod
def get_edge_id(n1, n2):
    ...
```

### Edge.get_edge_type

[Show source in scene_graph.py:24](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L24)

#### Signature

```python
@staticmethod
def get_edge_type(n1, n2):
    ...
```

### Edge.get_str_from_types

[Show source in scene_graph.py:20](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L20)

#### Signature

```python
@staticmethod
def get_str_from_types(nt1, nt2):
    ...
```



## SceneGraph

[Show source in scene_graph.py:227](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L227)

#### Signature

```python
class SceneGraph(object):
    def __init__(
        self,
        edge_radius,
        nodes=None,
        adj_mat=np.zeros((0, 0)),
        weight_mat=np.zeros((0, 0)),
        node_type_mat=np.zeros((0, 0)),
        node_index_lookup=None,
        edge_scaling=None,
    ):
        ...
```

### SceneGraph().get_connection_mask

[Show source in scene_graph.py:281](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L281)

#### Signature

```python
def get_connection_mask(self, node_index):
    ...
```

### SceneGraph().get_edge_scaling

[Show source in scene_graph.py:265](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L265)

#### Signature

```python
def get_edge_scaling(self, node=None):
    ...
```

### SceneGraph().get_edge_weight

[Show source in scene_graph.py:273](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L273)

#### Signature

```python
def get_edge_weight(self, node=None):
    ...
```

### SceneGraph().get_index

[Show source in scene_graph.py:246](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L246)

#### Signature

```python
def get_index(self, node):
    ...
```

### SceneGraph().get_neighbors

[Show source in scene_graph.py:252](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L252)

Get all neighbors of a node.

#### Arguments

- `node` - Node for which all neighbors are returned.
- `node_type` - Specifies node types which are returned.

#### Returns

List of all neighbors.

#### Signature

```python
def get_neighbors(self, node, node_type):
    ...
```

### SceneGraph().get_num_edges

[Show source in scene_graph.py:249](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L249)

#### Signature

```python
def get_num_edges(self):
    ...
```



## TemporalSceneGraph

[Show source in scene_graph.py:76](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L76)

#### Signature

```python
class TemporalSceneGraph(object):
    def __init__(
        self,
        edge_radius,
        nodes=None,
        adj_cube=np.zeros((1, 0, 0)),
        weight_cube=np.zeros((1, 0, 0)),
        node_type_mat=np.zeros((0, 0)),
        edge_scaling=None,
    ):
        ...
```

### TemporalSceneGraph.calculate_edge_scaling

[Show source in scene_graph.py:189](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L189)

#### Signature

```python
@staticmethod
def calculate_edge_scaling(adj_cube, edge_addition_filter, edge_removal_filter):
    ...
```

### TemporalSceneGraph().calculate_node_index_lookup

[Show source in scene_graph.py:96](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L96)

#### Signature

```python
def calculate_node_index_lookup(self):
    ...
```

### TemporalSceneGraph.create_from_temp_scene_dict

[Show source in scene_graph.py:109](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L109)

Construct a spatiotemporal graph from node positions in a dataset.

#### Arguments

- `scene_temp_dict` - Dict with all nodes in scene as keys and np.ndarray with positions as value
- [attention_radius](#scenegraph) - Attention radius dict.
- `duration` - Temporal duration of the graph.
- [edge_addition_filter](#scenegraph) - -
- [edge_removal_filter](#scenegraph) - -

#### Returns

TemporalSceneGraph

#### Signature

```python
@classmethod
def create_from_temp_scene_dict(
    cls,
    scene_temp_dict,
    attention_radius,
    duration=1,
    edge_addition_filter=None,
    edge_removal_filter=None,
    online=False,
):
    ...
```

### TemporalSceneGraph().get_index

[Show source in scene_graph.py:106](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L106)

#### Signature

```python
def get_index(self, node):
    ...
```

### TemporalSceneGraph().get_num_edges

[Show source in scene_graph.py:103](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L103)

#### Signature

```python
def get_num_edges(self, t=0):
    ...
```

### TemporalSceneGraph().to_scene_graph

[Show source in scene_graph.py:205](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L205)

Creates a Scene Graph from a Temporal Scene Graph

#### Arguments

- `t` - Time in Temporal Scene Graph for which Scene Graph is created.
- `t_hist` - Number of history timesteps which are considered to form edges in Scene Graph.
- `t_fut` - Number of future timesteps which are considered to form edges in Scene Graph.

#### Returns

SceneGraph

#### Signature

```python
def to_scene_graph(self, t, t_hist=0, t_fut=0):
    ...
```



## UndirectedEdge

[Show source in scene_graph.py:42](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L42)

#### Signature

```python
class UndirectedEdge(Edge):
    def __init__(self, curr_node, other_node):
        ...
```

#### See also

- [Edge](#edge)

### UndirectedEdge.get_edge_id

[Show source in scene_graph.py:46](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L46)

#### Signature

```python
@staticmethod
def get_edge_id(n1, n2):
    ...
```

### UndirectedEdge.get_edge_type

[Show source in scene_graph.py:54](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L54)

#### Signature

```python
@staticmethod
def get_edge_type(n1, n2):
    ...
```

### UndirectedEdge.get_str_from_types

[Show source in scene_graph.py:50](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L50)

#### Signature

```python
@staticmethod
def get_str_from_types(nt1, nt2):
    ...
```


