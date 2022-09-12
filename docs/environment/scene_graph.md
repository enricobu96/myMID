# SceneGraph

[Mymid Index](../README.md#mymid-index) /
[Environment](./index.md#environment) /
SceneGraph

> Auto-generated documentation for [environment.scene_graph](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py) module.

#### Attributes

- `B` - # # # # # # # # # # # # # # # #
  Testing edge mask calculation #
  # # # # # # # # # # # # # # # #: `np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]])[...]`


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

[Show source in scene_graph.py:65](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L65)

Used only inside Trajectron++ files.

#### Signature

```python
class DirectedEdge(Edge):
    def __init__(self, curr_node, other_node):
        ...
```

#### See also

- [Edge](#edge)

### DirectedEdge.get_edge_id

[Show source in scene_graph.py:72](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L72)

#### Signature

```python
@staticmethod
def get_edge_id(n1, n2):
    ...
```

### DirectedEdge.get_edge_type

[Show source in scene_graph.py:80](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L80)

#### Signature

```python
@staticmethod
def get_edge_type(n1, n2):
    ...
```

### DirectedEdge.get_str_from_types

[Show source in scene_graph.py:76](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L76)

#### Signature

```python
@staticmethod
def get_str_from_types(nt1, nt2):
    ...
```



## Edge

[Show source in scene_graph.py:9](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L9)

Used only inside Trajectron++ files.

#### Signature

```python
class Edge(object):
    def __init__(self, curr_node, other_node):
        ...
```

### Edge.get_edge_id

[Show source in scene_graph.py:19](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L19)

#### Signature

```python
@staticmethod
def get_edge_id(n1, n2):
    ...
```

### Edge.get_edge_type

[Show source in scene_graph.py:27](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L27)

#### Signature

```python
@staticmethod
def get_edge_type(n1, n2):
    ...
```

### Edge.get_str_from_types

[Show source in scene_graph.py:23](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L23)

#### Signature

```python
@staticmethod
def get_str_from_types(nt1, nt2):
    ...
```



## SceneGraph

[Show source in scene_graph.py:239](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L239)

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

[Show source in scene_graph.py:293](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L293)

#### Signature

```python
def get_connection_mask(self, node_index):
    ...
```

### SceneGraph().get_edge_scaling

[Show source in scene_graph.py:277](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L277)

#### Signature

```python
def get_edge_scaling(self, node=None):
    ...
```

### SceneGraph().get_edge_weight

[Show source in scene_graph.py:285](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L285)

#### Signature

```python
def get_edge_weight(self, node=None):
    ...
```

### SceneGraph().get_index

[Show source in scene_graph.py:258](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L258)

#### Signature

```python
def get_index(self, node):
    ...
```

### SceneGraph().get_neighbors

[Show source in scene_graph.py:264](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L264)

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

[Show source in scene_graph.py:261](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L261)

#### Signature

```python
def get_num_edges(self):
    ...
```



## TemporalSceneGraph

[Show source in scene_graph.py:85](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L85)

Used only inside Trajectron++ files.

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

[Show source in scene_graph.py:201](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L201)

#### Signature

```python
@staticmethod
def calculate_edge_scaling(adj_cube, edge_addition_filter, edge_removal_filter):
    ...
```

### TemporalSceneGraph().calculate_node_index_lookup

[Show source in scene_graph.py:108](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L108)

#### Signature

```python
def calculate_node_index_lookup(self):
    ...
```

### TemporalSceneGraph.create_from_temp_scene_dict

[Show source in scene_graph.py:121](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L121)

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

[Show source in scene_graph.py:118](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L118)

#### Signature

```python
def get_index(self, node):
    ...
```

### TemporalSceneGraph().get_num_edges

[Show source in scene_graph.py:115](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L115)

#### Signature

```python
def get_num_edges(self, t=0):
    ...
```

### TemporalSceneGraph().to_scene_graph

[Show source in scene_graph.py:217](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L217)

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

[Show source in scene_graph.py:45](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L45)

Used only inside Trajectron++ files.

#### Signature

```python
class UndirectedEdge(Edge):
    def __init__(self, curr_node, other_node):
        ...
```

#### See also

- [Edge](#edge)

### UndirectedEdge.get_edge_id

[Show source in scene_graph.py:52](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L52)

#### Signature

```python
@staticmethod
def get_edge_id(n1, n2):
    ...
```

### UndirectedEdge.get_edge_type

[Show source in scene_graph.py:60](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L60)

#### Signature

```python
@staticmethod
def get_edge_type(n1, n2):
    ...
```

### UndirectedEdge.get_str_from_types

[Show source in scene_graph.py:56](https://github.com/enricobu96/myMID/blob/main/environment/scene_graph.py#L56)

#### Signature

```python
@staticmethod
def get_str_from_types(nt1, nt2):
    ...
```


