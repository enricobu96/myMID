# Scene

[Mymid Index](../README.md#mymid-index) /
[Environment](./index.md#environment) /
Scene

> Auto-generated documentation for [environment.scene](https://github.com/enricobu96/myMID/blob/main/environment/scene.py) module.

- [Scene](#scene)
  - [Scene](#scene-1)
    - [Scene().add_robot_from_nodes](#scene()add_robot_from_nodes)
    - [Scene().augment](#scene()augment)
    - [Scene().calculate_scene_graph](#scene()calculate_scene_graph)
    - [Scene().duration](#scene()duration)
    - [Scene().get_clipped_pos_dict](#scene()get_clipped_pos_dict)
    - [Scene().get_node_by_id](#scene()get_node_by_id)
    - [Scene().get_nodes_clipped_at_time](#scene()get_nodes_clipped_at_time)
    - [Scene().get_scene_graph](#scene()get_scene_graph)
    - [Scene().present_nodes](#scene()present_nodes)
    - [Scene().sample_timesteps](#scene()sample_timesteps)

## Scene

[Show source in scene.py:7](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L7)

Scene class, represents a scene with its params.

Params
------
timesteps : int
    number of timesteps
map : object (None)
    map for visualization purposes
dt : float
    hyperparameter, dt of the time for derivation of space (in order to get velocity and acceleration)
name : str
    name of the dataset
frequency_multiplier : int
    multiplier for the frequency when operating in node_freq_mult
aug_func : function
    augmentation function
non_aug_scene : Scene
    non-augmented scene

Attributes
----------
map : object (None)
    map for visualization purposes
timesteps : int
    number of timesteps
dt : float
    hyperparameter, dt of the time for derivation of space (in order to get velocity and acceleration)
name : str
    name of the dataset
robot : Node
    robot node when operating with it
temporal_scene_graph : SceneGraph
    see SceneGraph documentation
frequency_multiplier : int
    multiplier for the frequency when operating in node_freq_mult
description : str
    description of the dataset
aug_func : function
    augmentation function
non_aug_scene : Scene
    non-augmented scene

Methods
-------
add_robot_from_nodes(robot_type) -> None
    appends robot node to nodes when operating with robot node
get_clipped_pos_dict(timestep, state) -> dict()
    returns nodes clipped at time
get_scene_graph(timestep, attention_radius=None, edge_addition_filter=None, edge_removal_filter=None) -> SceneGraph
    returns the Scene Graph for a given timestep. Read documentation of the method
calculate_scene_graph(attention_radius,edge_addition_filter=None, edge_removal_filter=None) -> None
    calculate the Temporal Scene Graph for the entire Scene. Read documentation fo the method
duration() -> int
    returns the duration of the scene
present_nodes(timesteps, type=None, min_history_timesteps=0, min_future_timesteps=0, return_robot=True) -> dict
    finds all present nodes in the scene at a given timestemp
get_nodes_clipped_at_time(timesteps, state) -> list(Nodes)
    clips nodes at time
sample_timesteps(batch_size, min_future_timesteps=0) -> np.ndarray
    sample a batch size of possible timesteps for the scene
augment() -> self
    performs augmentation
get_node_by_id(id) -> Node
    returns node by id

#### Signature

```python
class Scene(object):
    def __init__(
        self,
        timesteps,
        map=None,
        dt=1,
        name="",
        frequency_multiplier=1,
        aug_func=None,
        non_aug_scene=None,
    ):
        ...
```

### Scene().add_robot_from_nodes

[Show source in scene.py:94](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L94)

#### Signature

```python
def add_robot_from_nodes(self, robot_type):
    ...
```

### Scene().augment

[Show source in scene.py:257](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L257)

#### Signature

```python
def augment(self):
    ...
```

### Scene().calculate_scene_graph

[Show source in scene.py:150](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L150)

Calculate the Temporal Scene Graph for the entire Scene.

#### Arguments

- `attention_radius` - Attention radius for each node type permutation.
- `edge_addition_filter` - Filter for adding edges.
- `edge_removal_filter` - Filter for removing edges.

#### Returns

None

#### Signature

```python
def calculate_scene_graph(
    self, attention_radius, edge_addition_filter=None, edge_removal_filter=None
) -> None:
    ...
```

### Scene().duration

[Show source in scene.py:177](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L177)

Calculates the duration of the scene.

#### Returns

Duration of the scene in s.

#### Signature

```python
def duration(self):
    ...
```

### Scene().get_clipped_pos_dict

[Show source in scene.py:103](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L103)

#### Signature

```python
def get_clipped_pos_dict(self, timestep, state):
    ...
```

### Scene().get_node_by_id

[Show source in scene.py:263](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L263)

#### Signature

```python
def get_node_by_id(self, id):
    ...
```

### Scene().get_nodes_clipped_at_time

[Show source in scene.py:221](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L221)

#### Signature

```python
def get_nodes_clipped_at_time(self, timesteps, state):
    ...
```

### Scene().get_scene_graph

[Show source in scene.py:113](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L113)

Returns the Scene Graph for a given timestep. If the Temporal Scene Graph was pre calculated,
the temporal scene graph is sliced. Otherwise the scene graph is calculated on the spot.

#### Arguments

- `timestep` - Timestep for which the scene graph is returned.
- `attention_radius` - Attention radius for each node type permutation. (Only online)
- `edge_addition_filter` - Filter for adding edges (Only online)
- `edge_removal_filter` - Filter for removing edges (Only online)

#### Returns

Scene Graph for given timestep.

#### Signature

```python
def get_scene_graph(
    self,
    timestep,
    attention_radius=None,
    edge_addition_filter=None,
    edge_removal_filter=None,
) -> SceneGraph:
    ...
```

### Scene().present_nodes

[Show source in scene.py:185](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L185)

Finds all present nodes in the scene at a given timestemp

#### Arguments

- `timesteps` - Timestep(s) for which all present nodes should be returned
- `type` - Node type which should be returned. If None all node types are returned.
- `min_history_timesteps` - Minimum history timesteps of a node to be returned.
- `min_future_timesteps` - Minimum future timesteps of a node to be returned.
- `return_robot` - Return a node if it is the robot.

#### Returns

Dictionary with timesteps as keys and list of nodes as value.

#### Signature

```python
def present_nodes(
    self,
    timesteps,
    type=None,
    min_history_timesteps=0,
    min_future_timesteps=0,
    return_robot=True,
) -> dict:
    ...
```

### Scene().sample_timesteps

[Show source in scene.py:245](https://github.com/enricobu96/myMID/blob/main/environment/scene.py#L245)

Sample a batch size of possible timesteps for the scene.

#### Arguments

- `batch_size` - Number of timesteps to sample.
- `min_future_timesteps` - Minimum future timesteps in the scene for a timestep to be returned.

#### Returns

Numpy Array of sampled timesteps.

#### Signature

```python
def sample_timesteps(self, batch_size, min_future_timesteps=0) -> np.ndarray:
    ...
```


