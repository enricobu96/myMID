# Preprocessing

[Mymid Index](../README.md#mymid-index) /
[Dataset](./index.md#dataset) /
Preprocessing

> Auto-generated documentation for [dataset.preprocessing](https://github.com/enricobu96/myMID/blob/main/dataset/preprocessing.py) module.

- [Preprocessing](#preprocessing)
  - [collate](#collate)
  - [get_node_timestep_data](#get_node_timestep_data)
  - [get_relative_robot_traj](#get_relative_robot_traj)
  - [get_timesteps_data](#get_timesteps_data)
  - [restore](#restore)

## collate

[Show source in preprocessing.py:26](https://github.com/enricobu96/myMID/blob/main/dataset/preprocessing.py#L26)

#### Signature

```python
def collate(batch):
    ...
```



## get_node_timestep_data

[Show source in preprocessing.py:70](https://github.com/enricobu96/myMID/blob/main/dataset/preprocessing.py#L70)

Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
as well as the neighbour data for it.

#### Arguments

- `env` - Environment
- `scene` - Scene
- `t` - Timestep in scene
- `node` - Node
- `state` - Specification of the node state
- `pred_state` - Specification of the prediction state
- `edge_types` - List of all Edge Types for which neighbours are pre-processed
- `max_ht` - Maximum history timesteps
- `max_ft` - Maximum future timesteps (prediction horizon)
- `hyperparams` - Model hyperparameters
- `scene_graph` - If scene graph was already computed for this scene and time you can pass it here

#### Returns

Batch Element

#### Signature

```python
def get_node_timestep_data(
    env,
    scene,
    t,
    node,
    state,
    pred_state,
    edge_types,
    max_ht,
    max_ft,
    hyperparams,
    scene_graph=None,
):
    ...
```



## get_relative_robot_traj

[Show source in preprocessing.py:55](https://github.com/enricobu96/myMID/blob/main/dataset/preprocessing.py#L55)

#### Signature

```python
def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    ...
```



## get_timesteps_data

[Show source in preprocessing.py:198](https://github.com/enricobu96/myMID/blob/main/dataset/preprocessing.py#L198)

Puts together the inputs for ALL nodes in a given scene and timestep in it.

#### Arguments

- `env` - Environment
- `scene` - Scene
- `t` - Timestep in scene
- `node_type` - Node Type of nodes for which the data shall be pre-processed
- `state` - Specification of the node state
- `pred_state` - Specification of the prediction state
- `edge_types` - List of all Edge Types for which neighbors are pre-processed
- `max_ht` - Maximum history timesteps
- `max_ft` - Maximum future timesteps (prediction horizon)
- `hyperparams` - Model hyperparameters

#### Signature

```python
def get_timesteps_data(
    env,
    scene,
    t,
    node_type,
    state,
    pred_state,
    edge_types,
    min_ht,
    max_ht,
    min_ft,
    max_ft,
    hyperparams,
):
    ...
```



## restore

[Show source in preprocessing.py:13](https://github.com/enricobu96/myMID/blob/main/dataset/preprocessing.py#L13)

In case we dilled some structures to share between multiple process this function will restore them.
If the data input are not bytes we assume it was not dilled in the first place

#### Arguments

- `data` - Possibly dilled data structure

#### Returns

Un-dilled data structure

#### Signature

```python
def restore(data):
    ...
```


