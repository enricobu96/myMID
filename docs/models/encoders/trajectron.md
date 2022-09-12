# Trajectron

[Mymid Index](../../README.md#mymid-index) /
[Models](../index.md#models) /
[Encoders](./index.md#encoders) /
Trajectron

> Auto-generated documentation for [models.encoders.trajectron](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py) module.

- [Trajectron](#trajectron)
  - [Trajectron](#trajectron-1)
    - [Trajectron().eval_loss](#trajectron()eval_loss)
    - [Trajectron().predict](#trajectron()predict)
    - [Trajectron().set_annealing_params](#trajectron()set_annealing_params)
    - [Trajectron().set_curr_iter](#trajectron()set_curr_iter)
    - [Trajectron().set_environment](#trajectron()set_environment)
    - [Trajectron().step_annealers](#trajectron()step_annealers)
    - [Trajectron().train_loss](#trajectron()train_loss)
  - [collate](#collate)
  - [get_node_timestep_data](#get_node_timestep_data)
  - [get_relative_robot_traj](#get_relative_robot_traj)
  - [get_timesteps_data](#get_timesteps_data)
  - [restore](#restore)

## Trajectron

[Show source in trajectron.py:241](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L241)

#### Signature

```python
class Trajectron(object):
    def __init__(self, model_registrar, hyperparams, device):
        ...
```

### Trajectron().eval_loss

[Show source in trajectron.py:333](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L333)

#### Signature

```python
def eval_loss(self, batch, node_type):
    ...
```

### Trajectron().predict

[Show source in trajectron.py:365](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L365)

#### Signature

```python
def predict(
    self,
    scene,
    timesteps,
    ph,
    num_samples=1,
    min_future_timesteps=0,
    min_history_timesteps=1,
    z_mode=False,
    gmm_mode=False,
    full_dist=True,
    all_z_sep=False,
    pcmd=False,
):
    ...
```

### Trajectron().set_annealing_params

[Show source in trajectron.py:290](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L290)

#### Signature

```python
def set_annealing_params(self):
    ...
```

### Trajectron().set_curr_iter

[Show source in trajectron.py:285](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L285)

#### Signature

```python
def set_curr_iter(self, curr_iter):
    ...
```

### Trajectron().set_environment

[Show source in trajectron.py:268](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L268)

#### Signature

```python
def set_environment(self, env):
    ...
```

### Trajectron().step_annealers

[Show source in trajectron.py:294](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L294)

#### Signature

```python
def step_annealers(self, node_type=None):
    ...
```

### Trajectron().train_loss

[Show source in trajectron.py:301](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L301)

#### Signature

```python
def train_loss(self, batch, node_type):
    ...
```



## collate

[Show source in trajectron.py:27](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L27)

#### Signature

```python
def collate(batch):
    ...
```



## get_node_timestep_data

[Show source in trajectron.py:71](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L71)

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

[Show source in trajectron.py:56](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L56)

#### Signature

```python
def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    ...
```



## get_timesteps_data

[Show source in trajectron.py:199](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L199)

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

[Show source in trajectron.py:14](https://github.com/enricobu96/myMID/blob/main/models/encoders/trajectron.py#L14)

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


