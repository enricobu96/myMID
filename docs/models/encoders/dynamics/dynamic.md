# Dynamic

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Dynamics](./index.md#dynamics) /
Dynamic

> Auto-generated documentation for [models.encoders.dynamics.dynamic](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py) module.

- [Dynamic](#dynamic)
  - [Dynamic](#dynamic-1)
    - [Dynamic().create_graph](#dynamic()create_graph)
    - [Dynamic().create_graph](#dynamic()create_graph-1)
    - [Dynamic().init_constants](#dynamic()init_constants)
    - [Dynamic().integrate_distribution](#dynamic()integrate_distribution)
    - [Dynamic().integrate_samples](#dynamic()integrate_samples)
    - [Dynamic().set_initial_condition](#dynamic()set_initial_condition)

## Dynamic

[Show source in dynamic.py:3](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py#L3)

#### Signature

```python
class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, model_registrar, xz_size, node_type):
        ...
```

### Dynamic().create_graph

[Show source in dynamic.py:20](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py#L20)

#### Signature

```python
def create_graph(self, xz_size):
    ...
```

### Dynamic().create_graph

[Show source in dynamic.py:29](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py#L29)

#### Signature

```python
def create_graph(self, xz_size):
    ...
```

### Dynamic().init_constants

[Show source in dynamic.py:17](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py#L17)

#### Signature

```python
def init_constants(self):
    ...
```

### Dynamic().integrate_distribution

[Show source in dynamic.py:26](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py#L26)

#### Signature

```python
def integrate_distribution(self, dist, x):
    ...
```

### Dynamic().integrate_samples

[Show source in dynamic.py:23](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py#L23)

#### Signature

```python
def integrate_samples(self, s, x):
    ...
```

### Dynamic().set_initial_condition

[Show source in dynamic.py:14](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/dynamic.py#L14)

#### Signature

```python
def set_initial_condition(self, init_con):
    ...
```


