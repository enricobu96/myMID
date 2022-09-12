# Unicycle

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Dynamics](./index.md#dynamics) /
Unicycle

> Auto-generated documentation for [models.encoders.dynamics.unicycle](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py) module.

- [Unicycle](#unicycle)
  - [Unicycle](#unicycle-1)
    - [Unicycle().compute_control_jacobian](#unicycle()compute_control_jacobian)
    - [Unicycle().compute_jacobian](#unicycle()compute_jacobian)
    - [Unicycle().create_graph](#unicycle()create_graph)
    - [Unicycle().dynamic](#unicycle()dynamic)
    - [Unicycle().init_constants](#unicycle()init_constants)
    - [Unicycle().integrate_distribution](#unicycle()integrate_distribution)
    - [Unicycle().integrate_samples](#unicycle()integrate_samples)

## Unicycle

[Show source in unicycle.py:8](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L8)

#### Signature

```python
class Unicycle(Dynamic):
    ...
```

### Unicycle().compute_control_jacobian

[Show source in unicycle.py:80](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L80)

TODO: Boris: Add docstring

#### Arguments

- `x`
- `u`

#### Signature

```python
def compute_control_jacobian(self, sample_batch_dim, components, x, u):
    ...
```

### Unicycle().compute_jacobian

[Show source in unicycle.py:133](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L133)

TODO: Boris: Add docstring

#### Arguments

- `x`
- `u`

#### Signature

```python
def compute_jacobian(self, sample_batch_dim, components, x, u):
    ...
```

### Unicycle().create_graph

[Show source in unicycle.py:14](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L14)

#### Signature

```python
def create_graph(self, xz_size):
    ...
```

### Unicycle().dynamic

[Show source in unicycle.py:18](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L18)

TODO: Boris: Add docstring

#### Arguments

- `x`
- `u`

#### Signature

```python
def dynamic(self, x, u):
    ...
```

### Unicycle().init_constants

[Show source in unicycle.py:9](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L9)

#### Signature

```python
def init_constants(self):
    ...
```

### Unicycle().integrate_distribution

[Show source in unicycle.py:185](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L185)

TODO: Boris: Add docstring

#### Arguments

- `x`
- `u`

#### Signature

```python
def integrate_distribution(self, control_dist_dphi_a, x):
    ...
```

### Unicycle().integrate_samples

[Show source in unicycle.py:55](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/unicycle.py#L55)

TODO: Boris: Add docstring

#### Arguments

- `x`
- `u`

#### Signature

```python
def integrate_samples(self, control_samples, x=None):
    ...
```


