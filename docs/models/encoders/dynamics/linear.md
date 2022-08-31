# Linear

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Dynamics](./index.md#dynamics) /
Linear

> Auto-generated documentation for [models.encoders.dynamics.linear](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/linear.py) module.

- [Linear](#linear)
  - [Linear](#linear-1)
    - [Linear().init_constants](#linear()init_constants)
    - [Linear().integrate_distribution](#linear()integrate_distribution)
    - [Linear().integrate_samples](#linear()integrate_samples)

## Linear

[Show source in linear.py:4](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/linear.py#L4)

#### Signature

```python
class Linear(Dynamic):
    ...
```

### Linear().init_constants

[Show source in linear.py:5](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/linear.py#L5)

#### Signature

```python
def init_constants(self):
    ...
```

### Linear().integrate_distribution

[Show source in linear.py:11](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/linear.py#L11)

#### Signature

```python
def integrate_distribution(self, v_dist, x):
    ...
```

### Linear().integrate_samples

[Show source in linear.py:8](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/linear.py#L8)

#### Signature

```python
def integrate_samples(self, v, x):
    ...
```


