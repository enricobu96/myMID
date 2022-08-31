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

[Show source in environment.py:20](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L20)

#### Signature

```python
def get_edge_types(self):
    ...
```

### Environment().get_standardize_params

[Show source in environment.py:23](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L23)

#### Signature

```python
def get_standardize_params(self, state, node_type):
    ...
```

### Environment().scenes_resample_prop

[Show source in environment.py:59](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L59)

#### Signature

```python
@property
def scenes_resample_prop(self):
    ...
```

### Environment().standardize

[Show source in environment.py:40](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L40)

#### Signature

```python
def standardize(self, array, state, node_type, mean=None, std=None):
    ...
```

### Environment().unstandardize

[Show source in environment.py:49](https://github.com/enricobu96/myMID/blob/main/environment/environment.py#L49)

#### Signature

```python
def unstandardize(self, array, state, node_type, mean=None, std=None):
    ...
```


