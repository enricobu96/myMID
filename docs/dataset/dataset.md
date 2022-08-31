# Dataset

[Mymid Index](../README.md#mymid-index) /
[Dataset](./index.md#dataset) /
Dataset

> Auto-generated documentation for [dataset.dataset](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py) module.

- [Dataset](#dataset)
  - [EnvironmentDataset](#environmentdataset)
    - [EnvironmentDataset().augment](#environmentdataset()augment)
    - [EnvironmentDataset().augment](#environmentdataset()augment-1)
  - [NodeTypeDataset](#nodetypedataset)
    - [NodeTypeDataset().index_env](#nodetypedataset()index_env)

## EnvironmentDataset

[Show source in dataset.py:6](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L6)

#### Signature

```python
class EnvironmentDataset(object):
    def __init__(
        self,
        env,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        **kwargs
    ):
        ...
```

### EnvironmentDataset().augment

[Show source in dataset.py:22](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L22)

#### Signature

```python
@property
def augment(self):
    ...
```

### EnvironmentDataset().augment

[Show source in dataset.py:26](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L26)

#### Signature

```python
@augment.setter
def augment(self, value):
    ...
```



## NodeTypeDataset

[Show source in dataset.py:36](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L36)

#### Signature

```python
class NodeTypeDataset(data.Dataset):
    def __init__(
        self,
        env,
        node_type,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        augment=False,
        **kwargs
    ):
        ...
```

### NodeTypeDataset().index_env

[Show source in dataset.py:53](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L53)

#### Signature

```python
def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
    ...
```


