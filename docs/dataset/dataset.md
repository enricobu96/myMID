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

Environment dataset class. It just creates the dataset for training and evaluation
in a proper way (sets hyperparameters, performs augmentation). Nothing crucial.

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

[Show source in dataset.py:26](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L26)

#### Signature

```python
@property
def augment(self):
    ...
```

### EnvironmentDataset().augment

[Show source in dataset.py:30](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L30)

#### Signature

```python
@augment.setter
def augment(self, value):
    ...
```



## NodeTypeDataset

[Show source in dataset.py:40](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L40)

Utility class for creating a NodeType dataset. Creates index and stuff, not crucial.

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

[Show source in dataset.py:60](https://github.com/enricobu96/myMID/blob/main/dataset/dataset.py#L60)

#### Signature

```python
def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
    ...
```


