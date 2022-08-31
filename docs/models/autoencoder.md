# AutoEncoder

[Mymid Index](../README.md#mymid-index) /
[Models](./index.md#models) /
AutoEncoder

> Auto-generated documentation for [models.autoencoder](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py) module.

- [AutoEncoder](#autoencoder)
  - [AutoEncoder](#autoencoder-1)
    - [AutoEncoder().encode](#autoencoder()encode)
    - [AutoEncoder().generate](#autoencoder()generate)
    - [AutoEncoder().get_loss](#autoencoder()get_loss)

## AutoEncoder

[Show source in autoencoder.py:10](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py#L10)

#### Signature

```python
class AutoEncoder(Module):
    def __init__(self, config, encoder):
        ...
```

### AutoEncoder().encode

[Show source in autoencoder.py:28](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py#L28)

Try try try

#### Signature

```python
def encode(self, batch, node_type):
    ...
```

### AutoEncoder().generate

[Show source in autoencoder.py:35](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py#L35)

#### Signature

```python
def generate(
    self, batch, node_type, num_points, sample, bestof, flexibility=0.0, ret_traj=False
):
    ...
```

### AutoEncoder().get_loss

[Show source in autoencoder.py:43](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py#L43)

#### Signature

```python
def get_loss(self, batch, node_type):
    ...
```


