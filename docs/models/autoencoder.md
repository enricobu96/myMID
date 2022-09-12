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

Custom AutoEncoder class for the project. In this project, Trajectron++ is used as the temporal-social encoder:
it encodes the history path and social interaction clues into a state embedding. Then, the decoder (transformer)
is used for the reverse diffusion process.

Attributes
----------
config : dict()
    configuration infos (retrieved from configuration file)
encoder : Trajectron
    encoder for the model
diffnet : TransformerConcatLinear in this configuration
    from diffusion.py file, read documentation from there
diffusion : DiffusionTraj
    from diffusion.py, read documentation from there

Methods
-------
encode(batch, node_type) -> Tensor
    performs encoding by getting latent representation
generate(batch, node_type, num_points, sample, bestof,flexibility, ret_traj) -> ?
    generates prediction
get_loss(batch, node_type) -> number
    returns loss using get_loss method from diffusion.py file

#### Signature

```python
class AutoEncoder(Module):
    def __init__(self, config, encoder):
        ...
```

### AutoEncoder().encode

[Show source in autoencoder.py:52](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py#L52)

#### Signature

```python
def encode(self, batch, node_type):
    ...
```

### AutoEncoder().generate

[Show source in autoencoder.py:56](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py#L56)

#### Signature

```python
def generate(
    self, batch, node_type, num_points, sample, bestof, flexibility=0.0, ret_traj=False
):
    ...
```

### AutoEncoder().get_loss

[Show source in autoencoder.py:64](https://github.com/enricobu96/myMID/blob/main/models/autoencoder.py#L64)

#### Signature

```python
def get_loss(self, batch, node_type):
    ...
```


