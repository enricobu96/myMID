# Transformer

[Mymid Index](../README.md#mymid-index) /
[Models](./index.md#models) /
Transformer

> Auto-generated documentation for [models.transformer](https://github.com/enricobu96/myMID/blob/main/models/transformer.py) module.

- [Transformer](#transformer)
  - [Decoder_TRANSFORMER](#decoder_transformer)
    - [Decoder_TRANSFORMER().forward](#decoder_transformer()forward)
  - [Encoder_TRANSFORMER](#encoder_transformer)
    - [Encoder_TRANSFORMER().forward](#encoder_transformer()forward)
  - [PositionalEncoding](#positionalencoding)
    - [PositionalEncoding().forward](#positionalencoding()forward)
  - [TimeEncoding](#timeencoding)
    - [TimeEncoding().forward](#timeencoding()forward)

## Decoder_TRANSFORMER

[Show source in transformer.py:131](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L131)

#### Signature

```python
class Decoder_TRANSFORMER(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_frames,
        num_classes,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        ablation=None,
        **kargs
    ):
        ...
```

### Decoder_TRANSFORMER().forward

[Show source in transformer.py:183](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L183)

#### Signature

```python
def forward(self, batch):
    ...
```



## Encoder_TRANSFORMER

[Show source in transformer.py:42](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L42)

#### Signature

```python
class Encoder_TRANSFORMER(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_frames,
        num_classes,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        **kargs
    ):
        ...
```

### Encoder_TRANSFORMER().forward

[Show source in transformer.py:92](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L92)

#### Signature

```python
def forward(self, batch):
    ...
```



## PositionalEncoding

[Show source in transformer.py:7](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L7)

#### Signature

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        ...
```

### PositionalEncoding().forward

[Show source in transformer.py:21](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L21)

#### Signature

```python
def forward(self, x):
    ...
```



## TimeEncoding

[Show source in transformer.py:28](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L28)

#### Signature

```python
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        ...
```

### TimeEncoding().forward

[Show source in transformer.py:33](https://github.com/enricobu96/myMID/blob/main/models/transformer.py#L33)

#### Signature

```python
def forward(self, x, mask, lengths):
    ...
```


