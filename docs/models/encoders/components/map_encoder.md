# Map Encoder

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Components](./index.md#components) /
Map Encoder

> Auto-generated documentation for [models.encoders.components.map_encoder](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/map_encoder.py) module.

- [Map Encoder](#map-encoder)
  - [CNNMapEncoder](#cnnmapencoder)
    - [CNNMapEncoder().forward](#cnnmapencoder()forward)

## CNNMapEncoder

[Show source in map_encoder.py:6](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/map_encoder.py#L6)

#### Signature

```python
class CNNMapEncoder(nn.Module):
    def __init__(
        self, map_channels, hidden_channels, output_size, masks, strides, patch_size
    ):
        ...
```

### CNNMapEncoder().forward

[Show source in map_encoder.py:23](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/map_encoder.py#L23)

#### Signature

```python
def forward(self, x, training):
    ...
```


