# AdditiveAttention

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Components](./index.md#components) /
AdditiveAttention

> Auto-generated documentation for [models.encoders.components.additive_attention](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/additive_attention.py) module.

- [AdditiveAttention](#additiveattention)
  - [AdditiveAttention](#additiveattention-1)
    - [AdditiveAttention().forward](#additiveattention()forward)
    - [AdditiveAttention().score](#additiveattention()score)
  - [TemporallyBatchedAdditiveAttention](#temporallybatchedadditiveattention)
    - [TemporallyBatchedAdditiveAttention().forward](#temporallybatchedadditiveattention()forward)
    - [TemporallyBatchedAdditiveAttention().score](#temporallybatchedadditiveattention()score)

## AdditiveAttention

[Show source in additive_attention.py:6](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/additive_attention.py#L6)

#### Signature

```python
class AdditiveAttention(nn.Module):
    def __init__(
        self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None
    ):
        ...
```

### AdditiveAttention().forward

[Show source in additive_attention.py:25](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/additive_attention.py#L25)

#### Signature

```python
def forward(self, encoder_states, decoder_state):
    ...
```

### AdditiveAttention().score

[Show source in additive_attention.py:19](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/additive_attention.py#L19)

#### Signature

```python
def score(self, encoder_state, decoder_state):
    ...
```



## TemporallyBatchedAdditiveAttention

[Show source in additive_attention.py:41](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/additive_attention.py#L41)

#### Signature

```python
class TemporallyBatchedAdditiveAttention(AdditiveAttention):
    def __init__(
        self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None
    ):
        ...
```

#### See also

- [AdditiveAttention](#additiveattention)

### TemporallyBatchedAdditiveAttention().forward

[Show source in additive_attention.py:55](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/additive_attention.py#L55)

#### Signature

```python
def forward(self, encoder_states, decoder_state):
    ...
```

### TemporallyBatchedAdditiveAttention().score

[Show source in additive_attention.py:49](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/additive_attention.py#L49)

#### Signature

```python
def score(self, encoder_state, decoder_state):
    ...
```


