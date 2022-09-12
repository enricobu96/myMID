# Graph Attention

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Components](./index.md#components) /
Graph Attention

> Auto-generated documentation for [models.encoders.components.graph_attention](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/graph_attention.py) module.

- [Graph Attention](#graph-attention)
  - [GraphMultiTypeAttention](#graphmultitypeattention)
    - [GraphMultiTypeAttention().extra_repr](#graphmultitypeattention()extra_repr)
    - [GraphMultiTypeAttention().forward](#graphmultitypeattention()forward)
    - [GraphMultiTypeAttention().reset_parameters](#graphmultitypeattention()reset_parameters)

## GraphMultiTypeAttention

[Show source in graph_attention.py:10](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/graph_attention.py#L10)

#### Signature

```python
class GraphMultiTypeAttention(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True, types=1):
        ...
```

### GraphMultiTypeAttention().extra_repr

[Show source in graph_attention.py:55](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/graph_attention.py#L55)

#### Signature

```python
def extra_repr(self):
    ...
```

### GraphMultiTypeAttention().forward

[Show source in graph_attention.py:39](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/graph_attention.py#L39)

#### Signature

```python
def forward(self, inputs, types, edge_weights):
    ...
```

### GraphMultiTypeAttention().reset_parameters

[Show source in graph_attention.py:30](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/graph_attention.py#L30)

#### Signature

```python
def reset_parameters(self):
    ...
```


