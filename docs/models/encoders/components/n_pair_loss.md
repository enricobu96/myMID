# NPairLoss

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Components](./index.md#components) /
NPairLoss

> Auto-generated documentation for [models.encoders.components.n_pair_loss](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/n_pair_loss.py) module.

#### Attributes

- `N_PAIR` - Constants: `'n-pair'`


- [NPairLoss](#npairloss)
  - [NPairLoss](#npairloss-1)
    - [NPairLoss().forward](#npairloss()forward)
    - [NPairLoss.get_n_pairs](#npairlossget_n_pairs)
    - [NPairLoss.l2_loss](#npairlossl2_loss)
    - [NPairLoss.n_pair_loss](#npairlossn_pair_loss)

## NPairLoss

[Show source in n_pair_loss.py:16](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/n_pair_loss.py#L16)

N-Pair loss
Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
Processing Systems. 2016.
http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective

#### Signature

```python
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0.02, tao=1.0):
        ...
```

### NPairLoss().forward

[Show source in n_pair_loss.py:29](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/n_pair_loss.py#L29)

#### Signature

```python
def forward(self, embeddings, target):
    ...
```

### NPairLoss.get_n_pairs

[Show source in n_pair_loss.py:46](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/n_pair_loss.py#L46)

Get index of n-pairs and n-negatives

#### Arguments

- `labels` - label vector of mini-batch

#### Returns

A tuple of n_pairs (n, 2)
                and n_negatives (n, n-1)

#### Signature

```python
@staticmethod
def get_n_pairs(labels):
    ...
```

### NPairLoss.l2_loss

[Show source in n_pair_loss.py:95](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/n_pair_loss.py#L95)

Calculates L2 norm regularization loss

#### Arguments

- `anchors` - A torch.Tensor, (n, embedding_size)
- `positives` - A torch.Tensor, (n, embedding_size)

#### Returns

A scalar

#### Signature

```python
@staticmethod
def l2_loss(anchors, positives):
    ...
```

### NPairLoss.n_pair_loss

[Show source in n_pair_loss.py:76](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/n_pair_loss.py#L76)

Calculates N-Pair loss

#### Arguments

- `anchors` - A torch.Tensor, (n, embedding_size)
- `positives` - A torch.Tensor, (n, embedding_size)
- `negatives` - A torch.Tensor, (n, n-1, embedding_size)

#### Returns

A scalar

#### Signature

```python
@staticmethod
def n_pair_loss(anchors, positives, negatives, tao):
    ...
```


