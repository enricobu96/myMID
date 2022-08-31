# Common

[Mymid Index](../README.md#mymid-index) /
[Models](./index.md#models) /
Common

> Auto-generated documentation for [models.common](https://github.com/enricobu96/myMID/blob/main/models/common.py) module.

- [Common](#common)
  - [ConcatSquashLinear](#concatsquashlinear)
    - [ConcatSquashLinear().forward](#concatsquashlinear()forward)
  - [ConcatTransformerLinear](#concattransformerlinear)
    - [ConcatTransformerLinear().forward](#concattransformerlinear()forward)
  - [PositionalEncoding](#positionalencoding)
    - [PositionalEncoding().forward](#positionalencoding()forward)
  - [gaussian_entropy](#gaussian_entropy)
  - [get_linear_scheduler](#get_linear_scheduler)
  - [lr_func](#lr_func)
  - [reparameterize_gaussian](#reparameterize_gaussian)
  - [standard_normal_logprob](#standard_normal_logprob)
  - [truncated_normal_](#truncated_normal_)

## ConcatSquashLinear

[Show source in common.py:58](https://github.com/enricobu96/myMID/blob/main/models/common.py#L58)

#### Signature

```python
class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        ...
```

### ConcatSquashLinear().forward

[Show source in common.py:65](https://github.com/enricobu96/myMID/blob/main/models/common.py#L65)

#### Signature

```python
def forward(self, ctx, x):
    ...
```



## ConcatTransformerLinear

[Show source in common.py:75](https://github.com/enricobu96/myMID/blob/main/models/common.py#L75)

#### Signature

```python
class ConcatTransformerLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        ...
```

### ConcatTransformerLinear().forward

[Show source in common.py:83](https://github.com/enricobu96/myMID/blob/main/models/common.py#L83)

#### Signature

```python
def forward(self, ctx, x):
    ...
```



## PositionalEncoding

[Show source in common.py:35](https://github.com/enricobu96/myMID/blob/main/models/common.py#L35)

#### Signature

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        ...
```

### PositionalEncoding().forward

[Show source in common.py:51](https://github.com/enricobu96/myMID/blob/main/models/common.py#L51)

#### Signature

```python
def forward(self, x):
    ...
```



## gaussian_entropy

[Show source in common.py:14](https://github.com/enricobu96/myMID/blob/main/models/common.py#L14)

#### Signature

```python
def gaussian_entropy(logvar):
    ...
```



## get_linear_scheduler

[Show source in common.py:94](https://github.com/enricobu96/myMID/blob/main/models/common.py#L94)

#### Signature

```python
def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    ...
```



## lr_func

[Show source in common.py:107](https://github.com/enricobu96/myMID/blob/main/models/common.py#L107)

#### Signature

```python
def lr_func(epoch):
    ...
```



## reparameterize_gaussian

[Show source in common.py:8](https://github.com/enricobu96/myMID/blob/main/models/common.py#L8)

#### Signature

```python
def reparameterize_gaussian(mean, logvar):
    ...
```



## standard_normal_logprob

[Show source in common.py:20](https://github.com/enricobu96/myMID/blob/main/models/common.py#L20)

#### Signature

```python
def standard_normal_logprob(z):
    ...
```



## truncated_normal_

[Show source in common.py:26](https://github.com/enricobu96/myMID/blob/main/models/common.py#L26)

#### Signature

```python
def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    ...
```


