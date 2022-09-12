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

[Show source in common.py:88](https://github.com/enricobu96/myMID/blob/main/models/common.py#L88)

ConcatSquashLinear layer. Not sure of which part it is or why is it used.

#### Signature

```python
class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        ...
```

### ConcatSquashLinear().forward

[Show source in common.py:98](https://github.com/enricobu96/myMID/blob/main/models/common.py#L98)

#### Signature

```python
def forward(self, ctx, x):
    ...
```



## ConcatTransformerLinear

[Show source in common.py:108](https://github.com/enricobu96/myMID/blob/main/models/common.py#L108)

Not used

#### Signature

```python
class ConcatTransformerLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        ...
```

### ConcatTransformerLinear().forward

[Show source in common.py:119](https://github.com/enricobu96/myMID/blob/main/models/common.py#L119)

#### Signature

```python
def forward(self, ctx, x):
    ...
```



## PositionalEncoding

[Show source in common.py:47](https://github.com/enricobu96/myMID/blob/main/models/common.py#L47)

PositionalEncoding class. This class accounts for the "ordinality" of the model, i.e.
the relations between space of the trajectory points.
In practice, it injects some information about the relative or absolute positions of the tokens
in the sequence. The positional encodings have the same dimension as the emberddings (so that
these can be summed).

Attributes
----------
dropout : Dropout
    just dropout hyperparam

Methods
-------
forward(x) -> net
    executes the forward pass. In practice, it should embeds stuff in the existing data x

#### Signature

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        ...
```

### PositionalEncoding().forward

[Show source in common.py:81](https://github.com/enricobu96/myMID/blob/main/models/common.py#L81)

#### Signature

```python
def forward(self, x):
    ...
```



## gaussian_entropy

[Show source in common.py:17](https://github.com/enricobu96/myMID/blob/main/models/common.py#L17)

Not used

#### Signature

```python
def gaussian_entropy(logvar):
    ...
```



## get_linear_scheduler

[Show source in common.py:130](https://github.com/enricobu96/myMID/blob/main/models/common.py#L130)

Not used

#### Signature

```python
def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    ...
```



## lr_func

[Show source in common.py:146](https://github.com/enricobu96/myMID/blob/main/models/common.py#L146)

Function which changes the learning rate w.r.t. the epoch.

#### Signature

```python
def lr_func(epoch):
    ...
```



## reparameterize_gaussian

[Show source in common.py:8](https://github.com/enricobu96/myMID/blob/main/models/common.py#L8)

Not used

#### Signature

```python
def reparameterize_gaussian(mean, logvar):
    ...
```



## standard_normal_logprob

[Show source in common.py:26](https://github.com/enricobu96/myMID/blob/main/models/common.py#L26)

Not used

#### Signature

```python
def standard_normal_logprob(z):
    ...
```



## truncated_normal_

[Show source in common.py:35](https://github.com/enricobu96/myMID/blob/main/models/common.py#L35)

Not used

#### Signature

```python
def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    ...
```


