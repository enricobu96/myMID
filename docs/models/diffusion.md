# Diffusion

[Mymid Index](../README.md#mymid-index) /
[Models](./index.md#models) /
Diffusion

> Auto-generated documentation for [models.diffusion](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py) module.

- [Diffusion](#diffusion)
  - [DiffusionTraj](#diffusiontraj)
    - [DiffusionTraj().get_loss](#diffusiontraj()get_loss)
    - [DiffusionTraj().sample](#diffusiontraj()sample)
  - [LinearDecoder](#lineardecoder)
    - [LinearDecoder().forward](#lineardecoder()forward)
  - [TrajNet](#trajnet)
    - [TrajNet().forward](#trajnet()forward)
  - [TransformerConcatLinear](#transformerconcatlinear)
    - [TransformerConcatLinear().forward](#transformerconcatlinear()forward)
  - [TransformerLinear](#transformerlinear)
    - [TransformerLinear().forward](#transformerlinear()forward)
  - [VarianceSchedule](#varianceschedule)
    - [VarianceSchedule().get_sigmas](#varianceschedule()get_sigmas)
    - [VarianceSchedule().uniform_sample_t](#varianceschedule()uniform_sample_t)

## DiffusionTraj

[Show source in diffusion.py:200](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L200)

#### Signature

```python
class DiffusionTraj(Module):
    def __init__(self, net, var_sched: VarianceSchedule):
        ...
```

#### See also

- [VarianceSchedule](#varianceschedule)

### DiffusionTraj().get_loss

[Show source in diffusion.py:207](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L207)

#### Signature

```python
def get_loss(self, x_0, context, t=None):
    ...
```

### DiffusionTraj().sample

[Show source in diffusion.py:226](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L226)

#### Signature

```python
def sample(
    self,
    num_points,
    context,
    sample,
    bestof,
    point_dim=2,
    flexibility=0.0,
    ret_traj=False,
):
    ...
```



## LinearDecoder

[Show source in diffusion.py:173](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L173)

#### Signature

```python
class LinearDecoder(Module):
    def __init__(self):
        ...
```

### LinearDecoder().forward

[Show source in diffusion.py:189](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L189)

#### Signature

```python
def forward(self, code):
    ...
```



## TrajNet

[Show source in diffusion.py:60](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L60)

#### Signature

```python
class TrajNet(Module):
    def __init__(self, point_dim, context_dim, residual):
        ...
```

### TrajNet().forward

[Show source in diffusion.py:76](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L76)

#### Arguments

- `x` - Point clouds at some timestep t, (B, N, d).
- `beta` - Time. (B, ).
- `context` - Shape latents. (B, F).

#### Signature

```python
def forward(self, x, beta, context):
    ...
```



## TransformerConcatLinear

[Show source in diffusion.py:103](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L103)

#### Signature

```python
class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        ...
```

### TransformerConcatLinear().forward

[Show source in diffusion.py:118](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L118)

#### Signature

```python
def forward(self, x, beta, context):
    ...
```



## TransformerLinear

[Show source in diffusion.py:135](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L135)

#### Signature

```python
class TransformerLinear(Module):
    def __init__(self, point_dim, context_dim, residual):
        ...
```

### TransformerLinear().forward

[Show source in diffusion.py:148](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L148)

#### Signature

```python
def forward(self, x, beta, context):
    ...
```



## VarianceSchedule

[Show source in diffusion.py:9](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L9)

#### Signature

```python
class VarianceSchedule(Module):
    def __init__(
        self, num_steps, mode="linear", beta_1=0.0001, beta_T=0.05, cosine_s=0.008
    ):
        ...
```

### VarianceSchedule().get_sigmas

[Show source in diffusion.py:55](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L55)

#### Signature

```python
def get_sigmas(self, t, flexibility):
    ...
```

### VarianceSchedule().uniform_sample_t

[Show source in diffusion.py:51](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L51)

#### Signature

```python
def uniform_sample_t(self, batch_size):
    ...
```


