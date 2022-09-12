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

[Show source in diffusion.py:269](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L269)

DiffusionTraj class, used as diffusion model for trajectories (crucial part of the project). This
contains in turn the net (in this case TransformerConcatLinear) and the variance schedule.

#### Signature

```python
class DiffusionTraj(Module):
    def __init__(self, net, var_sched: VarianceSchedule):
        ...
```

#### See also

- [VarianceSchedule](#varianceschedule)

### DiffusionTraj().get_loss

[Show source in diffusion.py:279](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L279)

#### Signature

```python
def get_loss(self, x_0, context, t=None):
    ...
```

### DiffusionTraj().sample

[Show source in diffusion.py:298](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L298)

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

[Show source in diffusion.py:238](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L238)

LinearDecoder class, apparently not used in this project; TransformerConcatLinear is used instead. # TODO test MID with
this model instead of the former. However, this was probably used for ablation studies (Section 4.3 MID paper).

#### Signature

```python
class LinearDecoder(Module):
    def __init__(self):
        ...
```

### LinearDecoder().forward

[Show source in diffusion.py:258](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L258)

#### Signature

```python
def forward(self, code):
    ...
```



## TrajNet

[Show source in diffusion.py:93](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L93)

TrajNet class, apparently not used in this project; TransformerConcatLinear is used instead. # TODO test MID with
this model instead of the former. However, this was probably used for ablation studies (Section 4.3 MID paper).

#### Signature

```python
class TrajNet(Module):
    def __init__(self, point_dim, context_dim, residual):
        ...
```

### TrajNet().forward

[Show source in diffusion.py:112](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L112)

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

[Show source in diffusion.py:139](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L139)

TransformerConcatLinear class. This is a very crucial part of the project, since it's the Transformer
model used for the decoding part. It inherits from torch module, i.e. the init method creates the network
(and it therefore represents the decoder network itself), and the forward method represents a forward pass
in the model.

Attributes
----------
residual : bool
    wether to use residual connections (?). Set to False in this project
pos_emb : PositionalEncoding
    positional encoding layer. Read documentation of the class, but the idea is that
    this manages the "ordinality" of states
concat1 : ConcatSquashLinear
    concat squash linear layer. Read documentation of the class (it's a custom
    class, it doesn't come with standard pytorch)
layer : TransformerEncoderLayer
    encoder layer made up of self-attention and a ffnn. Based on the original
    transformer paper
transformer_encoder : TransformerEncoder
    stack of n encoder layers
concat3 : ConcatSquashLinear
    same as above
concat4 : ConcatSquashLinear
    same as above
linear : ConcatSquashLinear
    same as above

Methods
-------
forward(x, beta, context) -> nn.Linear
    forward pass for the decoder model

#### Signature

```python
class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        ...
```

### TransformerConcatLinear().forward

[Show source in diffusion.py:186](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L186)

#### Signature

```python
def forward(self, x, beta, context):
    ...
```



## TransformerLinear

[Show source in diffusion.py:203](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L203)

TransformerLinear class, apparently not used in this project; TransformerConcatLinear is used instead. # TODO test MID with
this model instead of the former. However, this was probably used for ablation studies (Section 4.3 MID paper).

#### Signature

```python
class TransformerLinear(Module):
    def __init__(self, point_dim, context_dim, residual):
        ...
```

### TransformerLinear().forward

[Show source in diffusion.py:219](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L219)

#### Signature

```python
def forward(self, x, beta, context):
    ...
```



## VarianceSchedule

[Show source in diffusion.py:9](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L9)

Class representing the variance schedule.

Attributes
----------
num_steps : int
    number of steps in the variance schedule
mode : str
    type of variance schedule: linear or cosine
beta : number
    hyperparameter beta
beta_T : number
    hyperparameter betaT
cosine_s : number
    hyperparameter s for cosine variance schedule

Methods
-------
uniform_sample_t(batch_size) -> list()
    returns a uniform distribution with range [1;number_of_steps+1]
get_sigmas(t, flexibility) -> list()
    returns sigmas

Detailed description
--------------------
When initialized a VarianceSchedule object, the following happens:
0. Attributes are initialized
1. A schedule is decided and the betas are then generated:
    1.1. If the schedule is linear, betas are just a linear space between beta1 and betaT.
    The number of steps is the one given at initialization; betas are in a tensor
    1.2. If the schedule is cosine, uses formula 17 from Improved DDPM paper (https://arxiv.org/pdf/2102.09672.pdf) to compute betas and (first part) alphas
2. Pads the betas in order to match dimensions
3. Computes alphas, alpha_logs and sigmas

#### Signature

```python
class VarianceSchedule(Module):
    def __init__(
        self, num_steps, mode="linear", beta_1=0.0001, beta_T=0.05, cosine_s=0.008
    ):
        ...
```

### VarianceSchedule().get_sigmas

[Show source in diffusion.py:88](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L88)

#### Signature

```python
def get_sigmas(self, t, flexibility):
    ...
```

### VarianceSchedule().uniform_sample_t

[Show source in diffusion.py:84](https://github.com/enricobu96/myMID/blob/main/models/diffusion.py#L84)

#### Signature

```python
def uniform_sample_t(self, batch_size):
    ...
```


