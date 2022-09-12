# DiscreteLatent

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Components](./index.md#components) /
DiscreteLatent

> Auto-generated documentation for [models.encoders.components.discrete_latent](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py) module.

- [DiscreteLatent](#discretelatent)
  - [DiscreteLatent](#discretelatent-1)
    - [DiscreteLatent.all_one_hot_combinations](#discretelatentall_one_hot_combinations)
    - [DiscreteLatent().dist_from_h](#discretelatent()dist_from_h)
    - [DiscreteLatent().get_p_dist_probs](#discretelatent()get_p_dist_probs)
    - [DiscreteLatent().kl_q_p](#discretelatent()kl_q_p)
    - [DiscreteLatent().p_log_prob](#discretelatent()p_log_prob)
    - [DiscreteLatent().q_log_prob](#discretelatent()q_log_prob)
    - [DiscreteLatent().sample_p](#discretelatent()sample_p)
    - [DiscreteLatent().sample_q](#discretelatent()sample_q)
    - [DiscreteLatent().summarize_for_tensorboard](#discretelatent()summarize_for_tensorboard)
  - [ModeKeys](#modekeys)

## DiscreteLatent

[Show source in discrete_latent.py:12](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L12)

#### Signature

```python
class DiscreteLatent(object):
    def __init__(self, hyperparams, device):
        ...
```

### DiscreteLatent.all_one_hot_combinations

[Show source in discrete_latent.py:100](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L100)

#### Signature

```python
@staticmethod
def all_one_hot_combinations(N, K):
    ...
```

### DiscreteLatent().dist_from_h

[Show source in discrete_latent.py:25](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L25)

#### Signature

```python
def dist_from_h(self, h, mode):
    ...
```

### DiscreteLatent().get_p_dist_probs

[Show source in discrete_latent.py:97](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L97)

#### Signature

```python
def get_p_dist_probs(self):
    ...
```

### DiscreteLatent().kl_q_p

[Show source in discrete_latent.py:69](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L69)

#### Signature

```python
def kl_q_p(self, log_writer=None, prefix=None, curr_iter=None):
    ...
```

### DiscreteLatent().p_log_prob

[Show source in discrete_latent.py:92](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L92)

#### Signature

```python
def p_log_prob(self, z):
    ...
```

### DiscreteLatent().q_log_prob

[Show source in discrete_latent.py:87](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L87)

#### Signature

```python
def q_log_prob(self, z):
    ...
```

### DiscreteLatent().sample_p

[Show source in discrete_latent.py:42](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L42)

#### Signature

```python
def sample_p(
    self, num_samples, mode, most_likely_z=False, full_dist=True, all_z_sep=False
):
    ...
```

### DiscreteLatent().sample_q

[Show source in discrete_latent.py:36](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L36)

#### Signature

```python
def sample_q(self, num_samples, mode):
    ...
```

### DiscreteLatent().summarize_for_tensorboard

[Show source in discrete_latent.py:104](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L104)

#### Signature

```python
def summarize_for_tensorboard(self, log_writer, prefix, curr_iter):
    ...
```



## ModeKeys

[Show source in discrete_latent.py:7](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/discrete_latent.py#L7)

#### Signature

```python
class ModeKeys(Enum):
    ...
```


