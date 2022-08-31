# GMM2D

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Components](./index.md#components) /
GMM2D

> Auto-generated documentation for [models.encoders.components.gmm2d](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py) module.

- [GMM2D](#gmm2d)
  - [GMM2D](#gmm2d-1)
    - [GMM2D.from_log_pis_mus_cov_mats](#gmm2dfrom_log_pis_mus_cov_mats)
    - [GMM2D().get_covariance_matrix](#gmm2d()get_covariance_matrix)
    - [GMM2D().get_for_node_at_time](#gmm2d()get_for_node_at_time)
    - [GMM2D().log_prob](#gmm2d()log_prob)
    - [GMM2D().mode](#gmm2d()mode)
    - [GMM2D().reshape_to_components](#gmm2d()reshape_to_components)
    - [GMM2D().rsample](#gmm2d()rsample)
    - [GMM2D().traj_sample](#gmm2d()traj_sample)
  - [to_one_hot](#to_one_hot)

## GMM2D

[Show source in gmm2d.py:8](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L8)

Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
Cholesky decompesition and affine transformation for sampling:


```Z \sim N(0, I)
.. math:: S = \mu + LZ

.. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

where :math:`L = chol(\Sigma)` and

.. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

such that

.. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

:param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
:param mus: Mixture Components mean :math:`\mu`. [..., N * 2]
:param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * 2]
:param corrs: Cholesky factor of correlation :math:`\rho`. [..., N]
:param clip_lo: Clips the lower end of the standard deviation.
:param clip_hi: Clips the upper end of the standard deviation.
```

#### Signature

```python
class GMM2D(td.Distribution):
    def __init__(self, log_pis, mus, log_sigmas, corrs):
        ...
```

### GMM2D.from_log_pis_mus_cov_mats

[Show source in gmm2d.py:57](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L57)

#### Signature

```python
@classmethod
def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
    ...
```

### GMM2D().get_covariance_matrix

[Show source in gmm2d.py:160](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L160)

#### Signature

```python
def get_covariance_matrix(self):
    ...
```

### GMM2D().get_for_node_at_time

[Show source in gmm2d.py:121](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L121)

#### Signature

```python
def get_for_node_at_time(self, n, t):
    ...
```

### GMM2D().log_prob

[Show source in gmm2d.py:94](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L94)

Calculates the log probability of a value using the PDF for bivariate normal distributions:


```python
f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
\left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
{\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
{\sigma _{x}\sigma _{y}}}\right]\right)
```

#### Arguments

- `value` - The log probability density function is evaluated at those values.

#### Returns

Log probability

#### Signature

```python
def log_prob(self, value):
    ...
```

### GMM2D().mode

[Show source in gmm2d.py:125](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L125)

Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

#### Arguments

- `required_accuracy` - Accuracy of the meshgrid

#### Returns

Mode of the GMM

#### Signature

```python
def mode(self):
    ...
```

### GMM2D().reshape_to_components

[Show source in gmm2d.py:155](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L155)

#### Signature

```python
def reshape_to_components(self, tensor):
    ...
```

### GMM2D().rsample

[Show source in gmm2d.py:67](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L67)

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

#### Arguments

- `sample_shape` - Shape of the samples

#### Returns

Samples from the GMM.

#### Signature

```python
def rsample(self, sample_shape=torch.Size()):
    ...
```

### GMM2D().traj_sample

[Show source in gmm2d.py:88](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L88)

#### Signature

```python
def traj_sample(self, sample_shape=torch.Size()):
    ...
```



## to_one_hot

[Show source in gmm2d.py:5](https://github.com/enricobu96/myMID/blob/main/models/encoders/components/gmm2d.py#L5)

#### Signature

```python
def to_one_hot(labels, n_labels):
    ...
```


