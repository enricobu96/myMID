# SingleIntegrator

[Mymid Index](../../../README.md#mymid-index) /
[Models](../../index.md#models) /
[Encoders](../index.md#encoders) /
[Dynamics](./index.md#dynamics) /
SingleIntegrator

> Auto-generated documentation for [models.encoders.dynamics.single_integrator](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py) module.

- [SingleIntegrator](#singleintegrator)
  - [Dynamic](#dynamic)
    - [Dynamic().create_graph](#dynamic()create_graph)
    - [Dynamic().create_graph](#dynamic()create_graph-1)
    - [Dynamic().init_constants](#dynamic()init_constants)
    - [Dynamic().integrate_distribution](#dynamic()integrate_distribution)
    - [Dynamic().integrate_samples](#dynamic()integrate_samples)
    - [Dynamic().set_initial_condition](#dynamic()set_initial_condition)
  - [GMM2D](#gmm2d)
    - [GMM2D.from_log_pis_mus_cov_mats](#gmm2dfrom_log_pis_mus_cov_mats)
    - [GMM2D().get_covariance_matrix](#gmm2d()get_covariance_matrix)
    - [GMM2D().get_for_node_at_time](#gmm2d()get_for_node_at_time)
    - [GMM2D().log_prob](#gmm2d()log_prob)
    - [GMM2D().mode](#gmm2d()mode)
    - [GMM2D().reshape_to_components](#gmm2d()reshape_to_components)
    - [GMM2D().rsample](#gmm2d()rsample)
    - [GMM2D().traj_sample](#gmm2d()traj_sample)
  - [SingleIntegrator](#singleintegrator-1)
    - [SingleIntegrator().init_constants](#singleintegrator()init_constants)
    - [SingleIntegrator().integrate_distribution](#singleintegrator()integrate_distribution)
    - [SingleIntegrator().integrate_samples](#singleintegrator()integrate_samples)
  - [block_diag](#block_diag)
  - [to_one_hot](#to_one_hot)

## Dynamic

[Show source in single_integrator.py:194](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L194)

#### Signature

```python
class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, model_registrar, xz_size, node_type):
        ...
```

### Dynamic().create_graph

[Show source in single_integrator.py:212](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L212)

#### Signature

```python
def create_graph(self, xz_size):
    ...
```

### Dynamic().create_graph

[Show source in single_integrator.py:221](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L221)

#### Signature

```python
def create_graph(self, xz_size):
    ...
```

### Dynamic().init_constants

[Show source in single_integrator.py:209](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L209)

#### Signature

```python
def init_constants(self):
    ...
```

### Dynamic().integrate_distribution

[Show source in single_integrator.py:218](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L218)

#### Signature

```python
def integrate_distribution(self, dist, x):
    ...
```

### Dynamic().integrate_samples

[Show source in single_integrator.py:215](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L215)

#### Signature

```python
def integrate_samples(self, s, x):
    ...
```

### Dynamic().set_initial_condition

[Show source in single_integrator.py:205](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L205)

#### Signature

```python
def set_initial_condition(self, init_con):
    ...
```



## GMM2D

[Show source in single_integrator.py:33](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L33)

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

[Show source in single_integrator.py:82](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L82)

#### Signature

```python
@classmethod
def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
    ...
```

### GMM2D().get_covariance_matrix

[Show source in single_integrator.py:185](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L185)

#### Signature

```python
def get_covariance_matrix(self):
    ...
```

### GMM2D().get_for_node_at_time

[Show source in single_integrator.py:146](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L146)

#### Signature

```python
def get_for_node_at_time(self, n, t):
    ...
```

### GMM2D().log_prob

[Show source in single_integrator.py:119](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L119)

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

[Show source in single_integrator.py:150](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L150)

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

[Show source in single_integrator.py:180](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L180)

#### Signature

```python
def reshape_to_components(self, tensor):
    ...
```

### GMM2D().rsample

[Show source in single_integrator.py:92](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L92)

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

[Show source in single_integrator.py:113](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L113)

#### Signature

```python
def traj_sample(self, sample_shape=torch.Size()):
    ...
```



## SingleIntegrator

[Show source in single_integrator.py:226](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L226)

#### Signature

```python
class SingleIntegrator(Dynamic):
    ...
```

#### See also

- [Dynamic](#dynamic)

### SingleIntegrator().init_constants

[Show source in single_integrator.py:227](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L227)

#### Signature

```python
def init_constants(self):
    ...
```

### SingleIntegrator().integrate_distribution

[Show source in single_integrator.py:263](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L263)

Integrates the GMM velocity distribution to a distribution over position.
The Kalman Equations are used.


```\mu_{t+1} =\textbf{F} \mu_{t}
.. math:: \mathbf{\Sigma}_{t+1}={\textbf {F}} \mathbf{\Sigma}_{t} {\textbf {F}}^{T}

.. math::
    \textbf{F} = \left[
                    \begin{array}{cccc}
                        \sigma_x^2 & \rho_p \sigma_x \sigma_y & 0 & 0 \\
                        \rho_p \sigma_x \sigma_y & \sigma_y^2 & 0 & 0 \\
                        0 & 0 & \sigma_{v_x}^2 & \rho_v \sigma_{v_x} \sigma_{v_y} \\
                        0 & 0 & \rho_v \sigma_{v_x} \sigma_{v_y} & \sigma_{v_y}^2 \\
                    \end{array}
                \right]_{t}

:param v_dist: Joint GMM Distribution over velocity in x and y direction.
:param x: Not used for SI.
:return: Joint GMM Distribution over position in x and y direction.
```

#### Signature

```python
def integrate_distribution(self, v_dist, x=None):
    ...
```

### SingleIntegrator().integrate_samples

[Show source in single_integrator.py:232](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L232)

Integrates deterministic samples of velocity.

#### Arguments

- `v` - Velocity samples
- `x` - Not used for SI.

#### Returns

Position samples

#### Signature

```python
def integrate_samples(self, v, x=None):
    ...
```



## block_diag

[Show source in single_integrator.py:8](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L8)

Make a block diagonal matrix along dim=-3
EXAMPLE:
block_diag(torch.ones(4,3,2))
should give a 12 x 8 matrix with blocks of 3 x 2 ones.
Prepend batch dimensions if needed.
You can also give a list of matrices.
:type m: torch.Tensor, list

#### Returns

Type: *torch.Tensor*

#### Signature

```python
def block_diag(m):
    ...
```



## to_one_hot

[Show source in single_integrator.py:30](https://github.com/enricobu96/myMID/blob/main/models/encoders/dynamics/single_integrator.py#L30)

#### Signature

```python
def to_one_hot(labels, n_labels):
    ...
```


