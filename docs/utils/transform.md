# Transform

[Mymid Index](../README.md#mymid-index) /
[Utils](./index.md#utils) /
Transform

> Auto-generated documentation for [utils.transform](https://github.com/enricobu96/myMID/blob/main/utils/transform.py) module.

- [Transform](#transform)
  - [AddNoise](#addnoise)
  - [AddNoiseForEval](#addnoiseforeval)
  - [AddRandomNoise](#addrandomnoise)
  - [Center](#center)
  - [FixedPoints](#fixedpoints)
  - [IdentityTransform](#identitytransform)
  - [LinearTransformation](#lineartransformation)
  - [NormalizeScale](#normalizescale)
  - [RandomRotate](#randomrotate)
  - [RandomScale](#randomscale)
  - [RandomTranslate](#randomtranslate)
  - [Rotate](#rotate)

## AddNoise

[Show source in transform.py:152](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L152)

#### Signature

```python
class AddNoise(object):
    def __init__(self, std=0.01, noiseless_item_key="clean"):
        ...
```



## AddNoiseForEval

[Show source in transform.py:177](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L177)

#### Signature

```python
class AddNoiseForEval(object):
    def __init__(self, stds=[0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15]):
        ...
```



## AddRandomNoise

[Show source in transform.py:164](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L164)

#### Signature

```python
class AddRandomNoise(object):
    def __init__(self, std_range=[0, 0.1], noiseless_item_key="clean"):
        ...
```



## Center

[Show source in transform.py:10](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L10)

Centers node positions around the origin.

#### Signature

```python
class Center(object):
    def __init__(self, attr):
        ...
```



## FixedPoints

[Show source in transform.py:43](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L43)

Samples a fixed number of :obj:`num` points and features from a point
cloud.

#### Arguments

- `num` *int* - The number of points to sample.
- `replace` *bool, optional* - If set to :obj:`False`, samples fixed
    points without replacement. In case :obj:`num` is greater than
    the number of points, duplicated points are kept to a
    - `minimum.` *(default* - :obj:`True`)

#### Signature

```python
class FixedPoints(object):
    def __init__(self, num, replace=True):
        ...
```



## IdentityTransform

[Show source in transform.py:190](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L190)

#### Signature

```python
class IdentityTransform(object):
    ...
```



## LinearTransformation

[Show source in transform.py:82](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L82)

Transforms node positions with a square transformation matrix computed
offline.

#### Arguments

- `matrix` *Tensor* - tensor with shape :math:`[D, D]` where :math:`D`
    corresponds to the dimensionality of node positions.

#### Signature

```python
class LinearTransformation(object):
    def __init__(self, matrix, attr):
        ...
```



## NormalizeScale

[Show source in transform.py:25](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L25)

Centers and normalizes node positions to the interval :math:`(-1, 1)`.

#### Signature

```python
class NormalizeScale(object):
    def __init__(self, attr):
        ...
```



## RandomRotate

[Show source in transform.py:116](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L116)

Rotates node positions around a specific axis by a randomly sampled
factor within a given interval.

#### Arguments

degrees (tuple or float): Rotation interval from which the rotation
    angle is sampled. If :obj:`degrees` is a number instead of a
    tuple, the interval is given by :math:`[-\mathrm{degrees},
    \mathrm{degrees}]`.
- `axis` *int, optional* - The rotation axis. (default: :obj:`0`)

#### Signature

```python
class RandomRotate(object):
    def __init__(self, degrees, attr, axis=0):
        ...
```



## RandomScale

[Show source in transform.py:196](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L196)

Scales node positions by a randomly sampled factor :math:`s` within a
given interval, *e.g.*, resulting in the transformation matrix
.. math

```python
\begin{bmatrix}
    s & 0 & 0 \\
    0 & s & 0 \\
    0 & 0 & s \\
\end{bmatrix}
```

for three-dimensional positions.

#### Arguments

- `scales` *tuple* - scaling factor interval, e.g. :obj:`(a, b)`, then scale
    is randomly sampled from the range
    :math:`a \leq \mathrm{scale} \leq b`.

#### Signature

```python
class RandomScale(object):
    def __init__(self, scales, attr):
        ...
```



## RandomTranslate

[Show source in transform.py:227](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L227)

Translates node positions by randomly sampled translation values
within a given interval. In contrast to other random transformations,
translation is applied separately at each position.

#### Arguments

translate (sequence or float or int): Maximum translation in each
    dimension, defining the range
    :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
    If :obj:`translate` is a number instead of a sequence, the same
    range is used for each dimension.

#### Signature

```python
class RandomTranslate(object):
    def __init__(self, translate, attr):
        ...
```



## Rotate

[Show source in transform.py:262](https://github.com/enricobu96/myMID/blob/main/utils/transform.py#L262)

Rotates node positions around a specific axis by a randomly sampled
factor within a given interval.

#### Arguments

degrees (tuple or float): Rotation interval from which the rotation
    angle is sampled. If :obj:`degrees` is a number instead of a
    tuple, the interval is given by :math:`[-\mathrm{degrees},
    \mathrm{degrees}]`.
- `axis` *int, optional* - The rotation axis. (default: :obj:`0`)

#### Signature

```python
class Rotate(object):
    def __init__(self, degree, attr, axis=0):
        ...
```


