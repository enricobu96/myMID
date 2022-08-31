# HomographyWarper

[Mymid Index](../README.md#mymid-index) /
[Dataset](./index.md#dataset) /
HomographyWarper

> Auto-generated documentation for [dataset.homography_warper](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py) module.

- [HomographyWarper](#homographywarper)
  - [HomographyWarper](#homographywarper-1)
    - [HomographyWarper().forward](#homographywarper()forward)
    - [HomographyWarper().warp_grid](#homographywarper()warp_grid)
  - [angle_to_rotation_matrix](#angle_to_rotation_matrix)
  - [convert_points_from_homogeneous](#convert_points_from_homogeneous)
  - [convert_points_to_homogeneous](#convert_points_to_homogeneous)
  - [create_batched_meshgrid](#create_batched_meshgrid)
  - [deg2rad](#deg2rad)
  - [get_rotation_matrix2d](#get_rotation_matrix2d)
  - [homography_warp](#homography_warp)
  - [multi_linspace](#multi_linspace)
  - [normal_transform_pixel](#normal_transform_pixel)
  - [src_norm_to_dst_norm](#src_norm_to_dst_norm)
  - [transform_points](#transform_points)
  - [transform_warp_impl](#transform_warp_impl)
  - [warp_affine_crop](#warp_affine_crop)

## HomographyWarper

[Show source in homography_warper.py:332](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L332)

Warps image patches or tensors by homographies.
.. math

```python
X_{dst} = H_{src}^{\{dst\}} * X_{src}
```

#### Arguments

- `height` *int* - The height of the image to warp.
- `width` *int* - The width of the image to warp.
- `mode` *str* - interpolation mode to calculate output values
  'bilinear' | 'nearest'. Default: 'bilinear'.
- `padding_mode` *str* - padding mode for outside grid values
  'zeros' | 'border' | 'reflection'. Default: 'zeros'.

#### Signature

```python
class HomographyWarper(nn.Module):
    def __init__(
        self,
        x_min: torch.Tensor,
        y_min: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
        height: int,
        width: int,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        ...
```

### HomographyWarper().forward

[Show source in homography_warper.py:386](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L386)

Warps an image or tensor from source into reference frame.

#### Arguments

- `patch_src` *torch.Tensor* - The image or tensor to warp.
                          Should be from source.
- `dst_homo_src` *torch.Tensor* - The homography or stack of homographies
 from source to destination. The homography assumes normalized
 coordinates [-1, 1].

#### Returns

    - `torch.Tensor` - Patch sampled at locations from source to destination.
Shape:
    - `-` *Input* - :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
    - `-` *Output* - :math:`(N, C, H, W)`

#### Examples

```python
>>> input = torch.rand(1, 3, 32, 32)
>>> homography = torch.eye(3).view(1, 3, 3)
>>> warper = kornia.HomographyWarper(32, 32)
>>> output = warper(input, homography)  # NxCxHxW
```

#### Signature

```python
def forward(self, patch_src: torch.Tensor, dst_homo_src: torch.Tensor) -> torch.Tensor:
    ...
```

### HomographyWarper().warp_grid

[Show source in homography_warper.py:364](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L364)

Computes the grid to warp the coordinates grid by an homography.

#### Arguments

- `dst_homo_src` *torch.Tensor* - Homography or homographies (stacked) to
                  transform all points in the grid. Shape of the
                  homography has to be :math:`(N, 3, 3)`.

#### Returns

- `torch.Tensor` - the transformed grid of shape :math:`(N, H, W, 2)`.

#### Signature

```python
def warp_grid(self, dst_homo_src: torch.Tensor) -> torch.Tensor:
    ...
```



## angle_to_rotation_matrix

[Show source in homography_warper.py:24](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L24)

Creates a rotation matrix out of angles in degrees

#### Arguments

- `angle` - (torch.Tensor): tensor of angles in degrees, any shape.

#### Returns

    - `torch.Tensor` - tensor of *x2x2 rotation matrices.
Shape:
    - `-` *Input* - :math:`(*)`
    - `-` *Output* - :math:`(*, 2, 2)`

#### Examples

```python
>>> input = torch.rand(1, 3)  # Nx3
>>> output = kornia.angle_to_rotation_matrix(input)  # Nx3x2x2
```

#### Signature

```python
def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    ...
```



## convert_points_from_homogeneous

[Show source in homography_warper.py:137](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L137)

Function that converts points from homogeneous to Euclidean space.
Examples

```python
>>> input = torch.rand(2, 4, 3)  # BxNx3
>>> output = kornia.convert_points_from_homogeneous(input)  # BxNx2
```

#### Signature

```python
def convert_points_from_homogeneous(
    points: torch.Tensor, eps: float = 1e-08
) -> torch.Tensor:
    ...
```



## convert_points_to_homogeneous

[Show source in homography_warper.py:121](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L121)

Function that converts points from Euclidean to homogeneous space.
Examples

```python
>>> input = torch.rand(2, 4, 3)  # BxNx3
>>> output = kornia.convert_points_to_homogeneous(input)  # BxNx4
```

#### Signature

```python
def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    ...
```



## create_batched_meshgrid

[Show source in homography_warper.py:213](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L213)

Generates a coordinate grid for an image.
When the flag `normalized_coordinates` is set to True, the grid is
normalized to be in the range [-1,1] to be consistent with the pytorch
function grid_sample.
http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

#### Arguments

- `height` *int* - the image height (rows).
- `width` *int* - the image width (cols).
- `normalized_coordinates` *Optional[bool]* - whether to normalize
  coordinates in the range [-1, 1] in order to be consistent with the
  PyTorch function grid_sample.

#### Returns

- `torch.Tensor` - returns a grid tensor with shape :math:`(1, H, W, 2)`.

#### Signature

```python
def create_batched_meshgrid(
    x_min: torch.Tensor,
    y_min: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
    height: int,
    width: int,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    ...
```



## deg2rad

[Show source in homography_warper.py:10](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L10)

Function that converts angles from degrees to radians.

#### Arguments

- `tensor` *torch.Tensor* - Tensor of arbitrary shape.

#### Returns

- `torch.Tensor` - tensor with same shape as input.

#### Signature

```python
def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    ...
```



## get_rotation_matrix2d

[Show source in homography_warper.py:44](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L44)

Calculates an affine matrix of 2D rotation.
The function calculates the following matrix:
.. math

```python
\begin{bmatrix}
    \alpha & \beta & (1 - \alpha) \cdot \text{x}
    - \beta \cdot \text{y} \\
    -\beta & \alpha & \beta \cdot \text{x}
    + (1 - \alpha) \cdot \text{y}
\end{bmatrix}
```

where
.. math

```python
\alpha = \text{scale} \cdot cos(\text{radian}) \\
\beta = \text{scale} \cdot sin(\text{radian})
```

The transformation maps the rotation center to itself
If this is not the target, adjust the shift.

#### Arguments

- `center` *Tensor* - center of the rotation in the source image.
- `angle` *Tensor* - rotation radian in degrees. Positive values mean
    counter-clockwise rotation (the coordinate origin is assumed to
    be the top-left corner).
- `scale` *Tensor* - isotropic scale factor.

#### Returns

    - `Tensor` - the affine matrix of 2D rotation.
Shape:
    - `-` *Input* - :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
    - `-` *Output* - :math:`(B, 2, 3)`

#### Examples

```python
>>> center = torch.zeros(1, 2)
>>> scale = torch.ones(1)
>>> radian = 45. * torch.ones(1)
>>> M = kornia.get_rotation_matrix2d(center, radian, scale)
tensor([[[ 0.7071,  0.7071,  0.0000],
         [-0.7071,  0.7071,  0.0000]]])
```

#### Signature

```python
def get_rotation_matrix2d(
    center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    ...
```



## homography_warp

[Show source in homography_warper.py:248](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L248)

Function that warps image patchs or tensors by homographies.
See :class:`~kornia.geometry.warp.HomographyWarper` for details.

#### Arguments

- `patch_src` *torch.Tensor* - The image or tensor to warp. Should be from
                          source of shape :math:`(N, C, H, W)`.
- `dst_homo_src` *torch.Tensor* - The homography or stack of homographies
                             from source to destination of shape
                             :math:`(N, 3, 3)`.
dsize (Tuple[int, int]): The height and width of the image to warp.
- `mode` *str* - interpolation mode to calculate output values
  'bilinear' | 'nearest'. Default: 'bilinear'.
- `padding_mode` *str* - padding mode for outside grid values
  'zeros' | 'border' | 'reflection'. Default: 'zeros'.

#### Returns

- `torch.Tensor` - Patch sampled at locations from source to destination.

#### Examples

```python
>>> input = torch.rand(1, 3, 32, 32)
>>> homography = torch.eye(3).view(1, 3, 3)
>>> output = kornia.homography_warp(input, homography, (32, 32))
```

#### Signature

```python
def homography_warp(
    patch_src: torch.Tensor,
    centers: torch.Tensor,
    dst_homo_src: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    ...
```



## multi_linspace

[Show source in homography_warper.py:199](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L199)

This function is just like np.linspace, but will create linearly
spaced vectors from a start to end vector.
Inputs:
    a - Start vector.
    b - End vector.
    num - Number of samples to generate. Default is 50. Must be above 0.
    endpoint - If True, b is the last sample.
               Otherwise, it is not included. Default is True.

#### Signature

```python
def multi_linspace(a, b, num, endpoint=True, device="cpu", dtype=torch.float):
    ...
```



## normal_transform_pixel

[Show source in homography_warper.py:285](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L285)

#### Signature

```python
def normal_transform_pixel(height, width):
    ...
```



## src_norm_to_dst_norm

[Show source in homography_warper.py:299](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L299)

#### Signature

```python
def src_norm_to_dst_norm(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int],
    dsize_dst: Tuple[int, int],
) -> torch.Tensor:
    ...
```



## transform_points

[Show source in homography_warper.py:164](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L164)

Function that applies transformations to a set of points.

#### Arguments

- `trans_01` *torch.Tensor* - tensor for transformations of shape
  :math:`(B, D+1, D+1)`.
- `points_1` *torch.Tensor* - tensor of points of shape :math:`(B, N, D)`.

#### Returns

    - `torch.Tensor` - tensor of N-dimensional points.
Shape:
    - `-` *Output* - :math:`(B, N, D)`

#### Examples

```python
>>> points_1 = torch.rand(2, 4, 3)  # BxNx3
>>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
>>> points_0 = kornia.transform_points(trans_01, points_1)  # BxNx3
```

#### Signature

```python
def transform_points(trans_01: torch.Tensor, points_1: torch.Tensor) -> torch.Tensor:
    ...
```



## transform_warp_impl

[Show source in homography_warper.py:320](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L320)

Compute the transform in normalized cooridnates and perform the warping.

#### Signature

```python
def transform_warp_impl(
    src: torch.Tensor,
    centers: torch.Tensor,
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int],
    dsize_dst: Tuple[int, int],
    grid_mode: str,
    padding_mode: str,
) -> torch.Tensor:
    ...
```



## warp_affine_crop

[Show source in homography_warper.py:417](https://github.com/enricobu96/myMID/blob/main/dataset/homography_warper.py#L417)

Applies an affine transformation to a tensor.

The function warp_affine transforms the source tensor using
the specified matrix:

.. math

```python
\text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
M_{21} x + M_{22} y + M_{23} \right )
```

#### Arguments

- `src` *torch.Tensor* - input tensor of shape :math:`(B, C, H, W)`.
- `M` *torch.Tensor* - affine transformation of shape :math:`(B, 2, 3)`.
dsize (Tuple[int, int]): size of the output image (height, width).
- `mode` *str* - interpolation mode to calculate output values
  'bilinear' | 'nearest'. Default: 'bilinear'.
- `padding_mode` *str* - padding mode for outside grid values
  'zeros' | 'border' | 'reflection'. Default: 'zeros'.

#### Returns

- `torch.Tensor` - the warped tensor.

Shape:
    - `-` *Output* - :math:`(B, C, H, W)`

.. note

```python
See a working example `here <https://kornia.readthedocs.io/en/latest/
tutorials/warp_affine.html>`__.
```

#### Signature

```python
def warp_affine_crop(
    src: torch.Tensor,
    centers: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    flags: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    ...
```


