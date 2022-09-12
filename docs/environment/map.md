# Map

[Mymid Index](../README.md#mymid-index) /
[Environment](./index.md#environment) /
Map

> Auto-generated documentation for [environment.map](https://github.com/enricobu96/myMID/blob/main/environment/map.py) module.

- [Map](#map)
  - [GeometricMap](#geometricmap)
    - [GeometricMap().as_image](#geometricmap()as_image)
    - [GeometricMap.batch_rotate](#geometricmapbatch_rotate)
    - [GeometricMap().get_cropped_maps](#geometricmap()get_cropped_maps)
    - [GeometricMap.get_cropped_maps_from_scene_map_batch](#geometricmapget_cropped_maps_from_scene_map_batch)
    - [GeometricMap().get_padded_map](#geometricmap()get_padded_map)
    - [GeometricMap().to_map_points](#geometricmap()to_map_points)
    - [GeometricMap().torch_map](#geometricmap()torch_map)
  - [ImageMap](#imagemap)
  - [Map](#map-1)
    - [Map().as_image](#map()as_image)
    - [Map().get_cropped_maps](#map()get_cropped_maps)
    - [Map().to_map_points](#map()to_map_points)

## GeometricMap

[Show source in map.py:22](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L22)

A Geometric Map is a int tensor of shape [layers, x, y]. The homography must transform a point in scene
coordinates to the respective point in map coordinates.

#### Arguments

- `data` - Numpy array of shape [layers, x, y]
- `homography` - Numpy array of shape [3, 3]

#### Signature

```python
class GeometricMap(Map):
    def __init__(self, data, homography, description=None):
        ...
```

#### See also

- [Map](#map)

### GeometricMap().as_image

[Show source in map.py:44](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L44)

#### Signature

```python
def as_image(self):
    ...
```

### GeometricMap.batch_rotate

[Show source in map.py:61](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L61)

As the input is a map and the warp_affine works on an image coordinate system we would have to
flip the y axis updown, negate the angles, and flip it back after transformation.
This, however, is the same as not flipping at and not negating the radian.

#### Arguments

- `map_batched`
- `centers`
- `angles`
- `out_height`
- `out_width`

#### Signature

```python
@staticmethod
def batch_rotate(map_batched, centers, angles, out_height, out_width):
    ...
```

### GeometricMap().get_cropped_maps

[Show source in map.py:146](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L146)

Returns rotated patches of the map around the transformed scene points.
___________________
|       |          |
|       |ps[3]     |
|       |          |
|       |          |
|      o|__________|
|       |    ps[2] |
|       |          |
|_______|__________|
ps = patch_size

#### Arguments

- `scene_pts` - Scene points: [bs, 2]
- `patch_size` - Extracted Patch size after rotation: [-lat, -long, +lat, +long]
- `rotation` - Rotations in degrees: [bs]
- `device` - Device on which the rotated tensors should be returned.

#### Returns

Rotated and cropped tensor patches.

#### Signature

```python
def get_cropped_maps(self, scene_pts, patch_size, rotation=None, device="cpu"):
    ...
```

### GeometricMap.get_cropped_maps_from_scene_map_batch

[Show source in map.py:81](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L81)

Returns rotated patches of each map around the transformed scene points.
___________________
|       |          |
|       |ps[3]     |
|       |          |
|       |          |
|      o|__________|
|       |    ps[2] |
|       |          |
|_______|__________|
ps = patch_size

#### Arguments

- `maps` - List of GeometricMap objects [bs]
- `scene_pts` - Scene points: [bs, 2]
- `patch_size` - Extracted Patch size after rotation: [-x, -y, +x, +y]
- `rotation` - Rotations in degrees: [bs]
- `device` - Device on which the rotated tensors should be returned.

#### Returns

Rotated and cropped tensor patches.

#### Signature

```python
@classmethod
def get_cropped_maps_from_scene_map_batch(
    cls, maps, scene_pts, patch_size, rotation=None, device="cpu"
):
    ...
```

### GeometricMap().get_padded_map

[Show source in map.py:49](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L49)

#### Signature

```python
def get_padded_map(self, padding_x, padding_y, device):
    ...
```

### GeometricMap().to_map_points

[Show source in map.py:169](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L169)

#### Signature

```python
def to_map_points(self, scene_pts):
    ...
```

### GeometricMap().torch_map

[Show source in map.py:38](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L38)

#### Signature

```python
def torch_map(self, device):
    ...
```



## ImageMap

[Show source in map.py:183](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L183)

#### Signature

```python
class ImageMap(Map):
    def __init__(self):
        ...
```

#### See also

- [Map](#map)



## Map

[Show source in map.py:6](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L6)

#### Signature

```python
class Map(object):
    def __init__(self, data, homography, description=None):
        ...
```

### Map().as_image

[Show source in map.py:12](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L12)

#### Signature

```python
def as_image(self):
    ...
```

### Map().get_cropped_maps

[Show source in map.py:15](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L15)

#### Signature

```python
def get_cropped_maps(self, world_pts, patch_size, rotation=None, device="cpu"):
    ...
```

### Map().to_map_points

[Show source in map.py:18](https://github.com/enricobu96/myMID/blob/main/environment/map.py#L18)

#### Signature

```python
def to_map_points(self, scene_pts):
    ...
```


