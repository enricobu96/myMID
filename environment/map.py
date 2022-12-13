import torch
import numpy as np
from dataset.homography_warper import get_rotation_matrix2d, warp_affine_crop
from PIL import Image
import cv2
from collections import OrderedDict
import torchvision.transforms as TT
import math

class Map(object):
    def __init__(self, data, homography=None, description=None, scene=None):
        self.data = np.asarray(Image.open(data))
        self.homography = np.loadtxt(homography)
        self.homography_inv = np.linalg.inv(self.homography)
        self.description = description
        self.scene = scene

    def as_image(self):
        return self.data
    
    def translate_trajectories(self, trajectories):
        trajs = trajectories.copy()
        if self.scene == 'sdd':
            return trajs
        elif self.scene in ['eth', 'hotel']:
            trajs = self._world_to_pixel(trajs, self.homography_inv)
            trajs = np.flip(trajs, axis=1)
            return trajs
        elif self.scene in ['univ', 'zara1', 'zara2']:
            return self._world_to_pixel(trajs, self.homography_inv)
        else:
            print('Unknown scene')
            return None

    def _world_to_pixel(self, world_pts, homography):
        world_pts = np.concatenate(
            (world_pts, np.ones((len(world_pts), 1))), axis=1)
        pixel_coord = np.dot(homography, world_pts.T).T
        pixel_coord = pixel_coord[:, 0:2] / pixel_coord[:, 2][:, np.newaxis]
        pixel_coord = pixel_coord
        return pixel_coord

    def get_cropped_maps(self, world_pts, patch_size, rotation=None, device='cpu'):
        raise NotImplementedError

    def to_map_points(self, scene_pts):
        raise NotImplementedError

class SemanticMap(object):
    def __init__(self, data, homography=None, description=None, scene=None):

        self.semantic_classes = OrderedDict([
            ('unlabeled', 'gray'),
            ('pavement', 'blue'),
            ('road', 'red'),
            ('structure', 'orange'),
            ('terrain', 'cyan'),
            ('tree', 'green'),
        ])
        sem_map = cv2.imread(data, flags=0)
        num_classes = len(self.semantic_classes)
        sem_map = [(sem_map == v) for v in range(num_classes)]
        sem_map = np.stack(sem_map, axis=-1).astype(int)
        self.data = sem_map
        self.tensor_image = self._create_tensor_image(down_factor=8)
        self.description = description
        self.scene = scene
    
    def as_image(self):
        return self.data

    def get_tensor_image(self):
        return self.tensor_image


    def _create_tensor_image(self, down_factor=1):
        img = TT.functional.to_tensor(self.data)
        C, H, W = img.shape
        new_heigth = int(H / down_factor)
        new_width = int(W / down_factor)
        tensor_image = TT.functional.resize(img, (new_heigth, new_width),
                                            interpolation=TT.InterpolationMode.NEAREST)
        return tensor_image


class GeometricMap(Map):
    """
    A Geometric Map is a int tensor of shape [layers, x, y]. The homography must transform a point in scene
    coordinates to the respective point in map coordinates.

    :param data: Numpy array of shape [layers, x, y]
    :param homography: Numpy array of shape [3, 3]
    """
    def __init__(self, data, homography, description=None):
        #assert isinstance(data.dtype, np.floating), "Geometric Maps must be float values."
        super(GeometricMap, self).__init__(data, homography, description=description)

        self._last_padding = None
        self._last_padded_map = None
        self._torch_map = None

    def torch_map(self, device):
        if self._torch_map is not None:
            return self._torch_map
        self._torch_map = torch.tensor(self.data, dtype=torch.uint8, device=device)
        return self._torch_map

    def as_image(self):
        # We have to transpose x and y to rows and columns. Assumes origin is lower left for image
        # Also we move the channels to the last dimension
        return (np.transpose(self.data, (2, 1, 0))).astype(np.uint)

    def get_padded_map(self, padding_x, padding_y, device):
        if self._last_padding == (padding_x, padding_y):
            return self._last_padded_map
        else:
            self._last_padding = (padding_x, padding_y)
            self._last_padded_map = torch.full((self.data.shape[0],
                                                self.data.shape[1] + 2 * padding_x,
                                                self.data.shape[2] + 2 * padding_y),
                                               False, dtype=torch.uint8)
            self._last_padded_map[..., padding_x:-padding_x, padding_y:-padding_y] = self.torch_map(device)
            return self._last_padded_map

    @staticmethod
    def batch_rotate(map_batched, centers, angles, out_height, out_width):
        """
        As the input is a map and the warp_affine works on an image coordinate system we would have to
        flip the y axis updown, negate the angles, and flip it back after transformation.
        This, however, is the same as not flipping at and not negating the radian.

        :param map_batched:
        :param centers:
        :param angles:
        :param out_height:
        :param out_width:
        :return:
        """
        M = get_rotation_matrix2d(centers, angles, torch.ones_like(angles))
        rotated_map_batched = warp_affine_crop(map_batched, centers, M,
                                               dsize=(out_height, out_width), padding_mode='zeros')

        return rotated_map_batched

    @classmethod
    def get_cropped_maps_from_scene_map_batch(cls, maps, scene_pts, patch_size, rotation=None, device='cpu'):
        """
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

        :param maps: List of GeometricMap objects [bs]
        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-x, -y, +x, +y]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        batch_size = scene_pts.shape[0]
        lat_size = 2 * np.max((patch_size[0], patch_size[2]))
        long_size = 2 * np.max((patch_size[1], patch_size[3]))
        assert lat_size % 2 == 0, "Patch width must be divisible by 2"
        assert long_size % 2 == 0, "Patch length must be divisible by 2"
        lat_size_half = lat_size // 2
        long_size_half = long_size // 2

        context_padding_x = int(np.ceil(np.sqrt(2) * lat_size))
        context_padding_y = int(np.ceil(np.sqrt(2) * long_size))

        centers = torch.tensor([s_map.to_map_points(scene_pts[np.newaxis, i]) for i, s_map in enumerate(maps)],
                               dtype=torch.long, device=device).squeeze(dim=1) \
                  + torch.tensor([context_padding_x, context_padding_y], device=device, dtype=torch.long)

        padded_map = [s_map.get_padded_map(context_padding_x, context_padding_y, device=device) for s_map in maps]

        padded_map_batched = torch.stack([padded_map[i][...,
                                          centers[i, 0] - context_padding_x: centers[i, 0] + context_padding_x,
                                          centers[i, 1] - context_padding_y: centers[i, 1] + context_padding_y]
                                          for i in range(centers.shape[0])], dim=0)

        center_patches = torch.tensor([[context_padding_y, context_padding_x]],
                                      dtype=torch.int,
                                      device=device).repeat(batch_size, 1)

        if rotation is not None:
            angles = torch.Tensor(rotation)
        else:
            angles = torch.zeros(batch_size)

        rotated_map_batched = cls.batch_rotate(padded_map_batched/255.,
                                                center_patches.float(),
                                                angles,
                                                long_size,
                                                lat_size)

        del padded_map_batched

        return rotated_map_batched[...,
               long_size_half - patch_size[1]:(long_size_half + patch_size[3]),
               lat_size_half - patch_size[0]:(lat_size_half + patch_size[2])]

    def get_cropped_maps(self, scene_pts, patch_size, rotation=None, device='cpu'):
        """
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

        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-lat, -long, +lat, +long]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        return self.get_cropped_maps_from_scene_map_batch([self]*scene_pts.shape[0], scene_pts,
                                                          patch_size, rotation=rotation, device=device)

    def to_map_points(self, scene_pts):
        org_shape = None
        if len(scene_pts.shape) > 2:
            org_shape = scene_pts.shape
            scene_pts = scene_pts.reshape((-1, 2))
        N, dims = scene_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = scene_pts.T
        map_points = (self.homography @ points_with_one).T[..., :dims]
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points

class ImageMap(Map):
    def __init__(self):
        raise NotImplementedError