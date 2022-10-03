import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle

from environment import Environment, Scene, Node, derivative_of
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 1

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

def maybe_makedirs(path_to_create):
    """
    This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    """
    Data augmentation function for scene (rotation). Returns Scene object with augmented data.
    """
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        # Create node and add it to the scene
        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    """
    Executes data augmentation.
    """
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

nl = 0
l = 0

data_folder_name = 'processed_data_noise'

maybe_makedirs(data_folder_name)
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


for data_class in ["train", "test"]:
    raw_path = "raw_data/stanford/longterm"
    out_path = "processed_data_noise"
    data_path = os.path.join(raw_path, f"{data_class}_longterm.pkl")
    print(f"Processing SDD {data_class}")
    data_out_path = os.path.join(out_path, f"sdd_{data_class}.pkl")
    df = pickle.load(open(data_path, "rb"))
    df = swap_columns(df, 'trackId', 'frame')
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")

    for scene, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        data['frame'] = data['frame'] // 30
        # data['frame'] -= data['frame'].min()

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # apply data scale as same as PECnet
        data['x'] = data['x']/50
        data['y'] = data['y']/50

        # Mean Position
        # data['x'] = data['x'] - data['x'].mean()
        # data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()

        if len(data) > 0:
            scene = Scene(timesteps=max_timesteps+1, dt=dt, name="sdd_" + data_class, aug_func=augment if data_class == 'train' else None)
            n=0
            for node_id in pd.unique(data['node_id']):
                nodes_df = data[data['node_id'] == node_id]
                for meta_id in pd.unique(nodes_df['metaId']):
                    node_df = nodes_df[nodes_df['metaId'] == meta_id]

                    # Mean position
                    node_df['x'] -= node_df['x'].mean()
                    node_df['y'] -= node_df['y'].mean()

                    # Normalization of frames
                    node_df['frame'] -= node_df['frame'].min()

                    if len(node_df) > 1:
                        assert np.all(np.diff(node_df['frame']) == 1)                         
                        node_values = node_df[['x', 'y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame'].iloc[0]

                        x = node_values[:, 0]
                        y = node_values[:, 1]
                        vx = derivative_of(x, scene.dt)
                        vy = derivative_of(y, scene.dt)
                        ax = derivative_of(vx, scene.dt)
                        ay = derivative_of(vy, scene.dt)

                        data_dict = {('position', 'x'): x,
                                    ('position', 'y'): y,
                                    ('velocity', 'x'): vx,
                                    ('velocity', 'y'): vy,
                                    ('acceleration', 'x'): ax,
                                    ('acceleration', 'y'): ay}

                        node_data = pd.DataFrame(data_dict, columns=data_columns)
                        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                        node.first_timestep = new_first_idx

                        scene.nodes.append(node)
            if data_class == 'train':
                scene.augmented = list()
                angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(scene)
            scenes.append(scene)
    env.scenes = scenes
    print(len(scenes))

    if len(scenes) > 0:
        with open(data_out_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
