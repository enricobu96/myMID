"""
process_data.py
This part of the program basically just processes data putting it into a convenient shape for further operations.
No real logic is applied here, since also the noising part of the diffusion model (the forward process) should be
performed in the second part of the program.
"""

import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle

from environment import Environment, Scene, Node, derivative_of
from environment.map import Map, SemanticMap

# For debug reasons
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

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


nl = 0
l = 0

data_folder_name = 'processed_data_noise'

maybe_makedirs(data_folder_name)

data_file_name_to_scene = {
            "biwi_eth": 'eth',
            "biwi_hotel": 'hotel',
            "students001": 'univ',
            "students003": 'univ',
            "uni_examples": 'univ',
            "crowds_zara01": 'zara1',
            "crowds_zara02": 'zara2',
            "crowds_zara03": 'zara2',
            }

"""
Process data for ETH-UCY dataset.
"""
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:
        # Creates Environment object. See documentation for Environment class.
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join(data_folder_name, '_'.join([desired_source, data_class]) + '.pkl')

        for subdir, dirs, files in os.walk(os.path.join('raw_data', desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):                    
                    # Reads input txt file and loads it into dataframe
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                    data['frame_id'] = data['frame_id'] // 10

                    data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)

                    data.sort_values('frame_id', inplace=True)

                    # Standardizes positions for ADE and FDE evaluation, which are then multiplied by 0.6
                    if desired_source == "eth" and data_class == "test":
                        data['pos_x'] = data['pos_x'] * 0.6
                        data['pos_y'] = data['pos_y'] * 0.6

                    mean_x = data['pos_x'].mean()
                    mean_y = data['pos_y'].mean()
                    # if data_class == "train":
                    #     #data_gauss = data.copy(deep=True)
                    #     data['pos_x'] = data['pos_x'] + 2 * np.random.normal(0,1)
                    #     data['pos_y'] = data['pos_y'] + 2 * np.random.normal(0,1)

                        #data = pd.concat([data, data_gauss])

                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    file_name = file.replace("_train", "").replace("_val", "").replace(".txt", "")
                    map_path = 'raw_data/eth_scenes/' + data_file_name_to_scene[file_name] + '_reference.jpg'
                    homography_path = 'raw_data/eth_scenes/' + data_file_name_to_scene[file_name] + '_H.txt'
                    semantic_map_gt_path = 'raw_data/eth_scenes/' + data_file_name_to_scene[file_name] + '_mask.png'
                    semantic_map_pred_path = 'raw_data/eth_scenes/' + data_file_name_to_scene[file_name] + '_pred_mask.png'

                    scene = Scene(
                        timesteps=max_timesteps+1,
                        map=Map(
                            data=map_path,
                            homography=homography_path,
                            scene=desired_source
                        ),
                        semantic_map_gt=SemanticMap(
                            data=semantic_map_gt_path,
                            homography=homography_path,
                            scene=desired_source
                        ),
                        semantic_map_pred=SemanticMap(
                            data=semantic_map_pred_path,
                            homography=homography_path,
                            scene=desired_source
                        ),
                        dt=dt,
                        name=desired_source + "_" + data_class,
                        aug_func=augment if data_class == 'train' else None,
                        mean_x=mean_x,
                        mean_y=mean_y
                    )

                    # For each node
                    for node_id in pd.unique(data['node_id']):

                        node_df = data[data['node_id'] == node_id]

                        node_values = node_df[['pos_x', 'pos_y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0]

                        # Get x,y positions and compute velocity and acceleration derivating them from space
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
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
exit()
# Process Stanford Drone. Data obtained from Y-Net github repo
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


for data_class in ["train", "test"]:
    raw_path = "raw_data/stanford"
    out_path = "processed_data_noise"
    data_path = os.path.join(raw_path, f"{data_class}_trajnet.pkl")
    print(f"Processing SDD {data_class}")
    data_out_path = os.path.join(out_path, f"sdd_{data_class}.pkl")
    df = pickle.load(open(data_path, "rb"))
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")

    for scene, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        data['frame'] = data['frame'] // 12

        data['frame'] -= data['frame'].min()

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # apply data scale as same as PECnet
        data['x'] = data['x']/50
        data['y'] = data['y']/50

        # Computing mean for subsequent visualization
        mean_x = data['x'].mean()
        mean_y = data['y'].mean()

        # Mean Position
        data['x'] = data['x'] - data['x'].mean()
        data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()

        scene_id = data['sceneId'].iloc[0]

        if len(data) > 0:
            map_path = raw_path + '/maps/' + data_class + '/' + scene_id + '/reference.jpg'
            homography_path = raw_path + '/maps/' + data_class + '/' + scene_id + '/H.txt'
            semantic_map_gt_path = raw_path + '/maps/' + data_class + '/' + scene_id + '/mask.png'
            semantic_map_pred_path = raw_path + '/maps/' + data_class + '/' + scene_id + '/pred_mask.png'
            scene = Scene(
                timesteps=max_timesteps+1,
                map=Map(
                    data = map_path,
                    homography = homography_path,
                    scene = 'sdd'
                ),
                semantic_map_gt=SemanticMap(
                    data = semantic_map_gt_path,
                    homography = homography_path,
                    scene = 'sdd'
                ),
                semantic_map_pred=SemanticMap(
                    data = semantic_map_pred_path,
                    homography = homography_path,
                    scene = 'sdd'
                ),
                dt=dt,
                name="sdd_" + data_class,
                aug_func=augment if data_class == 'train' else None,
                mean_x=mean_x,
                mean_y=mean_y
            )
            n=0
            for node_id in pd.unique(data['node_id']):

                node_df = data[data['node_id'] == node_id]


                if len(node_df) > 1:
                    assert np.all(np.diff(node_df['frame']) == 1)
                    if not np.all(np.diff(node_df['frame']) == 1):
                        pdb.set_trace()

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

    if len(scenes) > 0:
        with open(data_out_path, 'wb') as f:
            #pdb.set_trace()
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)