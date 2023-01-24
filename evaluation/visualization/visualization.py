from evaluation.trajectory_utils import prediction_output_to_trajectories
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns

"""
File containing visualization utilities. This file also comes with Trajectron++ code (https://github.com/StanfordASL/Trajectron-plus-plus/),
but has been modified for MID by @enricobu96.
"""

"""
STANDARD VISUALIZATION
Visualizes history, future gt and predictions without the map and without wandb.
"""
def plot_trajectories(i,j,fig, ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    ax.axis('equal')
    fig.savefig('plots/scene_' + str(i) + '_' + str(j) + '.png')

def visualize_prediction(i, j, fig, ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    # assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='upper', alpha=0.5)

    plot_trajectories(i, j, fig, ax, prediction_dict, histories_dict, futures_dict, *kwargs)

"""
WANDB VISUALIZATION
Visualizes history, future gt and predictions with map (if present in the scene) and using wandb.
"""

def plot_wandb(fig, ax, prediction_output_dict, dt, max_hl, ph, map=None, batch_num=0, mean_x=None, mean_y=None):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(
        prediction_output_dict,
        dt,
        max_hl,
        ph,
        map=map
        )
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]
    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    # If the map is present plot it
    if map is not None:
        ax.imshow(map.as_image(), alpha=0.7, origin='upper')

    fig, ax = plot_trajectories_wandb(
        fig,
        ax,
        prediction_dict,
        histories_dict,
        futures_dict,
        mean_x=mean_x,
        mean_y=mean_y,
        map=map
        )

    return fig, ax

def plot_trajectories_wandb(fig, ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=2,
                      node_circle_size=10,
                      batch_num=0,
                      kde=False,
                      mean_x=None,
                      mean_y=None,
                      map=None):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        """
        Trajectories retrieving and normalization (ugly but easy, and works):
        1. Get history, future and predictions for the current node
        2. Normalize the trajectories (undo the normalization done during the data preprocessing)
        3. Translate trajectories using homography
        """
        history = histories_dict[node]
        future = futures_dict[node]
        if map.scene == 'sdd':
            history[: , 0] = (history[: , 0]+mean_x)*50
            history[: , 1] = (history[: , 1]+mean_y)*50
            future[: , 0] = (future[: , 0]+mean_x)*50
            future[: , 1] = (future[: , 1]+mean_y)*50
        elif map.scene == 'eth':
            history[: ,0] = (history[: ,0]+mean_x)/0.6
            history[: ,1] = (history[: ,1]+mean_y)/0.6
            future[: ,0] = (future[: ,0]+mean_x)/0.6
            future[: ,1] = (future[: ,1]+mean_y)/0.6
        else:
            history[: ,0] = history[: ,0]+mean_x
            history[: ,1] = history[: ,1]+mean_y
            future[: ,0] = future[: ,0]+mean_x
            future[: ,1] = future[: ,1]+mean_y
        history = histories_dict[node] if not map else map.translate_trajectories(histories_dict[node])
        future = futures_dict[node] if not map else map.translate_trajectories(futures_dict[node])

        predictions = prediction_dict[node]
        for sample_num in range(prediction_dict[node].shape[1]):
            if map.scene == 'sdd':
                predictions[batch_num, sample_num, :, 0] = (predictions[batch_num, sample_num, :, 0]+mean_x)*50
                predictions[batch_num, sample_num, :, 1] = (predictions[batch_num, sample_num, :, 1]+mean_y)*50
            elif map.scene == 'eth':
                predictions[batch_num, sample_num, :, 0] = (predictions[batch_num, sample_num, :, 0]+mean_x)/0.6
                predictions[batch_num, sample_num, :, 1] = (predictions[batch_num, sample_num, :, 1]+mean_y)/0.6
            else:
                predictions[batch_num, sample_num, :, 0] = predictions[batch_num, sample_num, :, 0]+mean_x
                predictions[batch_num, sample_num, :, 1] = predictions[batch_num, sample_num, :, 1]+mean_y
            predictions[batch_num, sample_num, :] = map.translate_trajectories(
                predictions[batch_num, sample_num, :]
                ) if map else predictions[batch_num, sample_num, :]

        if np.isnan(history[-1]).any():
            continue
        
        # Plot history trajectory
        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            # Plot predicted trajectories
            ax.plot(predictions[batch_num, sample_num, :, 0],
                    predictions[batch_num, sample_num, :, 1],
                    '-o', zorder=1)

            # Plot ground truth future trajectory
            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()],
                    zorder=2)

            # Plot current node position
            circle = plt.Circle((history[-1, 0],
                                history[-1, 1]),
                                0.3,
                                facecolor='g',
                                edgecolor='k',
                                lw=0.5,
                                zorder=3)
            ax.add_artist(circle)

    ax.axis('off')
    return fig, ax