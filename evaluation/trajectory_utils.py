import numpy as np
import pdb

"""
File with trajectory-specific utility function. This file also comes with Trajectron++ code (https://github.com/StanfordASL/Trajectron-plus-plus/).

Functions
prediction_output_to_trajectories (prediction_output_dict, dt, max_h, ph, map,  prune_ph_to_future) -> dict(), dict(), dict()
    returns three dictionaries of dictionaries, containing information on output, history and futures. Used only internally
    for evaluation of the model.
"""

def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {'position': ['x', 'y']}

            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]
            future = node.get(np.array([t + 1, t + ph]), position_state)
            future = future[~np.isnan(future.sum(axis=1))]

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :, :future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output

            histories_dict[t][node] = history
            output_dict[t][node] = trajectory
            futures_dict[t][node] = future

    return output_dict, histories_dict, futures_dict
