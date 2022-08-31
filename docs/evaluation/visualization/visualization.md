# Visualization

[Mymid Index](../../README.md#mymid-index) /
[Evaluation](../index.md#evaluation) /
[Visualization](./index.md#visualization) /
Visualization

> Auto-generated documentation for [evaluation.visualization.visualization](https://github.com/enricobu96/myMID/blob/main/evaluation/visualization/visualization.py) module.

- [Visualization](#visualization)
  - [plot_trajectories](#plot_trajectories)
  - [visualize_prediction](#visualize_prediction)

## plot_trajectories

[Show source in visualization.py:8](https://github.com/enricobu96/myMID/blob/main/evaluation/visualization/visualization.py#L8)

#### Signature

```python
def plot_trajectories(
    ax,
    prediction_dict,
    histories_dict,
    futures_dict,
    line_alpha=0.7,
    line_width=0.2,
    edge_width=2,
    circle_edge_width=0.5,
    node_circle_size=0.3,
    batch_num=0,
    kde=False,
):
    ...
```



## visualize_prediction

[Show source in visualization.py:63](https://github.com/enricobu96/myMID/blob/main/evaluation/visualization/visualization.py#L63)

#### Signature

```python
def visualize_prediction(
    ax, prediction_output_dict, dt, max_hl, ph, robot_node=None, map=None, **kwargs
):
    ...
```


