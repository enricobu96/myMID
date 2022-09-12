# Mgcvae

[Mymid Index](../../README.md#mymid-index) /
[Models](../index.md#models) /
[Encoders](./index.md#encoders) /
Mgcvae

> Auto-generated documentation for [models.encoders.mgcvae](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py) module.

- [Mgcvae](#mgcvae)
  - [MultimodalGenerativeCVAE](#multimodalgenerativecvae)
    - [MultimodalGenerativeCVAE().add_submodule](#multimodalgenerativecvae()add_submodule)
    - [MultimodalGenerativeCVAE().clear_submodules](#multimodalgenerativecvae()clear_submodules)
    - [MultimodalGenerativeCVAE().create_edge_models](#multimodalgenerativecvae()create_edge_models)
    - [MultimodalGenerativeCVAE().create_graphical_model](#multimodalgenerativecvae()create_graphical_model)
    - [MultimodalGenerativeCVAE().create_new_scheduler](#multimodalgenerativecvae()create_new_scheduler)
    - [MultimodalGenerativeCVAE().create_node_models](#multimodalgenerativecvae()create_node_models)
    - [MultimodalGenerativeCVAE().decoder](#multimodalgenerativecvae()decoder)
    - [MultimodalGenerativeCVAE().encode_edge](#multimodalgenerativecvae()encode_edge)
    - [MultimodalGenerativeCVAE().encode_node_future](#multimodalgenerativecvae()encode_node_future)
    - [MultimodalGenerativeCVAE().encode_node_history](#multimodalgenerativecvae()encode_node_history)
    - [MultimodalGenerativeCVAE().encode_robot_future](#multimodalgenerativecvae()encode_robot_future)
    - [MultimodalGenerativeCVAE().encode_total_edge_influence](#multimodalgenerativecvae()encode_total_edge_influence)
    - [MultimodalGenerativeCVAE().encoder](#multimodalgenerativecvae()encoder)
    - [MultimodalGenerativeCVAE().eval_loss](#multimodalgenerativecvae()eval_loss)
    - [MultimodalGenerativeCVAE().get_latent](#multimodalgenerativecvae()get_latent)
    - [MultimodalGenerativeCVAE().obtain_encoded_tensors](#multimodalgenerativecvae()obtain_encoded_tensors)
    - [MultimodalGenerativeCVAE().p_y_xz](#multimodalgenerativecvae()p_y_xz)
    - [MultimodalGenerativeCVAE().p_z_x](#multimodalgenerativecvae()p_z_x)
    - [MultimodalGenerativeCVAE().predict](#multimodalgenerativecvae()predict)
    - [MultimodalGenerativeCVAE().project_to_GMM_params](#multimodalgenerativecvae()project_to_gmm_params)
    - [MultimodalGenerativeCVAE().q_z_xy](#multimodalgenerativecvae()q_z_xy)
    - [MultimodalGenerativeCVAE().set_annealing_params](#multimodalgenerativecvae()set_annealing_params)
    - [MultimodalGenerativeCVAE().set_curr_iter](#multimodalgenerativecvae()set_curr_iter)
    - [MultimodalGenerativeCVAE().step_annealers](#multimodalgenerativecvae()step_annealers)
    - [MultimodalGenerativeCVAE().summarize_annealers](#multimodalgenerativecvae()summarize_annealers)
    - [MultimodalGenerativeCVAE().train_loss](#multimodalgenerativecvae()train_loss)

## MultimodalGenerativeCVAE

[Show source in mgcvae.py:12](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L12)

#### Signature

```python
class MultimodalGenerativeCVAE(object):
    def __init__(self, env, node_type, model_registrar, hyperparams, device, edge_types):
        ...
```

### MultimodalGenerativeCVAE().add_submodule

[Show source in mgcvae.py:58](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L58)

#### Signature

```python
def add_submodule(self, name, model_if_absent):
    ...
```

### MultimodalGenerativeCVAE().clear_submodules

[Show source in mgcvae.py:61](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L61)

#### Signature

```python
def clear_submodules(self):
    ...
```

### MultimodalGenerativeCVAE().create_edge_models

[Show source in mgcvae.py:235](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L235)

#### Signature

```python
def create_edge_models(self, edge_types):
    ...
```

### MultimodalGenerativeCVAE().create_graphical_model

[Show source in mgcvae.py:264](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L264)

Creates or queries all trainable components.

#### Arguments

- `edge_types` - List containing strings for all possible edge types for the node type.

#### Returns

None

#### Signature

```python
def create_graphical_model(self, edge_types):
    ...
```

### MultimodalGenerativeCVAE().create_new_scheduler

[Show source in mgcvae.py:287](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L287)

#### Signature

```python
def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
    ...
```

### MultimodalGenerativeCVAE().create_node_models

[Show source in mgcvae.py:64](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L64)

#### Signature

```python
def create_node_models(self):
    ...
```

### MultimodalGenerativeCVAE().decoder

[Show source in mgcvae.py:931](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L931)

Decoder of the CVAE.

#### Arguments

- `mode` - Mode in which the model is operated. E.g. Train, Eval, Predict.
- `x` - Input / Condition tensor.
- `x` - Input / Condition tensor.
- `x_nr_t` - Joint state of node and robot (if robot is in scene).
- `y` - Future tensor.
- `y_r` - Encoded future tensor.
- `n_s_t0` - Standardized current state of the node.
- `z` - Stacked latent state.
- `prediction_horizon` - Number of prediction timesteps.
- `num_samples` - Number of samples from the latent space.

#### Returns

Log probability of y over p.

#### Signature

```python
def decoder(
    self, mode, x, x_nr_t, y, y_r, n_s_t0, z, labels, prediction_horizon, num_samples
):
    ...
```

### MultimodalGenerativeCVAE().encode_edge

[Show source in mgcvae.py:540](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L540)

#### Signature

```python
def encode_edge(
    self,
    mode,
    node_history,
    node_history_st,
    edge_type,
    neighbors,
    neighbors_edge_value,
    first_history_indices,
):
    ...
```

### MultimodalGenerativeCVAE().encode_node_future

[Show source in mgcvae.py:665](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L665)

Encodes the node future (during training) using a bi-directional LSTM

#### Arguments

- `mode` - Mode in which the model is operated. E.g. Train, Eval, Predict.
- `node_present` - Current state of the node. [bs, state]
- `node_future` - Future states of the node. [bs, ph, state]

#### Returns

Encoded future.

#### Signature

```python
def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
    ...
```

### MultimodalGenerativeCVAE().encode_node_history

[Show source in mgcvae.py:519](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L519)

Encodes the nodes history.

#### Arguments

- `mode` - Mode in which the model is operated. E.g. Train, Eval, Predict.
- `node_hist` - Historic and current state of the node. [bs, mhl, state]
- `first_history_indices` - First timestep (index) in scene for which data is available for a node [bs]

#### Returns

Encoded node history tensor. [bs, enc_rnn_dim]

#### Signature

```python
def encode_node_history(self, mode, node_hist, first_history_indices):
    ...
```

### MultimodalGenerativeCVAE().encode_robot_future

[Show source in mgcvae.py:695](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L695)

Encodes the robot future (during training) using a bi-directional LSTM

#### Arguments

- `mode` - Mode in which the model is operated. E.g. Train, Eval, Predict.
- `robot_present` - Current state of the robot. [bs, state]
- `robot_future` - Future states of the robot. [bs, ph, state]

#### Returns

Encoded future.

#### Signature

```python
def encode_robot_future(self, mode, robot_present, robot_future) -> torch.Tensor:
    ...
```

### MultimodalGenerativeCVAE().encode_total_edge_influence

[Show source in mgcvae.py:622](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L622)

#### Signature

```python
def encode_total_edge_influence(
    self, mode, encoded_edges, node_history_encoder, batch_size
):
    ...
```

### MultimodalGenerativeCVAE().encoder

[Show source in mgcvae.py:895](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L895)

Encoder of the CVAE.

#### Arguments

- `mode` - Mode in which the model is operated. E.g. Train, Eval, Predict.
- `x` - Input / Condition tensor.
- `y_e` - Encoded future tensor.
- `num_samples` - Number of samples from the latent space during Prediction.

#### Returns

tuple(z, kl_obj)
    WHERE
    - z: Samples from the latent space.
    - kl_obj: KL Divergenze between q and p

#### Signature

```python
def encoder(self, mode, x, y_e, num_samples=None):
    ...
```

### MultimodalGenerativeCVAE().eval_loss

[Show source in mgcvae.py:1102](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L1102)

Calculates the evaluation loss for a batch.

#### Arguments

- `inputs` - Input tensor including the state for each agent over time [bs, t, state].
- `inputs_st` - Standardized input tensor.
- `first_history_indices` - First timestep (index) in scene for which data is available for a node [bs]
- `labels` - Label tensor including the label output for each agent over time [bs, t, pred_state].
- `labels_st` - Standardized label tensor.
- `neighbors` - Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                    [[bs, t, neighbor state]]
- `neighbors_edge_value` - Preprocessed edge values for all neighbor nodes [[N]]
- `robot` - Standardized robot state over time. [bs, t, robot_state]
- `map` - Tensor of Map information. [bs, channels, x, y]
- `prediction_horizon` - Number of prediction timesteps.

#### Returns

tuple(nll_q_is, nll_p, nll_exact, nll_sampled)

#### Signature

```python
def eval_loss(
    self,
    inputs,
    inputs_st,
    first_history_indices,
    labels,
    labels_st,
    neighbors,
    neighbors_edge_value,
    robot,
    map,
    prediction_horizon,
) -> torch.Tensor:
    ...
```

### MultimodalGenerativeCVAE().get_latent

[Show source in mgcvae.py:959](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L959)

Calculates the training loss for a batch.

#### Arguments

- `inputs` - Input tensor including the state for each agent over time [bs, t, state].
- `inputs_st` - Standardized input tensor.
- `first_history_indices` - First timestep (index) in scene for which data is available for a node [bs]
- `labels` - Label tensor including the label output for each agent over time [bs, t, pred_state].
- `labels_st` - Standardized label tensor.
- `neighbors` - Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                    [[bs, t, neighbor state]]
- `neighbors_edge_value` - Preprocessed edge values for all neighbor nodes [[N]]
- `robot` - Standardized robot state over time. [bs, t, robot_state]
- `map` - Tensor of Map information. [bs, channels, x, y]
- `prediction_horizon` - Number of prediction timesteps.

#### Returns

Scalar tensor -> nll loss

#### Signature

```python
def get_latent(
    self,
    inputs,
    inputs_st,
    first_history_indices,
    labels,
    labels_st,
    neighbors,
    neighbors_edge_value,
    robot,
    map,
    prediction_horizon,
) -> torch.Tensor:
    ...
```

### MultimodalGenerativeCVAE().obtain_encoded_tensors

[Show source in mgcvae.py:363](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L363)

Encodes input and output tensors for node and robot.

#### Arguments

- `mode` - Mode in which the model is operated. E.g. Train, Eval, Predict.
- `inputs` - Input tensor including the state for each agent over time [bs, t, state].
- `inputs_st` - Standardized input tensor.
- `labels` - Label tensor including the label output for each agent over time [bs, t, pred_state].
- `labels_st` - Standardized label tensor.
- `first_history_indices` - First timestep (index) in scene for which data is available for a node [bs]
- `neighbors` - Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                    [[bs, t, neighbor state]]
- `neighbors_edge_value` - Preprocessed edge values for all neighbor nodes [[N]]
- `robot` - Standardized robot state over time. [bs, t, robot_state]
- `map` - Tensor of Map information. [bs, channels, x, y]

#### Returns

tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
    WHERE
    - x: Encoded input / condition tensor to the CVAE x_e.
    - x_r_t: Robot state (if robot is in scene).
    - y_e: Encoded label / future of the node.
    - y_r: Encoded future of the robot.
    - y: Label / future of the node.
    - n_s_t0: Standardized current state of the node.

#### Signature

```python
def obtain_encoded_tensors(
    self,
    mode,
    inputs,
    inputs_st,
    labels,
    labels_st,
    first_history_indices,
    neighbors,
    neighbors_edge_value,
    robot,
    map,
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    ...
```

### MultimodalGenerativeCVAE().p_y_xz

[Show source in mgcvae.py:786](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L786)

```p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)
:param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
:param x: Input / Condition tensor.
:param x_nr_t: Joint state of node and robot (if robot is in scene).
:param y: Future tensor.
:param y_r: Encoded future tensor.
:param n_s_t0: Standardized current state of the node.
:param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
:param prediction_horizon: Number of prediction timesteps.
:param num_samples: Number of samples from the latent space.
:param num_components: Number of GMM components.
:param gmm_mode: If True: The mode of the GMM is sampled.
:return: GMM2D. If mode is Predict, also samples from the GMM.
```

#### Signature

```python
def p_y_xz(
    self,
    mode,
    x,
    x_nr_t,
    y_r,
    n_s_t0,
    z_stacked,
    prediction_horizon,
    num_samples,
    num_components=1,
    gmm_mode=False,
):
    ...
```

### MultimodalGenerativeCVAE().p_z_x

[Show source in mgcvae.py:748](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L748)

```p_\theta(z \mid \mathbf{x}_i)
:param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
:param x: Input / Condition tensor.
:return: Latent distribution of the CVAE.
```

#### Signature

```python
def p_z_x(self, mode, x):
    ...
```

### MultimodalGenerativeCVAE().predict

[Show source in mgcvae.py:1158](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L1158)

Predicts the future of a batch of nodes.

#### Arguments

- `inputs` - Input tensor including the state for each agent over time [bs, t, state].
- `inputs_st` - Standardized input tensor.
- `first_history_indices` - First timestep (index) in scene for which data is available for a node [bs]
- `neighbors` - Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                    [[bs, t, neighbor state]]
- `neighbors_edge_value` - Preprocessed edge values for all neighbor nodes [[N]]
- `robot` - Standardized robot state over time. [bs, t, robot_state]
- `map` - Tensor of Map information. [bs, channels, x, y]
- `prediction_horizon` - Number of prediction timesteps.
- `num_samples` - Number of samples from the latent space.
- `z_mode` - If True: Select the most likely latent state.
- `gmm_mode` - If True: The mode of the GMM is sampled.
- `all_z_sep` - Samples each latent mode individually without merging them into a GMM.
- `full_dist` - Samples all latent states and merges them into a GMM as output.
- `pcmd` - If True: Sort the outputs for pcmd.

#### Signature

```python
def predict(
    self,
    inputs,
    inputs_st,
    first_history_indices,
    neighbors,
    neighbors_edge_value,
    robot,
    map,
    prediction_horizon,
    num_samples,
    z_mode=False,
    gmm_mode=False,
    full_dist=True,
    all_z_sep=False,
    pcmd=False,
):
    ...
```

### MultimodalGenerativeCVAE().project_to_GMM_params

[Show source in mgcvae.py:768](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L768)

Projects tensor to parameters of a GMM with N components and D dimensions.

#### Arguments

- `tensor` - Input tensor.

#### Returns

tuple(log_pis, mus, log_sigmas, corrs)
    WHERE
    - log_pis: Weight (logarithm) of each GMM component. [N]
    - mus: Mean of each GMM component. [N, D]
    - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
    - corrs: Correlation between the GMM components. [N]

#### Signature

```python
def project_to_GMM_params(
    self, tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    ...
```

### MultimodalGenerativeCVAE().q_z_xy

[Show source in mgcvae.py:725](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L725)

```q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)
:param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
:param x: Input / Condition tensor.
:param y_e: Encoded future tensor.
:return: Latent distribution of the CVAE.
```

#### Signature

```python
def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
    ...
```

### MultimodalGenerativeCVAE().set_annealing_params

[Show source in mgcvae.py:308](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L308)

#### Signature

```python
def set_annealing_params(self):
    ...
```

### MultimodalGenerativeCVAE().set_curr_iter

[Show source in mgcvae.py:55](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L55)

#### Signature

```python
def set_curr_iter(self, curr_iter):
    ...
```

### MultimodalGenerativeCVAE().step_annealers

[Show source in mgcvae.py:341](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L341)

#### Signature

```python
def step_annealers(self):
    ...
```

### MultimodalGenerativeCVAE().summarize_annealers

[Show source in mgcvae.py:356](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L356)

#### Signature

```python
def summarize_annealers(self):
    ...
```

### MultimodalGenerativeCVAE().train_loss

[Show source in mgcvae.py:1017](https://github.com/enricobu96/myMID/blob/main/models/encoders/mgcvae.py#L1017)

Calculates the training loss for a batch.

#### Arguments

- `inputs` - Input tensor including the state for each agent over time [bs, t, state].
- `inputs_st` - Standardized input tensor.
- `first_history_indices` - First timestep (index) in scene for which data is available for a node [bs]
- `labels` - Label tensor including the label output for each agent over time [bs, t, pred_state].
- `labels_st` - Standardized label tensor.
- `neighbors` - Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                    [[bs, t, neighbor state]]
- `neighbors_edge_value` - Preprocessed edge values for all neighbor nodes [[N]]
- `robot` - Standardized robot state over time. [bs, t, robot_state]
- `map` - Tensor of Map information. [bs, channels, x, y]
- `prediction_horizon` - Number of prediction timesteps.

#### Returns

Scalar tensor -> nll loss

#### Signature

```python
def train_loss(
    self,
    inputs,
    inputs_st,
    first_history_indices,
    labels,
    labels_st,
    neighbors,
    neighbors_edge_value,
    robot,
    map,
    prediction_horizon,
) -> torch.Tensor:
    ...
```


