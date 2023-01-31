import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# custom transformer goal based on SAR
class TransformerGoalSAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 2
        self.embedding_size = 32
        self.nhead = 8
        self.d_hidden = 2048
        self.n_layers_temporal = 1
        self.dropout_prob = 0
        self.output_size = 2
        self.extra_features = 4

        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(
            self.input_size, self.embedding_size)

        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_input_temporal = nn.Dropout(self.dropout_prob)

        # Temporal encoder layer for temporal sequence
        self.temporal_encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_size + self.extra_features * self.nhead,
            nhead=self.nhead,
            dim_feedforward=self.d_hidden)

        # Temporal encoder for temporal sequence
        self.temporal_encoder = TransformerEncoder(
            self.temporal_encoder_layer,
            num_layers=self.n_layers_temporal)

        # Fusion layer
        self.fusion_layer = nn.Linear(
            self.embedding_size + self.nhead * self.extra_features + \
            self.extra_features * 2 - 1, self.embedding_size)

        # Output layer
        self.output_layer = nn.Linear(
            self.embedding_size, self.output_size)

    def forward(self, history, future, goal, num_samples=1, if_test=False):
        x_0_pos = future # (B, 12, 2)
        x_0_pos = torch.cumsum(x_0_pos, dim=0) # integrate to get positions

        history_pos = history[:, :, :2] # (B, 8, 2)

        obs_length = history_pos.shape[1]
        seq_length = future.shape[1]

        # Remove NaNs
        nan_mask = torch.isnan(x_0_pos)
        not_nan_mask = torch.logical_not(nan_mask)
        x_0_pos[nan_mask] = torch.masked_select(x_0_pos, not_nan_mask)[0]
        nan_mask = torch.isnan(history_pos)
        not_nan_mask = torch.logical_not(nan_mask)
        history_pos[nan_mask] = torch.masked_select(history_pos, not_nan_mask)[0]

        batch_coords = torch.cat([history_pos, x_0_pos], dim=1)
        batch_coords = batch_coords.permute(1, 0, 2)
        
        seq_length, num_agents, _ = batch_coords.shape

        # START SAMPLES LOOP
        all_outputs = []
        for sample_idx in range(num_samples):
            # Output tensor of shape (seq_length,N,2)
            outputs = torch.zeros(seq_length, num_agents,
                                  self.output_size).to(history.device)
            # add observation as first output
            outputs[0:obs_length] = batch_coords[0:obs_length]

            goal_point = goal['goal_point'].squeeze(0)

            for frame_idx in range(obs_length, seq_length):
                if if_test and frame_idx >= obs_length:
                    current_agents = torch.cat((
                        batch_coords[:obs_length],
                        outputs[obs_length:frame_idx]))
                else:
                    current_agents = batch_coords[:frame_idx]

                ##################
                # RECURRENT MODULE
                ##################

                # Input Embedding
                temporal_input_embedded = self.dropout_input_temporal(self.relu(
                    self.input_embedding_layer_temporal(current_agents)))

                # compute current positions and current time step
                # and distance to goal
                last_positions = current_agents[-1]
                current_time_step = torch.full(size=(last_positions.shape[0], 1),
                                               fill_value=frame_idx).to(history.device)
                distance_to_goal = goal_point - last_positions
                # prepare everything for concatenation
                # Transformers need everything to be multiple of nhead
                last_positions_to_cat = last_positions.repeat(
                    frame_idx, 1, self.nhead//2)
                current_time_step_to_cat = current_time_step.repeat(
                    frame_idx, 1, self.nhead)
                final_positions_pred_to_cat = goal_point.repeat(
                    frame_idx, 1, self.nhead//2)
                distance_to_goal_to_cat = distance_to_goal.repeat(
                    frame_idx, 1, self.nhead//2)

                # concat additional info BEFORE temporal transformer
                temporal_input_cat = torch.cat(
                    (temporal_input_embedded,
                     final_positions_pred_to_cat,
                     last_positions_to_cat,
                     distance_to_goal_to_cat,
                     current_time_step_to_cat,
                     ), dim=2)

                # temporal transformer encoding
                temporal_output = self.temporal_encoder(
                    temporal_input_cat)
                # Take last temporal encoding
                temporal_output_last = temporal_output[-1]

                # concat additional info AFTER temporal transformer
                fusion_feat = torch.cat((
                    temporal_output_last,
                    last_positions,
                    goal_point,
                    distance_to_goal,
                    current_time_step,
                ), dim=1)

                # Fusion FC layer
                fusion_feat = self.fusion_layer(fusion_feat)

                # Output FC decoder
                outputs_current = self.output_layer(fusion_feat)
                # Append to outputs
                outputs[frame_idx] = outputs_current

            all_outputs.append(outputs)

        # stack predictions
        all_outputs = torch.stack(all_outputs)
        futures = all_outputs.squeeze(0).permute(1, 0, 2)[:, obs_length:, :]

        # copy x_0_pos into futures
        # futures = x_0_pos.clone().detach().requires_grad_(True).to(history.device)

        return futures


class TransformerGoal(nn.Module):
    """
    Transformer for the Goal Module. This net was used in combination with the standard net (and in particular
    the goal module part of its forward method) to enable the model to use the data from the goal module.
    Now I've replaced it with the TransformerGoalSAR model.

    Args:
        none so far, everything is hardcoded since this is just an auxiliary net

    Methods:
        forward(x_0, goal_point): Tensor : performs forward pass. x_0 is the history path, goal_point is the goal point.
        Returns a tensor of shape (B, 12, 2), containing the predicted future path
    """
    def __init__(self):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.2, max_len=24)
        self.up_x = nn.Linear(8, 64)
        self.up_goal = nn.Linear(1, 64)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, 12)

    def forward(self, x_0, goal_point):
        x_0_pos = x_0[:, :, :2] # (B, 8, 2)

        # Replace nan values with the first non-nan value
        nan_mask = torch.isnan(x_0_pos)
        not_nan_mask = torch.logical_not(nan_mask)
        x_0_pos[nan_mask] = torch.masked_select(x_0_pos, not_nan_mask)[0]
        
        # Permute and upscale the history path
        x_0_pos = x_0_pos.permute(0, 2, 1) # (B, 2, 8)
        x_emb = self.up_x(x_0_pos) # (B, 2, 64)

        # Permute and upscale the goal point
        gp = goal_point.squeeze(0).unsqueeze(1).permute(0, 2, 1) # (B, 2, 1)
        gp_emb = self.up_goal(gp) # (B, 2, 64)

        # Concatenate the two and pass them through the transformer
        x = torch.cat((x_emb, gp_emb), dim=2) # (B, 2, 128)
        x = x.permute(1, 0, 2)
        x = self.pos_emb(x) # (2, B, 128)
        x = self.transformer_encoder(x) # (2, B, 128)
        x = x.permute(1, 0, 2) # (B, 2, 128)
        x = self.linear(x) # (B, 2, 12)
        x = x.permute(0, 2, 1) # (B, 12, 2)
        
        return x
