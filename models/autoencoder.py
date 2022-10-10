import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb

class AutoEncoder(Module):
    """
    Custom AutoEncoder class for the project. In this project, Trajectron++ is used as the temporal-social encoder:
    it encodes the history path and social interaction clues into a state embedding. Then, the decoder (transformer)
    is used for the reverse diffusion process.

    Args:
        config: dict() : configuration infos (retrieved from configuration file)
        encoder: Trajectron : encoder for the model
        diffnet: TransformerConcatLinear in this configuration : from diffusion.py file, read documentation from there
        diffusion: DiffusionTraj : from diffusion.py, read documentation from there

    Methods:
        encode(batch,node_type): Tensor : performs encoding by getting latent representation
        generate(batch,node_type, num_points, sample, bestof,flexibility, ret_traj): ? : generates prediction
        get_loss(batch,node_type): number : returns loss using get_loss method from diffusion.py file
    """
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='cosine'

            )
        )

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z

    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False):

        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        feat_x_encoded = self.encode(batch,node_type) # B * 64
        loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        return loss
