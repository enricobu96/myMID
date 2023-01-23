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
            net = self.diffnet(point_dim=2,
                context_dim=config.encoder_dim,
                tf_layer=config.tf_layer,
                residual=False,
                longterm=self.config.sdd_longterm,
                dataset=self.config['dataset'],
                learn_sigmas=self.config.learn_sigmas,
                use_goal=self.config.use_goal,
                num_image_channels=self.config.g_image_channels,
                obs_len=self.config.g_obs_len,
                pred_len=self.config.g_pred_len,
                sampler_temp=self.config.g_sampler_temperature,
                use_ttst=self.config.g_use_ttst,
            ),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode=self.config.variance_sched,
                cosine_sched=self.config.cosine_sched
            ),
            learn_sigmas=self.config.learn_sigmas,
            lambda_vlb=self.config.lambda_vlb,
            learned_range=self.config.learned_range,
            loss_type=self.config.loss_type,
            ensemble_loss=self.config.ensemble_loss,
            ensemble_hybrid_steps=self.config.ensemble_hybrid_steps,
            use_goal=self.config.use_goal,
            g_loss_lambda=self.config.g_loss_lambda,
            g_weight_samples=self.config.g_weight_samples,
            pretrain_transformer=self.config.pretrain_transformer
        )

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z

    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False):
        goal = batch[-1]
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)

        # Note: the future in sample is used only for pretraining the goal transformer
        predicted_y_vel =  self.diffusion.sample(
            num_points,
            encoded_x,sample,
            bestof,
            flexibility=flexibility,
            ret_traj=ret_traj,
            goal=goal,
            history=batch[1],
            future=batch[2]
            )
            
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         goal) = batch

        feat_x_encoded = self.encode(batch,node_type) # B * 64

        loss = self.diffusion.get_loss(y_t.to(self.config.device), feat_x_encoded, goal=goal, history=x_t)
        
        return loss
