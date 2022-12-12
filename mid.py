import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle
from matplotlib import pyplot as plt
import pandas as pd

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation

import wandb
from evaluation.evaluation import compute_kde_nll
from evaluation.trajectory_utils import prediction_output_to_trajectories
from evaluation.visualization.visualization import plot_wandb
from scipy.stats import gaussian_kde

from utils.resample import UniformSampler, LossSecondMomentResampler

class MID():
    """
    Motion Indeterminacy Diffusion model from MID paper. The pipeline (when training)
    is the following:
    1. The object MID is created (by main.py)
    2. At instantiation, the following components are built:
        2.1. The encoder, i.e. a Trajectron object is instantiated [trajectron.py]
        2.2. The model, i.e. an Autoencoder object is instantiated with the previously
             instantiated Trajectron object as encoder [autoencoder.py]
        2.3. The loaders and the optimizer
    3. For each epoch, training is performed:
        3.1. Standard zero_grad
        3.2. get_loss, but this time it's a custom one in AutoEncoder object [autoencoder.py]
        3.3. Standard backprop and optimizer step
    4. Every now and then (specified in the config file) also evaluation with validation set

    Now, the important part is 2.1 and 2.2: this is the model, so start from those
    files and follow documentation to the other important files.
    
    The order is (files): (trajectron (mgcvae)) (autoencoder (diffusion (transformer)))
    Note: trajectron and mgcvae are not crucial, since MID is encoder-agnostic and trajectron
    is used just because of its representative power; however, trajectron should be fixed to 
    address the issue reported on github.

    In the paper (and therefore in the experiments) they used the following structure of objects:
        (Trajectron (MultimodalGenerativeCVAE)) -> temporal-social encoder. This "just" encodes history path and
        social interaction clues into a state embedding
        (AutoEncoder (Trajectron, DiffusionTraj (TransformerConcatLinear, VarianceSchedule))) -> AutoEncoder object
        contains:
            - The encoder, which is the Trajectron previously created (containing in turn MultimodalGenerativeCVAE)
            - The diffusion model (DiffusionTraj), which contains the transformer model (TransformerConcatLinear) and
            the variance schedule (VarianceSchedule).

    Args:
        config: dict() : configuration object got from config file
    
    Methods
    -------
    train(): None : trains the model
    eval(): None : evaluates the model
    _build(): None : builds dir, optimizer, encoder, loader, model
    _build_dir(): None : builds dir
    _build_optimizer(): None : builds optimizer
    _build_encoder_config(): None : builds encoder configuration
    _build_encoder(): None : builds encoder
    _build_model(): None : builds model
    _build_train_loader(): None : builds loader for training phase
    _build_eval_loader(): None : builds loader for evaluation phase
    _build_offline_scene_graph(): None : builds offline scene graph if that hyperparameter is set to yes
    """
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

    # def _get_scenes(self, data_loader, batch_acc, batch_size):
    #     scenes = []
    #     for i in range(batch_size):
    #         scene = data_loader.dataset.get_scene(batch_acc+i)
    #         scenes.append(scene)
    #     return scenes

    def train(self):
        """
        Method for model training
        """
        torch.cuda.empty_cache()

        if wandb.run is None and self.config.use_wandb:
            wandb.init(settings=wandb.Settings(start_method="thread"),
                       project="myMID", config=self.config,
                       group=self.config['dataset'],
                       job_type=self.config['dataset'],
                       tags=None, name=None)

        if self.config.save_losses_plots:
            it_steps = []
            i = 0
            log_loss = []
            log_fake_loss = []
            kde_lls = []
            fake_loss = nn.MSELoss(reduction='mean')
        for epoch in range(1, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                batch_acc = 0
                for batch in pbar:
                    # Resampling stuff
                    t_idx, weights = self.schedule_resampler.sample(
                        batch_size=len(batch[0]),
                        device=self.config.device
                        )
                    self.optimizer.zero_grad()
                    # scenes = self._get_scenes(data_loader, batch_acc, len(batch[0]))
                    
                    # for i in range(len(batch[0])):
                    #     fig, ax = plt.subplots(figsize=(24, 12))
                    #     ax.imshow(scenes[i].map.as_image(), alpha=0.7, origin='upper')
                    #     fig.savefig('testimages/'+str(batch_acc%256)+'_'+str(i)+'.png')
                    #     plt.close()

                    batch_acc += len(batch[0])

                    losses = self.model.get_loss(batch, node_type)
                    
                    if not self.config.reduce_grad_noise:
                        train_loss = torch.mean(losses)
                    else:
                        self.schedule_resampler.update_with_local_losses(t_idx, losses.detach())
                        train_loss = (losses * weights).mean()
                    
                    if self.config.save_losses_plots:
                        with torch.no_grad():
                            it_steps.append(i)
                            log_loss.append(np.log(train_loss.item()))
                            i += 1
                            traj_pred = self.model.generate(batch, node_type, num_points=self.hyperparams['prediction_horizon'], sample=1,bestof=False)
                            # Compute KDE
                            num_batches = traj_pred[0].shape[0]
                            num_timesteps = traj_pred[0].shape[1]
                            kde_ll = 0.
                            log_pdf_lower_bound = -20


                            for batch_num in range(num_batches):
                                for t in range(num_timesteps):
                                    try:
                                        kde = gaussian_kde(traj_pred[0][batch_num, t, :].T)
                                        pdf = np.clip(kde.logpdf(batch[2][batch_num, t, :].cpu().numpy().T), log_pdf_lower_bound, np.inf)[0]
                                        kde_ll += pdf / (num_batches * num_timesteps)
                                    except np.linalg.LinAlgError:
                                        kde_ll = np.nan
                                    except ValueError:
                                        kde_ll = np.nan

                            kde_ll = -kde_ll
                            # Compute fake loss
                            traj_pred = torch.FloatTensor(traj_pred[0])
                            traj_pred[0] = torch.FloatTensor(traj_pred[0])
                            fake_loss_res = fake_loss(traj_pred, batch[2])
                            log_fake_loss.append(np.log(fake_loss_res.item()))
                            kde_lls.append(kde_ll)
                        pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f} FAKE_LOSS: {fake_loss_res.item():.2f} KDE: {kde_ll:.2f}")
                    else:
                        pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}")
                    train_loss.backward()
                    self.optimizer.step()

            self.train_dataset.augment = False
            if epoch % self.config.eval_every == 0:
                self.model.eval()

                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []
                eval_kde_batch_errors = []

                ph = self.hyperparams['prediction_horizon']
                min_hl = self.hyperparams['minimum_history_length'] if self.config.sdd_longterm and self.config['dataset'] == 'sdd' else 7
                max_hl = self.hyperparams['maximum_history_length']

                """
                min_ht = minimum history timesteps
                max_ht = maximum history timesteps
                min_ft = minimum future timesteps (prediction horizon)
                max_ft = maximum future timesteps (predition horizon)
                """
                sc = 1
                for i, scene in enumerate(self.eval_scenes):
                    print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                    for t in tqdm(range(0, scene.timesteps, 10)):
                        timesteps = np.arange(t,t+10)
                        batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                                       pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                                       min_ht=min_hl, max_ht=max_hl, min_ft=12, max_ft=ph, hyperparams=self.hyperparams)
                        if batch is None:
                            continue
                        test_batch = batch[0]
                        nodes = batch[1]
                        timesteps_o = batch[2]
                        traj_pred = self.model.generate(test_batch, node_type, num_points=ph, sample=20,bestof=True) # B * 20 * 12 * 2
                        predictions = traj_pred
                        predictions_dict = {}
                        for i, ts in enumerate(timesteps_o):
                            if ts not in predictions_dict.keys():
                                predictions_dict[ts] = dict()
                            predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))

                        batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                               scene.dt,
                                                                               max_hl=max_hl,
                                                                               ph=ph,
                                                                               node_type_enum=self.eval_env.NodeType,
                                                                               kde=True,
                                                                               map=None,
                                                                               best_of=True,
                                                                               prune_ph_to_future=True)

                        eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                        eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))
                        eval_kde_batch_errors = np.hstack((eval_kde_batch_errors, batch_error_dict[node_type]['kde']))

                    """
                    WANDB VISUALIZATION
                    """
                    if self.config.use_wandb:
                        fig, ax = plt.subplots(figsize=(24, 12))
                        fig, ax = plot_wandb(fig, ax, predictions_dict, scene.dt, max_hl, ph, map=scene.map, mean_x=scene.mean_x, mean_y=scene.mean_y)
                        plt.legend(loc='best')
                        try:
                            os.makedirs('images')
                        except OSError:
                            if not os.path.isdir('images'):
                                raise
                        plt.savefig('images/'+self.config["dataset"]+'_train_traj_epoch'+str(epoch)+'_scene'+str(sc)+'.png')
                        wandb.log({"train/traj_image": wandb.Image(fig), "scene": str(sc)}, step=epoch)
                        sc += 1
                        plt.close()

                ade = np.mean(eval_ade_batch_errors)
                fde = np.mean(eval_fde_batch_errors)
                kde = np.mean(eval_kde_batch_errors)

                if self.config.dataset == "eth":
                    """
                    Check https://arxiv.org/pdf/1907.08752.pdf to understand why /0.6
                    """
                    ade = ade/0.6
                    fde = fde/0.6
                elif self.config.dataset == "sdd":
                    ade = ade * 50
                    fde = fde * 50

                """WANDB LOGGING"""
                train_metrics = {'ade': ade,
                                 'fde': fde}
                if self.config.use_wandb:
                    wandb.log({'learning_rate': self.optimizer.param_groups[0]['lr']}, step=epoch)
                    wandb.log(train_metrics, step=epoch)
                
                print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde} KDE: {kde}")
                self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")

                # Saving model
                checkpoint = {
                    'encoder': self.registrar.model_dict,
                    'ddpm': self.model.state_dict()
                 }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))

                self.model.train()

        if self.config.save_losses_plots:
            plot_filename = 'xy_'
            if self.config.loss_type == 'hybrid':
                plot_filename += 'hybrid'
            elif self.config.loss_type == 'vlb':
                if self.config.ensemble_loss:
                    plot_filename += 'ensemble'
                else:
                    if self.config.reduce_grad_noise:
                        plot_filename += 'vlb_is'
                    else:
                        plot_filename += 'vlb_normal'

            df = pd.DataFrame()
            df['iterations'] = it_steps
            df['log_loss'] = log_loss
            df['log_fake_loss'] = log_fake_loss
            df['kde_ll'] = kde_lls

            df.to_csv('docs/plot_losses/'+plot_filename+'.csv', index=False)

    def eval(self):
        torch.cuda.empty_cache()
        
        epoch = self.config.eval_at

        if wandb.run is None and self.config.use_wandb:
            wandb.init(settings=wandb.Settings(start_method="thread"),
                       project="myMID", config=self.config,
                       group=self.config['dataset'],
                       job_type=self.config['dataset'],
                       tags=None, name=None)

        for j in range(5):

            node_type = "PEDESTRIAN"
            eval_ade_batch_errors = []
            eval_fde_batch_errors = []
            eval_kde_batch_errors = []

            ph = self.hyperparams['prediction_horizon']
            max_hl = self.hyperparams['maximum_history_length']
            min_hl = self.hyperparams['minimum_history_length'] if self.config.sdd_longterm and self.config['dataset'] == 'sdd' else 7
            sc = 1
            for i, scene in enumerate(self.eval_scenes):
                print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                for t in tqdm(range(0, scene.timesteps, 10)):
                    timesteps = np.arange(t,t+10)
                    batch = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps,
                                            node_type=node_type, state=self.hyperparams['state'],
                                            pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                                            min_ht=min_hl, max_ht=max_hl, min_ft=12,
                                            max_ft=ph, hyperparams=self.hyperparams)
                    if batch is None:
                        continue
                    test_batch = batch[0]
                    nodes = batch[1]
                    timesteps_o = batch[2]
                    traj_pred = self.model.generate(test_batch, node_type, num_points=ph, sample=20,bestof=True) # B * 20 * 12 * 2
                    predictions = traj_pred
                    predictions_dict = {}
                    for i, ts in enumerate(timesteps_o):
                        if ts not in predictions_dict.keys():
                            predictions_dict[ts] = dict()
                        predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))



                    batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                           scene.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=self.eval_env.NodeType,
                                                                           kde=True,
                                                                           map=None,
                                                                           best_of=True,
                                                                           prune_ph_to_future=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))
                    eval_kde_batch_errors = np.hstack((eval_kde_batch_errors, batch_error_dict[node_type]['kde']))

                """
                WANDB VISUALIZATION
                """
                if self.config.use_wandb:
                        fig, ax = plt.subplots(figsize=(24, 12))
                        fig, ax = plot_wandb(fig, ax, predictions_dict, scene.dt, max_hl, ph, map=scene.map, mean_x=scene.mean_x, mean_y=scene.mean_y)
                        plt.legend(loc='best')
                        try:
                            os.makedirs('images')
                        except OSError:
                            if not os.path.isdir('images'):
                                raise
                        plt.savefig('images/'+self.config["dataset"]+'_test_traj_epoch'+str(epoch)+'_scene'+str(sc)+'_it'+str(j)+'.png', bbox_inches='tight')
                        wandb.log({"train/test_image": wandb.Image(fig), "scene": str(sc)}, step=epoch)
                        sc += 1
                        plt.close()

            ade = np.mean(eval_ade_batch_errors)
            fde = np.mean(eval_fde_batch_errors)
            kde = np.mean(eval_kde_batch_errors)

            if self.config.dataset == "eth":
                ade = ade/0.6
                fde = fde/0.6
            elif self.config.dataset == "sdd":
                ade = ade * 50
                fde = fde * 50
            
            """WANDB LOGGING"""
            train_metrics = {'ade': ade,
                             'fde': fde}
            if self.config.use_wandb:
                wandb.log(train_metrics, step=epoch)

            print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde} KDE: {kde}")
        #self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")


    def _build(self):
        """
        Builds everything (general control function).
        """
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_resampler()
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        """
        Builds experiments directory (contains then log and model binary files)
        """
        self.model_dir = osp.join("./experiments",self.config.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        """
        Builds optimizer, which is Adam as reported in the paper.
        Sets scheduler as ExponentialLR (from torch documentation:
        decays the learning rate of each parameter group by gamma
        every epoch).
        """
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self):
        """
        Builds configuration for encoder:
        - Sets hyperparameters
        - Builds registrar (see documentation, but nothing crucial)
        - Opens environments
        """
        self.hyperparams = get_traj_hypers(self.config.sdd_longterm, self.config['dataset'])
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registar
        self.registrar = ModelRegistrar(self.model_dir, self.config.device)

        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            self.checkpoint = torch.load(osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"), map_location = "cpu")

            self.registrar.load_models(self.checkpoint['encoder'])


        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(self.eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')

    def _build_encoder(self):
        """
        Builds encoder and sets environment. Encoder is the trajectron, strongly recommended
        to read documentation for it.
        """
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.config.device)

        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()

    def _build_model(self):
        """
        Builds model and redirect it to device (cuda/cpu). Model is an autoencoder (see documentation) with Trajectron as encoder.
        """
        config = self.config
        model = AutoEncoder(config, encoder = self.encoder)
        if self.config.device=='cpu' and self.config.eval_device=='cpu':
            self.model = model.to('cpu')
        else:
            self.model = model.cuda()

        if self.config.eval_mode:
            self.model.load_state_dict(self.checkpoint['ddpm'])
        print("> Model built!")

    """
    Resampler stuff
    Build the resampler for reducing gradient noise
    """
    def _build_resampler(self):
        if self.config.reduce_grad_noise:
            self.schedule_resampler = LossSecondMomentResampler(self.model.diffusion.var_sched)
        else:
            self.schedule_resampler = UniformSampler(self.model.diffusion.var_sched)

    def _build_train_loader(self):
        """
        Builds loader for training:
        - Sets attention radius and node types
        - Sets scenes
        - Sets training dataset (EnvironmentDataset, see documentation but not crucial)
        - Sets dataloaders
        """
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

        self.train_dataset = EnvironmentDataset(train_env,
                                           self.hyperparams['state'],
                                           self.hyperparams['pred_state'],
                                           scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
                                           node_freq_mult=self.hyperparams['node_freq_mult_train'],
                                           hyperparams=self.hyperparams,
                                           min_history_timesteps=1,
                                           min_future_timesteps=self.hyperparams['prediction_horizon'],
                                           return_robot=not self.config.incl_robot_node)
        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory = True,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=True, #TODO: if performances are bad, shuffle and find a way to get the scenes in order
                                                         num_workers=self.config.preprocess_workers)
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader


    def _build_eval_loader(self):
        """
        Builds loader for evaluation. Same functioning of _build_train_loader().
        """
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(self.eval_env,
                                              self.hyperparams['state'],
                                              self.hyperparams['pred_state'],
                                              scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                                              node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                                              hyperparams=self.hyperparams,
                                              min_history_timesteps=self.hyperparams['minimum_history_length'],
                                              min_future_timesteps=self.hyperparams['prediction_horizon'],
                                              return_robot=not config.incl_robot_node)
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                             collate_fn=collate,
                                                             pin_memory=True,
                                                             batch_size=config.eval_batch_size,
                                                             shuffle=True,
                                                             num_workers=config.preprocess_workers)
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        """
        Builds scene graphs for offline calculating. Not used in the paper model.
        """
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")
