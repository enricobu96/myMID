import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb
from .diffusion_utils import _extract_into_tensor

class VarianceSchedule(Module):
    """
    Class representing the variance schedule. In the original formulation, this class works both for the
    noise schedule and the variance schedule: in fact, the authors followed the DDPM paper, in which the
    sigmas (variances) are not learnt. In the new formulation (TODO), get_sigmas will need to return
    the learnt sigmas (or something like that).

    Args:
        num_steps: int : number of steps in the variance schedule
        mode: str : type of variance schedule: linear or cosine
        beta: number : hyperparameter beta
        beta_T: number : hyperparameter betaT
        cosine_s: number : hyperparameter s for cosine variance schedule

    Methods:
        uniform_sample_t(batch_size): list() : returns a uniform distribution with range [1;number_of_steps+1]
        get_sigmas(t,flexibility): list() : returns sigmas

    Detailed description
        When initialized a VarianceSchedule object, the following happens:
        0. Attributes are initialized
        1. A schedule is decided and the betas are then generated:
            1.1. If the schedule is linear, betas are just a linear space between beta1 and betaT.
            The number of steps is the one given at initialization; betas are in a tensor
            1.2. If the schedule is cosine, uses formula 17 from Improved DDPM paper (https://arxiv.org/pdf/2102.09672.pdf) to compute betas and (first part) alphas
        2. Pads the betas in order to match dimensions
        3. Computes alphas, alpha_logs and sigmas
    """
    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3,cosine_sched='sigmoid_2'): #original cosine_s=8e-3
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        """
        Building the noise schedule - betas, alphas, alpha bars and simgas.
        Note that sigmas has two different meanings/usages:
            - For the non-learned version, sigmas is the variance schedule
            - For the learned version, sigmas_inflex is used as the beta bars in expression 15 IDDPM to compute the variance
        """
        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            if cosine_sched == 'two_fifth_pi_cos_squared':
                betas = two_fifth_pi_cos_squared(cosine_s, num_steps)
            elif cosine_sched == 'piecewise_cos_inv':
                betas = piecewise_cos_inv(cosine_s, num_steps)
            elif cosine_sched == 'clipped_two_fifth':
                betas = clipped_two_fifth(cosine_s, num_steps)
            elif cosine_sched == 'sigmoid':
                betas = sigmoid(cosine_s, num_steps)
            elif cosine_sched == 'sigmoid_2':
                betas = sigmoid_2(cosine_s, num_steps)
            else:
                print('Warning: incorrect cosine schedule, rolling back on sigmoid')
                betas = sigmoid(cosine_s, num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        # Save stuff for the learned version
        self.betas_var = betas
        self.alpha_bars_var = alpha_bars
        self.beta_bars_var = sigmas_inflex

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

        # Compute stuff additional stuff for learned version
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.tensor(np.log( # TODO: fix possibly -inf values
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ))


    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    """
    Gets sigmas for the not-learned version: sigmas are used to compute the variance schedule, which is then constant
    """
    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
    """
    Gets sigmas for the learned version: sigmas_inflex becomes the beta bars and is used to compute expression 15 in IDDPM.
    The expression returns the variances for each step of the DiffusionTraj model.
    """
    def get_sigmas_learning(self, v, t):
        c0 = torch.log(self.betas_var[t]).view(-1, 1, 1).to(v.device)
        c1 = torch.log(self.beta_bars_var[t]).view(-1, 1, 1).to(v.device)
        sigmas = torch.exp(v*c0 + (1-v)*c1)
        return sigmas

class TransformerConcatLinear(Module):
    """
    TransformerConcatLinear class. This is a very crucial part of the project, since it's the Transformer
    model used for the decoding part. It inherits from torch module, i.e. the init method creates the network
    (and it therefore represents the decoder network itself), and the forward method represents a forward pass
    in the model.

    Args:
        residual: bool : wether to use residual connections (?). Set to False in this project
        pos_emb: PositionalEncoding : positional encoding layer. Read documentation of the class, but the idea is that this manages the "ordinality" of states
        concat1: ConcatSquashLinear : concat squash linear layer. Read documentation of the class (it's a custom class, it doesn't come with standard pytorch)
        layer: TransformerEncoderLayer : encoder layer made up of self-attention and a ffnn. Based on the original transformer paper
        transformer_encoder: TransformerEncoder : stack of n encoder layers
        concat3: ConcatSquashLinear : same as above
        concat4: ConcatSquashLinear : same as above
        linear: ConcatSquashLinear : same as above

    Methods:
    forward(x,beta,context): nn.Linear : forward pass for the decoder model
    """
    def __init__(self, point_dim, context_dim, tf_layer, residual, longterm=False, dataset=None, learn_sigmas=False):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=30 if longterm and dataset=='sdd' else 24)
        self.concat1 = ConcatSquashLinear(2,2*context_dim,context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 2 if not learn_sigmas else 4, context_dim+3)

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        x = self.concat1(ctx_emb,x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

class DiffusionTraj(Module):
    """
    DiffusionTraj class, used as diffusion model for trajectories (crucial part of the project).
    This contains in turn the net (in this case TransformerConcatLinear) and the variance schedule.
    """
    def __init__(self, net, var_sched:VarianceSchedule, learn_sigmas=False, lambda_vlb=1e-4, device='cpu'):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.learn_sigmas = learn_sigmas
        self.lambda_vlb = lambda_vlb
        self.device = device

    def get_loss(self, x_0, context, t=None):
        # Compute LOSS_SIMPLE (old code)
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        if self.device == 'cpu':
            beta = self.var_sched.betas[t].to('cpu')
            c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).to('cpu')    # (B, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).to('cpu')   # (B, 1, 1)
            e_rand = torch.randn_like(x_0).to('cpu')  # (B, N, d)
        else:
            beta = self.var_sched.betas[t].cuda()
            c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()    # (B, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)
            e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)

        out = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        if self.learn_sigmas:
            e_theta, variance_v = out.split(2, dim=2)
            sigmas = self.var_sched.get_sigmas_learning(variance_v, t)
        else:
            e_theta = out

        loss_simple = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')

        if self.learn_sigmas:
            # Compute LOSS_VLB
            loss_vlb = self.loss_vlb(
                mean=e_theta,
                sigma=sigmas,
                x_start = x_0,
                x_t = c0 * x_0 + c1 * e_rand,
                t=t
                )
            print('LOSS', loss_vlb)
            loss = loss_simple + self.lambda_vlb*loss_vlb
        else:
            loss = loss_simple
            print('LOSS', loss)

        return loss

    def sample(self, num_points, context, sample, bestof, point_dim=2, flexibility=0.0, ret_traj=False):
        traj_list = []
        for i in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
            traj = {self.var_sched.num_steps: x_T}
            for t in range(self.var_sched.num_steps, 0, -1):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t]*batch_size]
                out = self.net(x_t, beta=beta, context=context)

                if self.learn_sigmas:
                    e_theta, variance_v = out.split(2, dim=2)
                    sigmas = self.var_sched.get_sigmas_learning(variance_v, t)

                else:
                    e_theta = out

                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
        return torch.stack(traj_list)
    
    """
    Functions used to compute the vlb loss for learning variances
    """
    def loss_vlb(self, mean, sigma, x_start, x_t, t):
        # Learn variances using the variational bound but without letting it affect the mean
        loss = self._vb_terms_bpd(mean, sigma, x_start, x_t, t)
        return loss

    def _vb_terms_bpd(self, mean, sigma, x_start, x_t, t):
        true_mean, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        log_variance = torch.log(sigma)
        kl = self.normal_kl(true_mean, true_log_variance_clipped, mean, log_variance)
        kl = self.mean_flat(kl) / np.log(2.0)

        decoder_nll = -self.discretized_gaussian_log_likelihood(
            x_start, means=mean, log_scales=0.5 * log_variance
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = self.mean_flat(decoder_nll) / np.log(2.0)
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((torch.tensor(t).to(x_start.device) == 0), decoder_nll, kl)
        # output = torch.mean(decoder_nll)
        return output

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self.var_sched.posterior_mean_coef1[t].view(-1,1,1).to(x_start.device) * x_start
            + self.var_sched.posterior_mean_coef2[t].view(-1,1,1).to(x_start.device) * x_t
        )
        posterior_log_variance_clipped = self.var_sched.posterior_log_variance_clipped[t]
        assert (
            posterior_mean.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_log_variance_clipped
    
    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        logvar1 = logvar1.view(-1,1,1).to(mean1.device)
        return 0.5 * (
            -1.0
            + logvar2 - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def discretized_gaussian_log_likelihood(self, x, *, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs
    
    def approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

"""
NOISE SCHEDULE FUNCTIONS
"""
def two_fifth_pi_cos_squared(cosine_s, num_steps):
    betas = []
    f0 = math.cos(cosine_s/(1+cosine_s) * (2*math.pi/5)) ** 2
    for i in range(1, num_steps+1):
        tT = i/num_steps
        ft = math.cos((tT+cosine_s)/(1+cosine_s) * (2*math.pi/5)) ** 2
        alphat = ft/f0
        tTm1 = (i-1)/num_steps
        ftm1 = math.cos((tTm1+cosine_s)/(1+cosine_s) * (2*math.pi/5)) ** 2
        alphatm1 = ftm1/f0
        betas.append(min(1-(alphat/alphatm1), 0.999))
    betas = torch.Tensor(betas)
    return betas

def piecewise_cos_inv(cosine_s, num_steps):
    betas = [0]
    a = 2
    for i in range(1, num_steps+1):
        c_prev = math.cos(((betas[i-1]/num_steps + cosine_s)/(1+cosine_s))*math.pi/a)
        acos = np.clip((1-i)*c_prev, -1, 1)
        betas.append((math.acos(acos)*(a/math.pi)*(1+cosine_s) - cosine_s)*num_steps)
    betas = torch.Tensor(betas)
    return betas

def clipped_two_fifth(cosine_s, num_steps):
    betas = []
    clips = []
    for i in range(-num_steps//2, num_steps//2):
        f = (1/(-np.sign(i)*(num_steps//2)))*i+1
        clips.append(f/10)
    clips = np.nan_to_num(clips, nan=1)
    f0 = math.cos(cosine_s/(1+cosine_s) * (2*math.pi/5)) ** 2
    for i in range(1, num_steps+1):
        tT = i/num_steps
        ft = math.cos((tT+cosine_s)/(1+cosine_s) * (2*math.pi/5)) ** 2
        alphat = ft/f0
        tTm1 = (i-1)/num_steps
        ftm1 = math.cos((tTm1+cosine_s)/(1+cosine_s) * (2*math.pi/5)) ** 2
        alphatm1 = ftm1/f0
        betas.append(min(1-(alphat/alphatm1)+clips[i-1], 0.999))
    betas = torch.Tensor(betas)
    return betas

def sigmoid(cosine_s, num_steps):
    lambd = 6
    eps = 0.05
    norm = 10
    s = lambda x : 1/(1+math.exp(-lambd*(x-eps)))
    betas = []
    for i in range(-num_steps//2, num_steps//2):
        betas.append(s(i/(num_steps))/norm)
    betas = torch.Tensor(betas)
    return betas

def sigmoid_2(cosine_s, num_steps):
    lambd = 16
    eps = .7
    norm = 16
    s = lambda x : 1/(1+math.exp(-lambd*(x-eps)))
    betas = []
    for i in range(-num_steps//2, num_steps//2):
        betas.append(s(i/(num_steps))/norm)
    betas = torch.Tensor(betas)
    return betas

"""
UNUSED/UNIMPORTANT STUFF
"""
class TransformerLinear(Module):
    """
    TransformerLinear class, apparently not used in this project; TransformerConcatLinear is used instead.
    """
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim+3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):
        print('helo')
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1,0,2)
        #pdb.set_trace()
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(1,0,2)   # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)

class LinearDecoder(Module):
    """
    LinearDecoder class, apparently not used in this project; TransformerConcatLinear is used instead.
    """
    def __init__(self):
            super().__init__()
            self.act = F.leaky_relu
            self.layers = ModuleList([
                #nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                #nn.Linear(2, 64),
                #nn.Linear(2, 64),
            ])
    def forward(self, code):

        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out

class TrajNet(Module):
    """
    TrajNet class, apparently not used in this project; TransformerConcatLinear is used instead.
    """
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(2, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 2, context_dim+3),

        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out