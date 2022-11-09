import torch
import numpy as np
from copy import deepcopy

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def _vb_terms_bpd(mean, sigma, x_start, x_t, t, pmc1, pmc2, plvc):
    """
    Compute terms for the variational lower bound -> return the final output, i.e. the loss (L_vlb).
    At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    """
    true_mean, true_log_variance_clipped = q_posterior_mean_variance(
        x_start=x_start,
        x_t=x_t,
        t=t,
        pmc1=pmc1,
        pmc2=pmc2,
        plvc=plvc
        )
    log_variance = sigma
    kl = normal_kl(true_mean, true_log_variance_clipped, mean, log_variance)
    kl = mean_flat(kl) / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
        x_start, means=mean, log_scales=0.5 * log_variance
    )
    decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

    output = torch.where((torch.tensor(t).to(x_start.device) == 0), decoder_nll, kl)
    output = torch.mean(output)

    return output

def q_posterior_mean_variance(x_start, x_t, t, pmc1, pmc2, plvc):
    """
    Compute the mean and variance of the diffusion posterior: q(x_{t-1}|x_t,x_0)
    """
    posterior_mean = (
        pmc1[t].view(-1,1,1).to(x_start.device) * x_start
        + pmc2[t].view(-1,1,1).to(x_start.device) * x_t
    )
    posterior_log_variance_clipped = plvc[t]
    return posterior_mean, posterior_log_variance_clipped

"""
Utility functions.
- normal_kl(mean1, logvar1, mean2, logvar2) : KL divergence between two normal distributions
- mean_flat(tensor): mean over all non-batch dimensions
- discretized_gaussian_log_likelihood (x, means, log_scales): log likelihood of a gaussian distribution to the input data.
  This function has been modified: now it applies logarithm to the +/- 0.001*torch.abs(torch.min(x)) value instead of just 0.001.
- approx_standard_normal_cdf(x): (fast) approximation of the standard normal cdf
"""
def normal_kl(mean1, logvar1, mean2, logvar2):
    logvar1 = logvar1.view(-1,1,1).to(mean1.device)
    return 0.5 * (
        -1.0
        + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Version 0 - original one
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
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
    
    # # Version 1 of trying to fix it for the case - min(x) < discretization < max(x)
    # assert x.shape == means.shape == log_scales.shape
    # centered_x = x - means
    # min_x = torch.min(x) + 0.001*torch.abs(torch.min(x))
    # max_x = torch.max(x) - 0.001*torch.abs(torch.max(x))
    # inv_stdv = torch.exp(-log_scales)
    # plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    # cdf_plus = approx_standard_normal_cdf(plus_in)
    # min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    # cdf_min = approx_standard_normal_cdf(min_in)
    # log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    # log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    # cdf_delta = cdf_plus - cdf_min
    # log_probs = torch.where(
    #     x < min_x,
    #     log_cdf_plus,
    #     torch.where(x > max_x, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    # )
    # assert log_probs.shape == x.shape
    # return log_probs

    # # Version 2 of trying to fix it for the case - rescaling x to be in [-1,1] just before returning the log likelihood
    # assert x.shape == means.shape == log_scales.shape
    # centered_x = x - means
    # min_x = torch.min(x) + 0.001*torch.abs(torch.min(x))
    # max_x = torch.max(x) - 0.001*torch.abs(torch.max(x))
    # inv_stdv = torch.exp(-log_scales)
    # plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    # cdf_plus = approx_standard_normal_cdf(plus_in)
    # min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    # cdf_min = approx_standard_normal_cdf(min_in)
    # log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    # log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    # cdf_delta = cdf_plus - cdf_min
    # rescaled_x = deepcopy(x)
    # # rescale third dimension to be in [-1,1]
    # rescaled_x[:,:,0] = (rescaled_x[:,:,0] - min_x) / (max_x - min_x) if min_x != max_x else rescaled_x[:,:,0]
    # rescaled_x[:,:,0] = 2*rescaled_x[:,:,0] - 1 if min_x != max_x else rescaled_x[:,:,0]
    # log_probs = torch.where(
    #     rescaled_x < -0.999,
    #     log_cdf_plus,
    #     torch.where(rescaled_x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    # )
    # assert log_probs.shape == x.shape
    # return log_probs

    # # Version 3 of trying to fix it for the case - rescaling x to be in [-1,1] before doing everything else
    # assert x.shape == means.shape == log_scales.shape
    # min_x = torch.min(x) + 0.001*torch.abs(torch.min(x))
    # max_x = torch.max(x) - 0.001*torch.abs(torch.max(x))
    # x[:,:,0] = (x[:,:,0] - min_x) / (max_x - min_x) if min_x != max_x else x[:,:,0]
    # x[:,:,0] = 2*x[:,:,0] - 1 if min_x != max_x else x[:,:,0]
    # centered_x = x - means
    # inv_stdv = torch.exp(-log_scales)
    # plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    # cdf_plus = approx_standard_normal_cdf(plus_in)
    # min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    # cdf_min = approx_standard_normal_cdf(min_in)
    # log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    # log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    # cdf_delta = cdf_plus - cdf_min
    # log_probs = torch.where(
    #     x < -0.999,
    #     log_cdf_plus,
    #     torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    # )
    # assert log_probs.shape == x.shape
    # return log_probs

    # # Version 4 of trying to fix it for the case - just don't differentiate and always return the delta
    # assert x.shape == means.shape == log_scales.shape
    # centered_x = x - means
    # inv_stdv = torch.exp(-log_scales)
    # plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    # cdf_plus = approx_standard_normal_cdf(plus_in)
    # min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    # cdf_min = approx_standard_normal_cdf(min_in)
    # cdf_delta = cdf_plus - cdf_min
    # log_probs = cdf_delta
    # assert log_probs.shape == x.shape
    # return log_probs

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))