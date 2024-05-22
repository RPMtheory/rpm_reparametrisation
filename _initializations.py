# Imports
import copy
import torch
import numpy as np
import torch.nn.functional as F

import recognition
from data import RPMData
from flexible_multivariate_normal import vector_to_tril_diag_idx


def _default_field(param: dict, key: str, default):
    """Set Dictionaries default param if not provided"""
    if key not in param:
        param[key] = default


def _repeat_list(val, num):
    """Repeat element in list"""
    return [val for _ in range(num)]


class Mixin:
    """Mixin class containing necessary methods for initializing fast RPM model"""

    def _init_fit_params(self):
        """ Default Fit parameters """

        # Init dictionary
        if self.fit_params is None:
            self.fit_params = {}

        # Number of conditionally independent factors
        num_factors = self.num_factors

        # Default optimizer / scheduler
        optimizer_closure_default = lambda params: torch.optim.Adam(params, lr=1e-3)
        scheduler_closure_default = lambda optim: torch.optim.lr_scheduler.ConstantLR(optim, factor=1)

        # Latent dimensions
        _default_field(self.fit_params, key='dim_latent', default=1)

        # Iterations
        _default_field(self.fit_params, key='num_epoch', default=500)

        # Batch size (default to full batch)
        _default_field(self.fit_params, key='batch_size', default=self.num_observation)

        # Constrain auxiliary
        _default_field(self.fit_params, key='auxiliary_mode', default='constrained_prior')

        # Logger verbosity (percentage of iteration)
        _default_field(self.fit_params, key='pct', default=0.01)

        # Prior Parameters
        _default_field(self.fit_params, key='prior_params', default={})
        _default_field(self.fit_params['prior_params'], key='alpha', default=1)
        _default_field(self.fit_params['prior_params'], key='num_centroids', default=1)
        _default_field(self.fit_params['prior_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['prior_params'], key='scheduler', default=scheduler_closure_default)

        # Auxiliary Parameters
        _default_field(self.fit_params, key='auxiliary_params', default={})
        _default_field(self.fit_params['auxiliary_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['auxiliary_params'], key='scheduler', default=scheduler_closure_default)

        # Network Parameters
        _default_field(self.fit_params, key='factors_params', default={})
        _default_field(self.fit_params['factors_params'], key='channels', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_conv', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_pool', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='dim_hidden', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='non_linearity', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['factors_params'], key='dropout',  default=_repeat_list(0.0, num_factors))
        _default_field(self.fit_params['factors_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['factors_params'], key='scheduler', default=scheduler_closure_default)

        # Annealing Schemes
        _default_field(self.fit_params, key='annealing_a', default=lambda x: 1)
        _default_field(self.fit_params, key='annealing_b', default=lambda x: 1)

    def _init_prior(self):
        """ Prior Distribution: (mixture of) FlexibleMultivariateNormal """

        # Grasp Params
        params = self.fit_params['prior_params']

        if self.prior is None:

            # Hyperparams
            num_centroids = params['num_centroids']
            dim_latent = self.dim_latent

            # 1st Natural Parameters (with approx. uniform spacing)
            natural1, _ = _init_centroids(
                num_centroids,
                dim_latent,
                ite_max=1000,
                optimizer=lambda x: torch.optim.Adam(x, lr=1e-2),
            )

            # 2nd Natural Parameters (Vectorize Cholesky Decomposition)
            diag_idx = vector_to_tril_diag_idx(dim_latent)

            natural2_chol_vec = np.zeros((num_centroids, int(self.dim_latent * (self.dim_latent + 1) / 2)))
            natural2_chol_vec[:, diag_idx] = np.sqrt(0.5)
            natural2_chol_vec = torch.tensor(natural2_chol_vec)

            # Mixture weights
            responsibilities = torch.ones(num_centroids) / num_centroids


            # Prior
            self.prior = recognition.Mixture(
                responsibilities,
                natural1,
                natural2_chol_vec,
                diagonal_covariance=False,
            ).to(self.device.index).to(self.dtype)

        num_centroids = self.prior.num_centroids
        self.alpha = torch.tensor([params['alpha']], device=self.device, dtype=self.dtype)
        self.beta_constant = torch.lgamma(num_centroids * self.alpha) - num_centroids * torch.lgamma(self.alpha)

    def _init_factors(self, observations: RPMData):
        """Recognition Functions"""

        if self.recognition_factors is None:

            fit_params = self.fit_params['factors_params']

            # Diagonal Indices
            diag_idx = vector_to_tril_diag_idx(self.dim_latent)

            # Input dimensions
            dim_inputs = observations.dim_observation

            # Problem dimensions
            dim_latent = self.dim_latent
            num_factors = self.num_factors

            # Convolutional parameters
            channels = fit_params["channels"]
            kernel_conv = fit_params["kernel_conv"]
            kernel_pool = fit_params["kernel_pool"]
            dropout = fit_params["dropout"]

            # Fully connected layers parameters
            dim_hidden = fit_params["dim_hidden"]
            non_linearity = fit_params["non_linearity"]

            # Build and Append networks
            rec = []
            for obsi in range(num_factors):

                # Init Vectorized Cholesky decomposition of the negative precision
                chol_vec = np.zeros(int(self.dim_latent * (self.dim_latent + 1) / 2))
                chol_vec[diag_idx] = np.sqrt(0.5)
                chol_vec = torch.tensor(chol_vec, dtype=self.dtype)

                neti = recognition.Net(
                    dim_input=dim_inputs[obsi],
                    dim_latent=dim_latent,
                    precision_chol_vec=chol_vec,
                    kernel_conv=kernel_conv[obsi],
                    kernel_pool=kernel_pool[obsi],
                    channels=channels[obsi],
                    dim_hidden=dim_hidden[obsi],
                    non_linearity=non_linearity[obsi],
                    dropout=dropout[obsi],
                ).to(self.device.index).to(self.dtype)
                rec.append(neti)

            self.recognition_factors = rec

    def _init_auxiliary(self):

        natural1 = torch.zeros(
            self.prior.num_centroids,
            self.num_factors,
            self.num_observation,
            self.dim_latent,
        )

        natural2_chol_vec = torch.zeros(
            self.prior.num_centroids,
            self.num_factors,
            self.num_observation,
            int(self.dim_latent * (self.dim_latent + 1) / 2)
        )

        # 2nd Natural Parameters (Vectorize Cholesky Decomposition)
        diag_idx = vector_to_tril_diag_idx(self.dim_latent)

        natural2_chol_vec = np.zeros(
            (self.prior.num_centroids,
            self.num_factors,self.num_observation,
            int(self.dim_latent * (self.dim_latent + 1) / 2))
        )
        natural2_chol_vec[..., diag_idx] = np.sqrt(0.5)
        natural2_chol_vec = torch.tensor(natural2_chol_vec)

        dummy_responsibilities = torch.ones(
            self.prior.num_centroids,
        )

        auxiliary_mode = self.fit_params['auxiliary_mode']
        if auxiliary_mode == 'constrained_prior':
            update_params = False
        elif auxiliary_mode == 'flexible' or auxiliary_mode == 'semi_flexible':
            update_params = True
        else:
            raise NotImplementedError()

        self.recognition_auxiliary = recognition.Mixture(
            dummy_responsibilities,
            natural1,
            natural2_chol_vec,
            update_params=update_params
        ).to(self.device.index).to(self.dtype)




def pairwise_distance(samples):

    # Number of samples
    num_samples = samples.shape[0]

    # normalize samples on the sphere
    normalized_samples = samples / torch.sqrt((samples ** 2).sum(dim=-1, keepdim=True))

    # Compute pairwise distance
    pairwise_distances = ((normalized_samples.unsqueeze(0) - normalized_samples.unsqueeze(1)) ** 2).sum(-1)

    # Fill the diagonal
    pairwise_distances_mean_tmp = pairwise_distances.sum() / (num_samples * (num_samples - 1))
    diag_idx = range(num_samples)
    pairwise_distances[diag_idx, diag_idx] = pairwise_distances_mean_tmp

    # Maximize minimal distance
    loss = - pairwise_distances.min()

    return loss, pairwise_distances, normalized_samples


def _init_centroids(
        num_centroids,
        dim_centroids,
        ite_max=10000,
        optimizer=lambda x: torch.optim.Adam(x, lr=1e-2),
):
    """Small helper to initialize mixture centroids ~ uniformly on a hypersphere"""

    if num_centroids == 1:
        samples = torch.zeros(1, dim_centroids)
        loss_tot = 0

    elif dim_centroids == 1:
        samples = torch.linspace(-1, 1, num_centroids).unsqueeze(-1)
        loss_tot = 0

    elif num_centroids > 1 and dim_centroids > 1:

        # Init Centroids
        samples_cur = torch.randn(num_centroids, dim_centroids, requires_grad=True)

        # Optimizer
        optim = optimizer([samples_cur])

        loss_tot = []
        for ite in range(ite_max):
            optim.zero_grad()
            loss, _, _ = pairwise_distance(samples_cur)
            loss.backward()
            optim.step()
            loss_tot.append(loss.item())

        # Normalize Optimal samples
        samples = samples_cur.clone().detach()
        with torch.no_grad():
            _, pairwise_distances, samples = pairwise_distance(samples)

    else:
        raise NotImplementedError()

    return samples, loss_tot


