import torch
import numpy as np

from typing import Union, List, Dict, Tuple

from torch.utils.data import DataLoader

from utils import print_loss

from data import RPMData, merge_rpm_data

from flexible_multivariate_normal import (
    FlexibleMultivariateNormal,
    natural2_to_cholesky_vector
)

import recognition
import _initializations
import _forwards


class RPM(_initializations.Mixin, _forwards.Mixin):
    """
    Recognition Parametrised Factor Analysis (RPFA) with Recognition Parametrised Model (RPM)

    Args:
        - observations: Multimodal observations.
            sizes:
                len(observations) = num_factors
                observations[j] ~ num_observations x *dim_j
            where:
                num_factors: number of conditionally independent factors
                num_observations: number of observation samples
                dim_j: dimension of j=th observations
        - loss_tot: Stored Loss defined as - Free Energy
        - fit_params: Fit Parameters. See _initializations.py for Details
        - Prior: Prior Distribution Mixture
        - recognition_factors: Recognition Factors Module

    notation: for compactness, we sometimes denote:
        N: num_observations
        J: num_factors
        K: dim_latent
        M: num_centroids (in the prior)

    """

    def __init__(
            self,
            observations: RPMData,
            loss_tot: List = None,
            fit_params: Dict = None,
            prior: recognition.Mixture = None,
            recognition_factors: List[recognition.Encoder] = None,
    ):

        # Problem dimensions
        self.num_factors = observations.num_factors
        self.num_observation = len(observations)

        # Device and data type
        self.dtype = observations.dtype
        self.device = observations.device
        print('RPM on ' + str(self.device))

        # Fit / Config parameters
        self.fit_params = fit_params

        # Init Loss
        self.loss_tot = [] if loss_tot is None else loss_tot
        self.epoch = 0
        self.batch = 0

        # Initialize Distributions Parametrization
        self.prior = prior
        self.recognition_factors = recognition_factors

        self._init_all(observations)

        # Init Forwarded
        self.forwarded_prior = None
        self.forwarded_factors = None
        self.forwarded_auxiliary = None
        self.forwarded_variational = None

        # Init Annealing Schemes
        self.annealing_a = 1.0
        self.annealing_b = 1.0

        with torch.no_grad():

            # Get a batch
            batched_observations, _, idxs = next(iter(self.data_loader))

            # Forward
            self._forward_all(batched_observations, idxs)

            # Initial Loss
            self.loss_tot.append(self._get_loss().item())

    def _init_all(self, observations: RPMData):
        """ Init all parameters (see _initializations.Mixin) """

        # Fit parameters
        self._init_fit_params()
        self.dim_latent = self.fit_params['dim_latent']
        self.data_loader = DataLoader(observations, batch_size=self.fit_params['batch_size'])
        self._init_prior()
        self._init_auxiliary()
        self._init_factors(observations)

    def _forward_all(self, batched_observations, idxs=None): #: Union[List[torch.Tensor], tuple[torch.Tensor]]):
        """ Forward all recognition (see _forwards.Mixin) """

        self._forward_prior()
        self._forward_factors(batched_observations)

        self._get_psi()
        self._forward_auxiliary_offset(idxs)
        self._forward_variational()

    def _get_loss(self, average=True):
        """
        The loss is defined as the negative Free Energy
        Note:
            Once q has been updated, the Free Energy corresponds to the log normaliser
            of the variational mixture proportions
        """

        free_energy = self.responsibilities_normalizer.mean() if average else self.responsibilities_normalizer

        return - free_energy

    def fit(self):
        """Fit the model to  observations"""

        # Fit params
        fit_params = self.fit_params
        num_epoch = fit_params['num_epoch']

        # Recognition Factors Parameters
        factors_param = []
        for cur_factor in self.recognition_factors:
            factors_param += cur_factor.parameters()

        # Prior Parameters
        prior_params = self.prior.parameters()

        # Auxiliary Parameters
        auxiliary_params = self.recognition_auxiliary.parameters()

        # Gather All Parameters
        all_params = [
            [factors_param, fit_params['factors_params']],
            [prior_params, fit_params['prior_params']],
            [auxiliary_params, fit_params['auxiliary_params']]
        ]

        # Gather All optimizers
        all_optimizers = [
            opt['optimizer'](param) for param, opt in all_params
        ]

        # Gather All schedulers
        all_scheduler = [
            params[1]['scheduler'](opt) for params, opt in zip(all_params, all_optimizers)
        ]

        # Fit
        for epoch in range(num_epoch):

            # Store epoch
            self.epoch = epoch
            # batches = self.batches[epoch]

            # Annealing Scheme
            self.annealing_a = self.fit_params['annealing_a'](epoch)
            self.annealing_b = self.fit_params['annealing_b'](epoch)

            # Current epoch losses
            loss_batch = []

            for batch_id, [batched_observations, _, idxs] in enumerate(self.data_loader):

                # Store batch
                self.batch = batch_id

                # Forward observation
                self._forward_all(batched_observations, idxs)

                # Loss    
                loss = self._get_loss()
                loss_batch.append(loss.item())

                # Reset Optimizers
                for opt in all_optimizers:
                    opt.zero_grad()

                # Gradients
                loss.backward(retain_graph=True)

                # Gradient Steps
                for opt in all_optimizers:
                    opt.step()

                    # Scheduler Steps
            for sched in all_scheduler:
                sched.step()

            # Gather loss
            self.loss_tot.append(np.mean(loss_batch))

            # Logger
            print_loss(
                self.loss_tot[-1],
                epoch + 1,
                num_epoch,
                pct=self.fit_params['pct']
            )

    def get_posteriors(self, batched_observations): # Union[List[torch.Tensor], tuple[torch.Tensor]]):
        """ Approximate Posterior (Variational) and Recognition Posterior Distributions"""

        with (torch.no_grad()):

            # Eval Mode
            for reci in self.recognition_factors:
                reci.eval()

            # Model Distribution
            self._forward_all(batched_observations)

            # Train Mode
            for reci in self.recognition_factors:
                reci.train()

            # Prior params ~ M x 1 x 1 x (K (x K))
            naturals_prior = [
                nati.unsqueeze(1).unsqueeze(1)
                for nati in self.forwarded_prior
            ]

            # Factors "Likelihood" params ~ 1 x J x N x K (x K)
            naturals_factor = [
                nati.unsqueeze(0)
                for nati in self.forwarded_factors
            ]

            # Factors "Posteriors" params ~ M x J x N x (K x (x K))
            factors_posterior_responsibilities = naturals_prior[0]
            factors_posterior_natural1 = naturals_prior[1] + naturals_factor[0]
            factors_posterior_natural2 = naturals_prior[2] + naturals_factor[1]
            factors_posterior_chol_vec = natural2_to_cholesky_vector(factors_posterior_natural2)

            # Variational params ~ M x N x (K x (x K))
            naturals_variational = self.forwarded_variational

            # Prior Mixture ~ M (x K (x K))
            prior = self.prior
            prior._check_built_components()

            # Factors Posterior Mixture ~ M x J x N (x K (x K))
            factors_posterior = recognition.Mixture(
                log_responsibilities=torch.log(factors_posterior_responsibilities),
                centroids_natural1=factors_posterior_natural1,
                centroids_natural2_chol_vec=factors_posterior_chol_vec,
                update_params=False,
                build_components=True,
            )

            # Variational Mixture ~ M x N (x K (x K))
            variational = recognition.Mixture(
                log_responsibilities=torch.log(naturals_variational[0]),
                centroids_natural1=naturals_variational[1],
                centroids_natural2_chol_vec=natural2_to_cholesky_vector(naturals_variational[2]),
                update_params=False,
                build_components=True,
            )

            # Factors Recognition "Likelihoods" ~ M x N x K (x K)
            factors_likelihood = FlexibleMultivariateNormal(
                naturals_factor[0].squeeze(0),
                naturals_factor[1].squeeze(0),
                init_natural=True,
                init_cholesky=False,
                store_suff_stat_mean=True,
            )

            return {
                'prior': prior,
                'factors_likelihood': factors_likelihood,
                'factors_posterior': factors_posterior,
                'variational': variational,
            }

    def get_multi_loss(self, data_list: List[RPMData]):
        """
        Once recognition and prior have been trained, build a new RPM model with all observation in data_list
        and evaluate it
        """

        merged_observations, indices = merge_rpm_data(data_list)

        with torch.no_grad():

            # Eval Mode
            for reci in self.recognition_factors:
                reci.eval()

            # Forward Model
            self._forward_all(merged_observations.observations)

            # Train Mode
            for reci in self.recognition_factors:
                reci.train()

            # Get All Losses
            losses = self._get_loss(average=False).squeeze(0)

        # Dataset-wize loss
        losses = [losses[idx].mean().item() for idx in indices]

        return losses
