# Imports
import torch
from torch import matmul

import recognition
from flexible_multivariate_normal import (
    FlexibleMultivariateNormal,
    cholesky_vector_to_natural2,
    natural2_to_cholesky_vector,
    log_normalizer_sum_fast
)


class Mixin:
    """Mixin class containing necessary methods for forwarding fast RPM model"""

    def _get_gamma(self):
        """
        Build Gamma Parameters
        (Note: we distinguish N and N' for clarity despite N = N')
        """

        # Phi(eta_jn') ~ 1 x J x N' x 1
        log_normalizer_factors = self.log_normalizer_factors.unsqueeze(-1)

        # Psi_jn' ~ 1 x J x N' x 1
        psi = self.psi.unsqueeze(-1).unsqueeze(0)

        # Factors Natural Parameters ~ 1 x J x N' x 1 x K (x K)
        naturals_factors = [
            nati.unsqueeze(2).unsqueeze(0)
            for nati in self.forwarded_factors
        ]

        # Auxiliary Natural Parameters ~ M x J x 1 x N x K (x K)
        naturals_auxiliary = [
            nati.unsqueeze(2)
            for nati in self.forwarded_auxiliary
        ]

        # Log Normalizer Phi(eta_jn' + eta_tilde_mjn) ~ M x J x N' x N
        log_normalizer_sum = log_normalizer_sum_fast(
            naturals_auxiliary[0],
            naturals_auxiliary[1],
            naturals_factors[0],
            naturals_factors[1],
        )

        # Products ~ M x J x N' x N
        s_mjnn = log_normalizer_sum - psi - log_normalizer_factors

        # Log Gamma terms ~ M x J x N (sum is over N')
        self.log_gamma = s_mjnn.diagonal(dim1=-1, dim2=-2) - torch.logsumexp(s_mjnn, dim=2)

        # Store log normalizer Phi(eta_jn + eta_tilde_mjn) for latter use ~ M x J x N
        self.log_normalizer_factors_hat = log_normalizer_sum.diagonal(dim1=-1, dim2=-2)


    def _get_psi(self):

        # Prior Params ~ M x 1 x 1 (x K (x K))
        responsibilities_prior, natural1_prior, natural2_prior = [
            nat.unsqueeze(1).unsqueeze(1)
            for nat in self.forwarded_prior
        ]

        # Recognition Params ~ 1 x J x N x K (x  K)
        natural1_factors, natural2_factors = [
            nat.unsqueeze(0)
            for nat in self.forwarded_factors
        ]

        # Log Normalizer Phi(eta_0m) ~ M x 1 x 1
        log_normalizer_prior = FlexibleMultivariateNormal(
            natural1_prior,
            natural2_prior,
            init_natural=True,
            init_cholesky=False,
        ).log_normalizer

        # Log Normalizer Phi(eta_jn) ~ 1 x J x N
        log_normalizer_factors = FlexibleMultivariateNormal(
            natural1_factors,
            natural2_factors,
            init_natural=True,
            init_cholesky=False,
        ).log_normalizer

        # Log Normalizer Phi(eta_0m + eta_jn) ~ M x J x N
        log_normalizer_sum = log_normalizer_sum_fast(
            natural1_prior,
            natural2_prior,
            natural1_factors,
            natural2_factors
        )

        # Store Factors log normalizers for latter use
        self.log_normalizer_factors = log_normalizer_factors

        # Store Prior log normalizers for latter use
        self.log_normalizer_prior = log_normalizer_prior

        # Psi_jn ~ J x N
        self.psi = torch.logsumexp(
            torch.log(responsibilities_prior) + log_normalizer_sum - log_normalizer_prior - log_normalizer_factors,
            dim=0
        )

    def _get_offset(self):
        """
        Helper for auxiliary approximation
        Currently Implemented:
            - Constrained where F is approximated by prior
            - Constrained where F is moment matched
        """

        if self.fit_params['auxiliary_mode'] == 'constrained_moment_matched':
            # We offset by moment-matching the mixture F / prior

            # Recognition factors ~ J x N x K (x K)
            natural1_factors, natural2_factors = self.forwarded_factors

            # Normalizer ~ J x N x 1
            #psi = torch.nn.ELU()(self.psi.unsqueeze(-1) + 30) - 30
            epsi = torch.exp(-self.psi.unsqueeze(-1))


            # Weighted Mean of the 1st Natural Parameters ~ J x K
            natural1_weighted_mean = (natural1_factors * epsi).mean(dim=1)

            # Weighted Variances of the 1st Natural Parameters ~ J x K x K
            natural1_weighted_vari = (
                    matmul(
                        natural1_factors.unsqueeze(-1),
                        natural1_factors.unsqueeze(-2)
                    ) * epsi.unsqueeze(-1)
            ).mean(dim=1)
            natural1_weighted_vari -= matmul(
                natural1_weighted_mean.unsqueeze(-1),
                natural1_weighted_mean.unsqueeze(-2)
            )

            # Weighted Identity Matrix ~ J x K x K
            Id = torch.eye(self.dim_latent, dtype=self.dtype, device=self.device).unsqueeze(0)
            Id_weighted = (epsi.mean(dim=1).unsqueeze(-1)) * Id

            # From J x 1 x K x K to J x K x K and invert
            natural2_factors = natural2_factors.squeeze(1)
            natural2_factors_inv = torch.linalg.inv(natural2_factors)

            # TODO: check this
            IpVeta_inv = torch.linalg.inv(Id_weighted + matmul(natural1_weighted_vari, -0.5 * natural2_factors_inv))

            # Moment Matched 1st Natural Parameter ~ J x K
            natural1_offset = matmul(IpVeta_inv, natural1_weighted_mean.unsqueeze(-1)).squeeze(-1)

            # Moment Matched 2nd Natural Parameter ~ J x K x K
            natural2_offset = matmul(IpVeta_inv, natural2_factors)

        elif self.fit_params['auxiliary_mode'] == 'constrained_prior':
            # We do not offset ~ J  x K (x K)
            natural1_offset = torch.zeros(self.num_factors, self.dim_latent, device=self.device, dtype=self.dtype)
            natural2_offset = torch.zeros(
                self.num_factors, self.dim_latent, self.dim_latent, device=self.device, dtype=self.dtype
            )

        else:
            raise NotImplementedError()

        self.forwarded_offset = [natural1_offset, natural2_offset]

    def _forward_factors(self, observations):
        """ Forward Recognition Factors"""

        # Recognition Functions
        recognition_factors = self.recognition_factors

        # Forward all
        forwarded_params = [
            facti(obsi)
            for facti, obsi in zip(recognition_factors, observations)
        ]

        # 1st natural parameter ~ J x N x K
        natural1 = torch.cat(
            [reci[0].unsqueeze(0) for reci in forwarded_params],
        )

        # 2nd natural parameter ~ J x 1 x K x K
        natural2 = cholesky_vector_to_natural2(
            torch.cat(
                [reci[1].unsqueeze(0) for reci in forwarded_params],
            )
        ).unsqueeze(1)

        self.forwarded_factors = [natural1, natural2]

    def _forward_prior(self):
        self.forwarded_prior = self.prior.forward()

    def _forward_variational(self):
        """Closed Form Updates for Variational Mixture"""

        # Annealing Schemes
        a = self.annealing_a
        b = self.annealing_b

        # Natural Priors ~ M x 1 x K (x K)
        naturals_prior = [
            nati.unsqueeze(1) for nati in
            self.forwarded_prior[1:]
        ]

        # Sum of Natural Factors ~ 1 x N x K ( x K)
        naturals_factors_sum = [
            nati.sum(dim=0).unsqueeze(0)
            for nati in self.forwarded_factors
        ]

        # Sum of Natural Offsets ~ M x N x K (x K)
        naturals_auxiliary_offset_sum = [
            nati.sum(dim=1)
            for nati in self.forwarded_auxiliary_offset
        ]

        # Build Scale Parameter to update variational
        auxiliary_mode = self.fit_params['auxiliary_mode']
        if auxiliary_mode == 'constrained_prior' or auxiliary_mode == 'semi_flexible':
            scale = b
        elif auxiliary_mode == 'flexible':
            scale = b + a * self.num_factors
        else:
            raise NotImplementedError()

        # natural parameters of the variational mixture ~ M x N x (K (x K))
        naturals_variational = [[]] + [
            ((b - 1 + a) / scale) * nat_prio + (a / scale) * (nat_fac + nat_off)
            for nat_prio, nat_fac, nat_off in zip(naturals_prior, naturals_factors_sum, naturals_auxiliary_offset_sum)
        ]

        # Temporary Allocation (Mixture Weights not estimated yet)
        self.forwarded_variational = naturals_variational

        # The auxiliary factor update is here for convenience
        self._forward_auxiliary()

        # Obtain the Gamma Parameter
        self._get_gamma()

        # Log Gamma parameter ~ M x J x N -> M x N
        log_gamma = self.log_gamma.sum(dim=1)

        # Phi(eta_0m) ~ M x 1 x 1 -> M x 1
        log_normalizer_prior = self.log_normalizer_prior.squeeze(1)

        # Phi(eta_jn + eta_tilde_mjn) ~ M x J x N -> M x 1 x N
        log_normalizer_factors_hat = self.log_normalizer_factors_hat.sum(dim=1)

        # Phi(eta_q_mn) (* J+1) ~ M x N
        log_normalizer_variational = FlexibleMultivariateNormal(
            naturals_variational[1],
            naturals_variational[2],
            init_natural=True,
            init_cholesky=False,
        ).log_normalizer

        # -KL[q_mn || prior_m] - \sum KL[q_mn || f_hat_mjn] ~ M x N
        KLterms = ((a * self.num_factors + b) * log_normalizer_variational
                   - (b - 1 + a) * log_normalizer_prior - a * log_normalizer_factors_hat)

        # Prior respoonsibilities ~ M x 1
        log_responsibilities_prior = torch.log(self.forwarded_prior[0].unsqueeze(-1))

        # Variational Responsibilities ~ M x N
        non_normalized_log_responsibilities = ((b - 1 + a) * log_responsibilities_prior + KLterms + a * log_gamma) / b
        responsibilities_normalizer = torch.logsumexp(non_normalized_log_responsibilities, dim=0, keepdim=True)

        # Store responsibilities ~ M x N
        self.forwarded_variational[0] = torch.exp(non_normalized_log_responsibilities - responsibilities_normalizer)

        # TODO: Maybe remove second term !
        prior_mixture = torch.log(self.prior.responsibilities() + 1e-6).sum() * (self.alpha - 1) + self.beta_constant

        # Store Normalizer for loss calculation ~ 1 x N
        self.responsibilities_normalizer = b * (responsibilities_normalizer + prior_mixture)

    def _forward_auxiliary(self):
        """Update Auxiliary factors"""

        # Variational Natural Parameters M x 1 x N x K x (x K)
        naturals_variational = [
            nati.unsqueeze(1)
            for nati in self.forwarded_variational[1:]
        ]

        # Offset Parameters ~ M x J x N x K (x K)
        naturals_offset = [
            nati
            for nati in self.forwarded_auxiliary_offset
        ]

        # Auxiliary Parameters ~ M x J x N x K (x K)
        auxiliary_mode = self.fit_params['auxiliary_mode']
        if auxiliary_mode == 'constrained_prior':
            self.forwarded_auxiliary = [
                nat_var
                for nat_var in naturals_variational
            ]

        elif auxiliary_mode == 'flexible':
            self.forwarded_auxiliary = [
                nat_off
                for nat_off in naturals_offset
            ]

        elif auxiliary_mode == 'semi_flexible':
            self.forwarded_auxiliary = [
                nat_var + nat_off
                for nat_var, nat_off in zip(naturals_variational, naturals_offset)
            ]

    def _forward_auxiliary_offset(self, idxs=None):

        auxiliary_mode = self.fit_params['auxiliary_mode']
        if auxiliary_mode == 'constrained_prior':
            natural1 = torch.zeros(1, 1, 1, self.dim_latent, dtype=self.dtype, device=self.device)
            natural2 = torch.zeros(1, 1, 1, self.dim_latent, self.dim_latent, dtype=self.dtype, device=self.device)
            self.forwarded_auxiliary_offset = [natural1, natural2]
        elif auxiliary_mode == 'flexible' or auxiliary_mode == 'semi_flexible':
            self.forwarded_auxiliary_offset = [
                nat[:, :, idxs]
                for nat in self.recognition_auxiliary.forward()[1:]
            ]
        else:
            raise NotImplementedError()



