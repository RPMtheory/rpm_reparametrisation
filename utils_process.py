#%% Imports
import torch
import numpy as np
from torch import matmul
from typing import Union, List
import matplotlib.pyplot as plt
from flexible_multivariate_normal import natural2_to_cholesky_vector
import recognition

from utils import diagonalize
from rpfa import RPM


from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def plot_loss(
        rpm: RPM,
        start: int = 0,
        stop: int = -1,
        **kwargs
):

    plt.figure()
    plt.plot(rpm.loss_tot[start:stop], c='k', lw=2, **kwargs)
    plt.ylabel('loss')
    plt.xlabel('Iterations')
    plt.title('- Free Energy')
    plt.tight_layout()

def plot_helper_mixture(
        mixture: recognition.Mixture,
        nstd=2,
        npoints=200,
):
    # Copy the prior and move it to the cpu
    distribution = recognition.Mixture(
        log_responsibilities=mixture.log_responsibilities.to("cpu").clone(),
        centroids_natural1=mixture.natural1.to("cpu").clone(),
        centroids_natural2_chol_vec=mixture.natural2_chol_vec.to("cpu").clone(),
        build_components=True,
        update_params=False,
    )

    dim_latent = distribution.natural1.shape[-1]

    # Mean and variance of each components
    mean, vari = distribution.mixture_components.mean_covariance()

    # Extract giagonal
    vari_diag = vari.diagonal(dim1=-1, dim2=-2)

    # Get Bounds for pdf estimation
    pp = torch.cat([
        mean + nstd * torch.sqrt(vari_diag),
        mean - nstd * torch.sqrt(vari_diag)
    ], dim=0)

    pmins = pp.min(dim=0)[0]
    pmaxs = pp.max(dim=0)[0]

    results = []

    for uu in range(0, dim_latent - 1):
        for vv in range(uu + 1, dim_latent):
            # Get Marginal Parameters
            marginal_mean = mean[..., [uu, vv]]
            marginal_vari = vari[..., [uu, vv], :][..., [uu, vv]]
            marginal_pmins = pmins[..., [uu, vv]]
            marginal_pmaxs = pmaxs[..., [uu, vv]]

            # Get Marginal Distributions
            marginal_components = recognition.FlexibleMultivariateNormal(
                marginal_mean,
                marginal_vari,
                init_natural=False,
                init_cholesky=False,
            )

            # Copy the prior and move it to the cpu
            marginal_distribution = recognition.Mixture(
                log_responsibilities=distribution.log_responsibilities,
                centroids_natural1=marginal_components.natural1,
                centroids_natural2_chol_vec=natural2_to_cholesky_vector(marginal_components.natural2),
                build_components=True,
                update_params=False,
            )

            # Get 2D Grid and estimate pdf
            grid = [
                torch.linspace(marginal_pmins[ii], marginal_pmaxs[ii], npoints)
                for ii in range(2)
            ]
            ZZ = torch.meshgrid(grid)
            Zmeshgrid = torch.cat([zz.reshape(-1, 1) for zz in ZZ], dim=-1)
            prob = torch.exp(marginal_distribution.log_prob(Zmeshgrid).reshape(npoints, npoints)).detach().numpy()

            results.append(
                {
                    'dim0': uu,
                    'dim1': vv,
                    'grid': grid,
                    'prob': prob,
                }
            )

    return results


def regress(
    latent_fit: torch.Tensor,
    latent_var: torch.Tensor,
    latent_true: torch.Tensor,
    regression: str = 'krr',
    regression_param: dict = None,
):
    """ Use linear or kernel regression to regress the fit latent to the true latent when provided """

    shape_fit = latent_fit.shape
    shape_true = latent_true.shape

    unfolded_fit = [np.prod(shape_fit[:-1]).astype(int), shape_fit[-1]]
    unfolded_true = [np.prod(shape_true[:-1]).astype(int), shape_fit[-1]]

    # Unfold and 0 center Latents
    latent_true = latent_true.reshape(unfolded_true)
    latent_true = latent_true - latent_true.mean(dim=0)
    latent_fit = latent_fit.reshape(unfolded_fit)
    latent_fit = latent_fit - latent_fit.mean(dim=0)

    # Regress Latent - True Latent
    if regression == 'linear':
        latent_fit, latent_jac, regressor, jacobian = \
            regress_linear(latent_fit, latent_true, regress_param=regression_param)
    elif regression == 'krr':
        latent_fit, latent_jac, regressor, jacobian = \
            regress_krr(latent_fit, latent_true, regress_param=regression_param)
    else:
        raise NotImplementedError()

    # True new variance or linear approximation with Jacobian
    latent_var = latent_var.reshape(*unfolded_fit, latent_var.shape[-1])
    latent_var = matmul(matmul(latent_jac.transpose(-1, -2), latent_var), latent_jac)
    latent_var = latent_var.reshape(*shape_true, latent_fit.shape[-1])

    # Reshape True and Regressed latent
    latent_fit = latent_fit.reshape(shape_true)
    latent_true = latent_true.reshape(shape_true)

    return latent_fit, latent_var, latent_true, regressor, jacobian


def sample_XYtrain(X, Y, train_pct):
    len_input = X.shape[0]
    len_train = int(len_input * train_pct)
    idx_train = np.random.choice(len_input, len_train, replace=False)
    Xtrain = X[idx_train, :]
    Ytrain = Y[idx_train, :]

    return Xtrain, Ytrain


def regress_linear(X, Y, regress_param=None):

    if regress_param is None:
        regress_param = {}
        
    if not ('train_pct' in regress_param.keys()):
        train_pct = 0.8
    else:
        train_pct = regress_param['train_pct']

    if not ('alpha' in regress_param.keys()):
        alpha = 1e-6
    else:
        alpha = regress_param['alpha']

    Xtrain, Ytrain = sample_XYtrain(X, Y, train_pct)
    XXinv = torch.linalg.inv(alpha * torch.eye(Xtrain.shape[-1], device=Xtrain.device, dtype=Xtrain.dtype) + torch.matmul(Xtrain.transpose(-1, -2), Xtrain))
    beta_hat = matmul(XXinv, matmul(Xtrain.transpose(-1, -2), Ytrain))

    def regressor(X0):
        return matmul(X0, beta_hat)

    def jacobian(X0):
        return beta_hat.unsqueeze(0)

    Yhat = regressor(X)
    Jhat = jacobian(X)

    return Yhat, Jhat, regressor, jacobian


def regress_krr(X, Y, regress_param=None):

    # Default params
    if regress_param is None:
        regress_param = {}

    if 'train_pct' not in regress_param.keys():
        train_pct = 0.8
    else:
        train_pct = regress_param['train_pct']

    if 'alpha' not in regress_param.keys():
        alpha = 1e-3
    else:
        alpha = regress_param['alpha']

    if 'kernel_param' not in regress_param.keys():
        o1 = torch.ones(1, device=X.device, dtype=X.dtype)
        kernel_param = {'type': 'RBF', 'param': {'scale': o1, 'lengthscale': 2 * o1}}
    else:
        kernel_param = regress_param['kernel_param']

    # Init kernel
    if kernel_param['type'] == 'RBF':
        kernel = kernels.RBFKernel(**kernel_param['param'])
    if kernel_param['type'] == 'RQ':
        kernel = kernels.RQKernel(**kernel_param['param'])
    if kernel_param['type'] == 'POLY':
        kernel = kernels.POLYKernel(**kernel_param['param'])



    Xtrain, Ytrain = sample_XYtrain(X, Y, train_pct)
    KXtrainXtrain = kernel.forward(Xtrain, Xtrain).squeeze(0)
    INN = torch.eye(KXtrainXtrain.shape[0],device=KXtrainXtrain.device, dtype=KXtrainXtrain.dtype)
    beta_hat = matmul(torch.linalg.inv(KXtrainXtrain + alpha * INN), Ytrain)

    # Linear approximation to the new mean
    def regressor(X0):
        KxXtrain = kernel.forward(X0, Xtrain).squeeze(0)
        return matmul(KxXtrain, beta_hat)

    Yhat = regressor(X)

    if kernel_param['type'] == 'RBF':
        # Linear approximation to the variance
        def jacobian(X0):
            KxXtrain = kernel.forward(X0, Xtrain).squeeze(0).unsqueeze(-1)
            betaK = beta_hat.unsqueeze(0) * KxXtrain
            dX = Xtrain.unsqueeze(0) - X0.unsqueeze(1)
            return (2 * matmul(betaK.transpose(-1, -2), dX) / (kernel.lengthscale**2)).transpose(-1, -2)


        Jhat = jacobian(X)

    else:
        Jhat = None
        jacobian = None

    return Yhat, Jhat, regressor, jacobian




def plot_gradient_line(xx, cc, **kwargs):
    for tt in range(xx.shape[0]-1):
        plt.plot(xx[tt:tt+2, 0], xx[tt:tt+2,  1], c=cc[tt], **kwargs)


def max_diagonal_factors(factors):
    """
        Custom Varimax Criterion: Maximize the diagonal elements of the factors
    """

    # Factors dimension
    D = factors.shape[-1]

    # Fit parameters parameters
    ite_max = 1000
    reltol = 1e-6

    # Initialize rotation
    rotation = torch.eye(D, dtype=factors.dtype)

    # Iterate
    for ite_cur in range(ite_max):
        max_theta = torch.tensor(0)

        # Loop over all pairs of dimensions
        for ii in np.arange(0, D-1):
            for jj in range(ii+1, D):

                Vuu = factors[:, ii, ii]
                Vvv = factors[:, jj, jj]
                Vuv = factors[:, ii, jj]

                numer = torch.sum(4 * Vuv * (Vuu - Vvv), dim=0)
                denum = torch.sum(4 * Vuv ** 2 - (Vuu - Vvv) ** 2, dim=0)
                theta = torch.atan2(numer, denum) / 4 + 1 * torch.pi / 4

                max_theta = torch.max(max_theta, torch.abs(theta))

                rotation_cur = torch.eye(D, dtype=factors.dtype)
                rotation_cur[ii, ii] = torch.cos(theta)
                rotation_cur[ii, jj] = -torch.sin(theta)
                rotation_cur[jj, ii] = torch.sin(theta)
                rotation_cur[jj, jj] = torch.cos(theta)

                rotation = matmul(rotation_cur, rotation)
                factors = matmul(matmul(rotation_cur, factors), rotation_cur.transpose(-1, -2))

        if max_theta < reltol:
            print('theta converged ite=' + str(ite_cur))
            break

    return factors, rotation


def get_precision_gain(model):
    """Precision Gain from prior to Recognition Factors"""

    # Get Factor Posterior Means
    factors_mean = model.factors.suff_stat_mean[0]

    # Mean Center Factors across Time
    factors_mean = factors_mean - factors_mean.mean(dim=-2, keepdim=True)

    # Normalize factors to have max absolute value of 1
    factors_normalizer = diagonalize(factors_mean.abs().max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0])
    factors_mean = matmul(torch.linalg.inv(factors_normalizer), factors_mean.unsqueeze(-1)).squeeze(-1)

    # Prior Precision
    natural2_prior = diagonalize(model._get_prior_marginals()[1].permute((1, 0)))[0].unsqueeze(0)

    # Factors Precision (if not time dependent, then only one value)
    factors_precision = model.factors.natural2[:, 0, 0].unsqueeze(1).unsqueeze(1)

    # Precision gain (normalise)
    precision_gain = - factors_precision + natural2_prior
    precision_gain = matmul(matmul(factors_normalizer, precision_gain), factors_normalizer)
    precision_gain = precision_gain.squeeze(1).squeeze(1)

    return precision_gain, factors_mean


def trim_dimensions(precision, threshold=0.95, verbose=True):
    """ Remove Latent dimensions with no or few precision gain across all factors """

    # Look at the contribution of each latent dimension to the precision gain
    contrib = precision.diagonal(dim1=-1, dim2=-2)
    contrib = (contrib / contrib.sum(dim=-1, keepdim=True)).mean(dim=0)
    contrib, contrib_indices = contrib.sort(descending=True)
    contrib = contrib.cumsum(dim=-1)

    # Remove dimensions above threshold
    above_thr = contrib >= threshold
    thr_index = torch.argmax((above_thr != 0).to(dtype=torch.int), dim=-1)
    kept_dimensions = contrib_indices[:thr_index + 1].sort()[0]

    if above_thr.sum() > 1 and verbose:
        print('Removed ' + str(above_thr.sum().item() - 1) + ' dimensions')

    # Trim precision gain
    precision_gain_trimmed = precision[..., kept_dimensions][..., kept_dimensions, :]

    return precision_gain_trimmed, kept_dimensions, contrib


def rotate_and_trim_precision(precision, rotate=True, normalize=False, threshold=0.95, verbose=True):
    """ Trim and rotate precision gain to maximize diagonal criterion """

    if rotate:
        # Rotate precision gain using diagonal criterion
        precision_rotated, rotation = max_diagonal_factors(precision)
        rotation = rotation.transpose(-1, -2)
    else:
        precision_rotated = precision
        rotation = torch.eye(precision.shape[-1], dtype=precision.dtype)

    # Trim dimensions
    precision_rotated_trimmed, kept_latent, contrib = \
        trim_dimensions(precision_rotated, threshold=threshold, verbose=verbose)

    # Incorporate Trimming in rotation
    Id = torch.eye(precision_rotated.shape[-1], dtype=precision_rotated.dtype)
    Id = Id[:, kept_latent]
    # rotation = matmul(rotation, Id)

    # Normalize over factors
    if normalize:
        precision_rotated_trimmed = precision_rotated_trimmed / precision_rotated_trimmed.sum(dim=0, keepdim=True)

    return precision_rotated_trimmed, rotation, kept_latent, contrib


def confidence_ellipse(loc, cov, ax, n_std=3.0, facecolor='none', **kwargs):

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(loc[0], loc[1])

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)