# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul
from utils import diagonalize
from flexible_multivariate_normal import vector_to_tril, FlexibleMultivariateNormal, vector_to_tril_diag_idx

from typing import List, Tuple, Union, Callable

__all__ = [
    'Net',
    'Mixture'
]

"""
    Recognition Function Architectures and Mixture for RPM.
    
    goal: map observation(s) (dimension: feature_size) to latent space (dimension: dim_latent) parameters. 
    
    input: observation(s)
        torch.Tensor or List[torch.Tensor] of size [*batch_size, *feature_size] 

    output: [param1, param2]
        concatenated torch.Tensor of size [*batch_size, dim_latent + dim_latent * (dim_latent + 1) / 2]
        param1 and param2 are then used to parameterize flexible multivariate normal distribution
        param1 [*batch_size, dim_latent] is either the mean or 1st natural parameter
        param2 [*batch_size, dim_latent * (dim_latent + 1) / 2] is either the vectorized Cholesky factor 
            of the variance or -2nd natural parameter   
             
"""


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()


class Net(Encoder):

    def __init__(
            self,
            dim_input: int,
            dim_latent: int,
            precision_chol_vec: torch.Tensor,
            channels: Union[Tuple, List] = (),
            dim_hidden: Union[Tuple, List] = (),
            kernel_conv: Union[Tuple, List] = (),
            kernel_pool: Union[Tuple, List] = (),
            non_linearity: Callable = F.relu,
            dropout: float = 0.0,
    ):

        """
        Neural Network that outputs natural parameters of Flexible Multivariate Normal distributions.

        :param dim_input: List of dimensions of each input
        :param dim_latent: Dimension of the latent space
        :param precision_chol_vec: An initital value for the Vectorized
            Cholesku decomposition of the negative precision matrix
        :param channels: Number of channels for convolutional layers
        :param dim_hidden: Dimensions of each hidden fully connected layer
        :param non_linearity: Non linearity function
        :param dropout: Dropout proportion
        """

        # Inherit
        super(Net, self).__init__()

        # Convolutional layers
        self.kernel_conv = kernel_conv
        self.kernel_pool = kernel_pool
        self.channels = channels

        # Fully connected layers
        self.dim_hidden = dim_hidden
        self.non_linearity = non_linearity

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)

        # Input/Output dimensions
        self.dim_input = dim_input
        self.mlp_input = dim_input
        self.dim_latent = dim_latent
        self.dim_output = dim_latent

        # Init layers
        self.layers = nn.ModuleList()
        self.mlp_input = append_cnn(self.layers, self.dim_input, self.kernel_conv, self.kernel_pool, self.channels)
        append_mlp(self.layers, self.mlp_input, self.dim_hidden, self.dim_output)

        # Vectorized Cholesky decomposition of the Precision
        self.precision_chol_vec = torch.nn.Parameter(precision_chol_vec, requires_grad=True)
        #from flexible_multivariate_normal import vector_to_tril_diag_idx
        #self.idx = vector_to_tril_diag_idx(self.dim_latent)
        #self.precision_chol_vec = torch.nn.Parameter(precision_chol_vec[self.idx], requires_grad=True)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        # Convolutional layers
        if len(self.kernel_conv) > 0:

            # Handle multi-batches and assume 1 input channel
            batch_dim = x.shape[:-2]
            x = x.reshape(-1, 1, *self.dim_input)

            for cl in range(len(self.kernel_conv)):
                x = self.non_linearity(F.max_pool2d(self.layers[cl](x), self.kernel_pool[cl]))

            x = x.reshape(*batch_dim, self.mlp_input)

        # Feedforward layers
        for layer in self.layers[len(self.kernel_conv):-1]:
            x = layer(x)
            x = self.dropout(x)
            x = self.non_linearity(x)
        x = self.layers[-1](x)

        #tmp = self.precision_chol_vec TODO remove this !
        #tmp = torch.zeros(int(self.dim_latent * (self.dim_latent + 1) / 2), device=x.device, dtype=x.dtype)
        #tmp[self.idx] = self.precision_chol_vec
        return x, self.precision_chol_vec


def conv_pool_dim(dim_input, kernel_conv, kernel_pool):
    """
    Convolutional and pooling layer output dimension
    :param dim_input: Input dimension
    :param kernel_conv: Kernel size for convolutional layer
    :param kernel_pool: Kernel size for pooling layer
    :return: Output dimension
    """

    out_conv = dim_input - kernel_conv + 1
    out_pool = out_conv // kernel_pool

    return out_pool


def append_cnn(layers, dim_input, kernel_conv, kernel_pool, channels):
    """
    Append convolutional Layers with a pooling layer
    :param layers: List of layers
    :param dim_input: Input dimension
    :param kernel_conv: Kernel sizes for convolutional layers
    :param kernel_pool: Kernel sizes for pooling layers
    :param channels: Number of channels for convolutional layers
    :return: Output dimension
    """

    # Use and append convolutional and pooling layer
    if len(kernel_conv) > 0:

        # Check sizes
        assert (len(kernel_conv) == len(kernel_pool))
        assert (len(kernel_conv) == len(channels) - 1)

        # Init channel number
        channels_ou = channels[0]

        # Init output size
        conv_output_x = dim_input[0]
        conv_output_y = dim_input[1]

        for cl in range(len(kernel_conv)):
            # Current channels
            channels_in = channels_ou
            channels_ou = channels[cl + 1]

            # Current output size
            conv_output_x = conv_pool_dim(conv_output_x, kernel_conv[cl], kernel_pool[cl])
            conv_output_y = conv_pool_dim(conv_output_y, kernel_conv[cl], kernel_pool[cl])

            # Append convolutional layer
            layers.append(nn.Conv2d(channels_in, channels_ou, kernel_size=kernel_conv[cl]))

        # CNN Output: linearized and collapsed across channels
        dim_output = int(channels_ou * conv_output_x * conv_output_y)

    else:
        dim_output = dim_input[0]

    return dim_output


def append_mlp(layers, dim_input, dim_hidden, dim_output=None, zero_init=False):
    """
    Append fully connected layers
    :param layers: List of layers
    :param dim_input: Input dimension
    :param dim_hidden: List of hidden dimensions
    :param dim_output: Output dimension
    :param zero_init: zero out the last layer
    """

    # Use and append fully connected layers
    for i in range(len(dim_hidden) + 1):
        if len(dim_hidden) > 0:
            if i == 0:
                layers.append(nn.Linear(dim_input, dim_hidden[i]))
            elif i == len(dim_hidden) and dim_output is not None:
                layers.append(nn.Linear(dim_hidden[i - 1], dim_output))
            elif i < len(dim_hidden):
                layers.append(nn.Linear(dim_hidden[i - 1], dim_hidden[i]))
        elif dim_output is not None:
            layers.append(nn.Linear(dim_input, dim_output))

        if zero_init:
            torch.nn.init.constant_(layers[-1].weight, 1e-6)
            torch.nn.init.constant_(layers[-1].bias, 1e-6)


class Mixture(Encoder):
    """
    Multivariate Normal Mixture Distribution (Used for prior and variational in RP-FA)
    :param log_responsibilities: Mixing proportions
    :param centroids_natural1: 1st natural parameters of the centroids
    :param centroids_natural2_chol_vec: Vectorized Cholesky decomposition of the precision of of the centroids
    :param update_params: Requires grad
    :param build_components: Build FlexibleMultivariate Distriubtions
    """

    def __init__(
            self,
            log_responsibilities: torch.Tensor,
            centroids_natural1: torch.Tensor,
            centroids_natural2_chol_vec: torch.Tensor,
            update_params: bool = True,
            build_components: bool = False,
            diagonal_covariance: bool = False,
    ):
        super().__init__()

        # Number of Components
        self.num_centroids = log_responsibilities.shape[0]
        assert self.num_centroids == centroids_natural1.shape[0], 'Invalid Shapes'
        assert self.num_centroids == centroids_natural2_chol_vec.shape[0], 'Invalid Shapes'

        # Responsibilities
        self.log_responsibilities = torch.nn.Parameter(log_responsibilities, requires_grad=update_params)

        # Natural Parameters
        self.natural1 = torch.nn.Parameter(centroids_natural1, requires_grad=update_params)
        self.natural2_chol_vec = torch.nn.Parameter(centroids_natural2_chol_vec, requires_grad=update_params)

        # Mixture components dimensions independents ?
        self.diagonal_covariance = diagonal_covariance

        # Build Centroid distributions
        self.mixture_components = None
        if build_components:
            self._check_built_components()

    def natural2_chol(self):
        if self.diagonal_covariance:
            return diagonalize(self.natural2_chol_vec)
        else:
            return vector_to_tril(self.natural2_chol_vec)

    def natural2(self, jitter=1e-12):
        """Build precision"""

        natural2_tril = self.natural2_chol()

        Id = torch.eye(
            natural2_tril.shape[-1],
            device=self.natural2_chol_vec.device,
            dtype=self.natural2_chol_vec.dtype
        )
        return - matmul(natural2_tril, natural2_tril.transpose(-1, -2)) - jitter * Id

    def responsibilities(self):
        return torch.nn.Softmax(dim=0)(self.log_responsibilities)

    def forward(self):
        """Build Responsibilities"""
        return self.responsibilities(), self.natural1, self.natural2()

    def _check_built_components(self):
        """ Build Centroid distributions if necessary"""
        if self.mixture_components is None:
            self.mixture_components = FlexibleMultivariateNormal(
                self.natural1,
                self.natural2_chol(),
                init_natural=True,
                init_cholesky=True,
                store_suff_stat_mean=True,
            )

    def log_prob(self, value: torch.Tensor):
        """Evaluate pdf"""
        self._check_built_components()

        # Mixture Component Log probabilites
        log_prob_comps = self.mixture_components.log_prob(value.unsqueeze(1))

        # log responsibilities
        log_prob_resps = torch.log(self.responsibilities()).unsqueeze(0)

        return torch.logsumexp(log_prob_comps + log_prob_resps, dim=1)

    def get_centroids(self):
        """Return Centroid locations"""
        self._check_built_components()
        return self.mixture_components.mean_covariance()[0]

    def find_mode(
            self,
            ite_max=1000,
            init_evals=None,
            optimizer=lambda x: torch.optim.Adam(x, lr=1e-2),
            criterion=lambda x: (x[-2] - x[-1]) / x[-2] < 1e-6 and (x[-2] - x[-1]) > 0,
    ):
        """" Empirical Search for the modes of the Mixture Distribution """

        # Mixture centroids
        centroids = self.get_centroids()

        if self.num_centroids == 1:
            # 1 Component: Mode is the Mean
            return centroids, None, None

        else:
            # Empirical Search
            # Init Empirical Modes
            if init_evals is None:
                zevals = centroids.clone().detach().requires_grad_(True)
            else:
                zevals = init_evals.clone().detach().requires_grad_(True)

            # Optimizer
            optimizer = optimizer([zevals])

            loss_tot = []
            for ite in range(ite_max):

                # Reset Gradients
                optimizer.zero_grad()

                # loss defined with log_probs
                loss = - self.log_prob(zevals).sum()

                # Backprop
                loss.backward()

                # Step
                optimizer.step()

                # Store loss
                loss_tot.append(loss.item())

                # Convergence Reached
                if ite > 0:
                    if criterion(loss_tot):
                        print('Mode-Search Converged')
                        break
            # Clone modes
            empirical_modes = zevals.clone().detach()
            empirical_modes_log_probs = torch.exp(self.log_prob(empirical_modes))

            empirical_modes = sort_with_scores(empirical_modes, -empirical_modes_log_probs, dim=0)
            empirical_modes_log_probs = torch.exp(self.log_prob(empirical_modes))

            return empirical_modes, empirical_modes_log_probs, loss_tot


def sort_with_scores(values, scores, dim):
    """ Sort values with scores along dim """

    # Get indices of all entries (~dirty trick)
    meshgrid = torch.where(scores == scores)

    # Index of sorted scores along desired dimension
    scores = torch.argsort(scores, dim=dim)

    # Copy values
    values_buffer = values.clone()

    # Loop over all element
    for jj in range(len(meshgrid[0])):
        # Get current coordinate
        coordinate = [mesh[jj].item() for mesh in meshgrid]

        # Replace coordinate with optimal value
        #values_buffer[*coordinate, :] = values[int(scores[*coordinate].numpy()), *coordinate[1:]]
        coordinate1 = tuple(coordinate)
        coordinate2 = tuple([int(scores[tuple(coordinate)].cpu().numpy()), *coordinate[1:]])
        values_buffer[coordinate1] = values[coordinate2]

    return values_buffer
