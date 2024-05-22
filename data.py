import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union, Tuple, List


class RPMData(Dataset):
    """
        Custom Dataset/Dataloader for RPM
    """
    def __init__(
            self,
            observations: Union[Tuple[torch.Tensor], List[torch.tensor]],
            latent_true: torch.Tensor = None,
            transform=None
    ):

        # Problem dimensions
        self.num_factors = len(observations)
        self.dim_observation = [obsi.shape[1:] for obsi in observations]

        # Device and data type
        self.dtype = observations[0].dtype
        self.device = observations[0].device

        # Sanity Check
        assert all(
            [obsi.shape[0] == observations[0].shape[0] for obsi in observations]
        ), "Inconsistent number of observations"

        # For latter use
        self.transform = transform

        # Store Observations
        self.observations = observations

        if latent_true is None:
            self.latent_true = torch.zeros(observations[0].shape[0], 0)
        else:
            self.latent_true = latent_true

    def __len__(self):
        return self.observations[0].shape[0]

    def __getitem__(self, idx):
        if self.transform:
            raise NotImplementedError()

        obs = [obsi[idx] for obsi in self.observations]
        lat = self.latent_true[idx]

        return obs, lat, idx


from typing import List, Union, Tuple


def merge_rpm_data(data_list: List[RPMData]):
    """ Combine multiple RPM datasets """

    # Store Shapes
    observations_num = [len(obs) for obs in data_list]
    start = np.cumsum([0, *observations_num])[:-1]
    stops = np.cumsum([0, *observations_num])[1:]
    indices = [range(i0, i1) for i0, i1 in zip(start, stops)]

    # Concatenate all Observations for all factors
    merged = [
        torch.cat(obs, dim = 0)
        for obs in zip(*[data.observations for data in data_list])
    ]

    return RPMData(merged), indices



