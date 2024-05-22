import torch
import numpy as np
from matplotlib import pyplot as plt


def plot_confusion(matrix, label = None, overlay_text = True, normalize= True,  **kwargs):

    # Normalize confusion matrix
    matrix = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix

    # Plot Matrix
    plt.imshow(matrix, **kwargs)

    if any(label == None) == None:
        label = np.arange(matrix.shape[0])
    plt.xticks(np.arange(8), label, rotation=25)
    plt.yticks(np.arange(8), label, rotation=0)

    if overlay_text:
        # Add the text
        x_start = 0 - 0.5
        y_start = 0 - 0.5
        x_end = matrix.shape[0] - 0.5
        y_end = matrix.shape[1] - 0.5
        size = len(matrix)

        # Add the text
        jump_x = (x_end - x_start) / (2.0 * size)
        jump_y = (y_end - y_start) / (2.0 * size)
        x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
        y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = np.round(matrix[y_index, x_index], 2)
                text_x = x + jump_x
                text_y = y + jump_y
                plt.text(text_x, text_y, label, color='black', ha='center', va='center')


def print_loss(loss, epoch_id, epoch_num, pct=0.001):
    """ Simple logger"""
    str_epoch = 'Epoch ' + str(epoch_id) + '/' + str(epoch_num)
    str_loss = ' Loss: %.6e' % loss

    if epoch_num < int(1/pct) or epoch_id % int(epoch_num * pct) == 0:
        print(str_epoch + str_loss)
        

def get_modulator0(x, amp):
    return (x - 0.5) / (1 - 0.5) * amp + (x - 1) / (0.5 - 1) * 1


def get_modulator1(x, amp):
    y = x
    y[x > 0.5] = get_modulator0(x[x > 0.5], amp)
    y[x <= 0.5] = 1/get_modulator0(1 - x[x <= 0.5], amp)

    return y


def get_color(index_base, index_sub=None, cmap=None, amp=3):

    # Init Base categories Color Map
    if cmap is None:
        cmap = plt.cm.tab10(np.linspace(0, 1, 10))

    # Get Base categories Color
    num_categories_basic = len(np.unique(index_base))
    colors_basic = cmap[:num_categories_basic]

    # Get Subcategories Color
    if index_sub is None:
        colors_categories = None

    else:
        num_categories = len(np.unique(index_sub))
        colors_categories = np.zeros((num_categories, 4))

        for ii in range(num_categories_basic):

            # Select Subcategories
            mask = np.where(index_base == ii)

            # Get Indices and numbers
            sub_categories_index = index_sub[mask]
            sub_categories_index_unique = np.unique(sub_categories_index)
            sub_categories_index_num = len(sub_categories_index_unique)

            # Current base color
            color_basic_cur = colors_basic[ii]

            # Color Modulation Index
            modulation_index = np.expand_dims(get_modulator1(np.linspace(0, 1, sub_categories_index_num), amp=amp), axis=1)

            # Modulate base color
            modulated_color = color_basic_cur ** modulation_index
            modulated_color[:, 3] = 1
            colors_categories[sub_categories_index_unique] = modulated_color

    return colors_basic, colors_categories


def diagonalize(z):
    """ Use a batch vector to create diagonal batch matrices """
    Z = torch.zeros((*z.shape, z.shape[-1]), device=z.device, dtype=z.dtype)
    Z[..., range(z.shape[-1]), range(z.shape[-1])] = z
    return Z


def chol_inv_det(nsd):
    chol = torch.linalg.cholesky(-nsd)
    inv = - torch.cholesky_inverse(chol)
    det = 2 * torch.log(chol.diagonal(dim1=-1, dim2=-2)).sum(dim=-1)

    return inv, det


def annealing_closure(
        num_epoch,
        plateau,
        period,
        vmin=0,
        vmax=1
):
    # Normalize x to zero - one from 0 to num_epoch
    f1 = lambda x: x / num_epoch

    # Build a Periodic ramping Function from 0 to one
    f2 = lambda x: np.mod(x, 1 / period) * period

    # Min(f2, plateau)
    f3 = lambda x: (np.abs(x + plateau) - np.abs(x - plateau)) / (2 * plateau)

    # Scale max and min variation
    f4 = lambda x: vmin + (vmax - vmin) * x

    return lambda x: f4(f3(f2(f1(x))))