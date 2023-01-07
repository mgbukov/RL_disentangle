import random

from matplotlib import cycler
import matplotlib.pyplot as plt
import numpy as np
import torch


def fix_random_seeds(seed):
    """Manually set the seed for random number generation.
    Also set CuDNN flags for reproducible results using deterministic algorithms.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_printoptions(precision, sci_mode):
    torch.set_printoptions(precision=precision, sci_mode=sci_mode)
    np.set_printoptions(precision=precision, suppress=~sci_mode)

def plt_style_use(figsize=None):
    """Use a custom stylesheet for creating plots.
    This function is similar to `plt.style.use("ggplot")`, however it introduces
    some customization.
    """
    # The cycler cycles through multiple properties producing a dictionary of
    # the type `{"color": "#EE6666"}`.
    # Use the cycler to set the `axes.prop_cycle` property.
    colors = cycler('color', [
        '#EE6666', '#3388BB', '#9988DD',
        '#EECC55', '#88BB44', '#FFBBBB',
    ])
    plt.rc("axes", prop_cycle=colors)

    # Set the background to gray.
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True)

    # Set the edge color of 2D faces to gray.
    plt.rc('patch', edgecolor='#E6E6E6')

    # Draw solid white grid lines.
    plt.rc('grid', color='w', linestyle='solid')

    # Draw ticks and labes on the outside in gray color.
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')

    plt.rc('lines', linewidth=2)

    # If a figsize parameter is provided then set it as default.
    if figsize is not None:
        plt.rc('figure', figsize=figsize)

#