import random
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

#