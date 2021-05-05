# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as np 
import jax.numpy as jnp 
from jax import random, jit

import time

# seed
seed = 0
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


########



