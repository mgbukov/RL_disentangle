# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as np 
import jax.numpy as jnp 
from jax import random, jit
from jax.experimental import stax
from jax.experimental.stax import GeneralConv, Dense, Flatten
from jax.experimental.stax import Dense, Relu, LogSoftmax # neural network layers

import time

# seed
seed = 0
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


########

# define system size
L=4	
N=4 

# define state
psi = np.random.uniform(size=(N,2**L,))
norms = np.linalg.norm(psi, axis=1)
psi = np.einsum('ij,i->ij', psi, 1.0/norms)

#psi = psi.reshape(-1,1,2**(L//2),2**(L//2))

psi = psi.reshape((-1,)+(2,)*L)

print(psi.shape)

####


dim_nums=('NCHW', 'OIHW', 'NCHW') # default 

out_chan = 2 
filter_shape = (5,5)	

out_dim = 1

init_fun, apply_fun = stax.serial(	
									Dense(out_dim,)
									# GeneralConv(dim_nums, 16, (4,4), strides=(1,1) ), # 
								)

input_shape = (-1, 4,) 

output_shape, init_params = init_fun(rng,input_shape)

print(output_shape)




