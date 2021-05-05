# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as np 
import jax.numpy as jnp 
from jax import random, jit
from jax.experimental import stax
from jax.experimental.stax import GeneralConv, Dense, Flatten
from jax.experimental.stax import Dense, Relu, LogSoftmax # neural network layers

from scipy.special import comb
from jax.scipy.special import logsumexp

import time

# seed
seed = 0
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


########

# define system size
L=4	
N=3 

# define state
psi = np.random.uniform(size=(N,2**L,))
norms = np.linalg.norm(psi, axis=1)
psi = np.einsum('ij,i->ij', psi, 1.0/norms)

#psi = psi.reshape(-1,1,2**(L//2),2**(L//2))

psi = psi.reshape((-1,)+(2,)*L)

print(psi.shape)


##########################################


assert(L==4)

U11=np.random.uniform(size=(2,2, 10,10))
U12=np.random.uniform(size=(2,2, 10,10))
b1=np.random.uniform(size=(10,10))

W21=np.random.uniform(size=(10,10, 20))
W22=np.random.uniform(size=(10,10, 20))
b2=np.random.uniform(size=(20,))

U3=np.random.uniform(size=(20,20, 14,14))
b3=np.random.uniform(size=(14,14))

W4=np.random.uniform(size=(14,14, 9,))
b4=np.random.uniform(size=(9,))


psi_1 = jnp.einsum('...klmn,klij,mnab->...ijab',psi,U11,U12) + b1

psi_2 = jnp.einsum('...ijkl,jks,lit->...st',psi_1,W21,W22) + b2

psi_3 = jnp.einsum('...ij,ijkl->...kl',psi_2,U3) + b3

psi_4 = jnp.einsum('...kl,kls->...s',psi_3,W4) + b4

print(psi_1.shape, psi_2.shape, psi_3.shape, psi_4.shape)

exit()


##################################################

assert(L==2)


N_actions = 3**2 * comb(L,2, exact=True)

layer_sizes = (L, 100, N_actions)


def init_random_params(scale, layer_sizes, rng=np.random.RandomState(0)):

	params=()
	for j, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):

		input_shape = (m,)*L 

		if j%2==0:
			output_shape = (n,)*L
		else:
			output_shape = (n,)*(L//2)

		params += ((scale * rng.randn(*(input_shape+output_shape)), scale * rng.randn(*output_shape), ), )

	return params



params=init_random_params(1.0, layer_sizes )


def predict(params, inputs):
	activations = inputs
	
	U, U_b = params[0]
	outputs = jnp.einsum('...ij,ijkl->...kl',activations, U) + U_b
	activations = jnp.tanh(outputs)

	W, W_b = params[1]
	logits = jnp.einsum('...ij,ijk->...k',activations, W) + W_b
	
	return logits - logsumexp(logits, axis=1, keepdims=True)


preds = predict(params, psi)

print(preds.shape)








