from transpose_lib import swap_axes
import numpy as np 
from itertools import permutations

import time

seed=0
np.random.seed(0)

B=2**18
L=6

psi = np.random.uniform(size=(B,)+(2,)*L, ).astype(np.complex128)
psi_T = np.zeros_like(psi)

sites=list(np.arange(L))
all_transposed_axes = np.array( list(permutations(sites, )), dtype=np.uint8 )

random_inds = np.random.choice(np.arange(all_transposed_axes.shape[0]), size=B)
transposed_axes = all_transposed_axes[random_inds]

transposed_axes_tuples = np.ndarray(shape=(B,), dtype=tuple, )
for i in range(B):
	transposed_axes_tuples[i] = transposed_axes[i] 

#transposed_axes_tuples=transposed_axes_tuples.astype(tuple)


#print(psi.shape, transposed_axes.shape, psi_T.shape)



# ti=time.time()
# swap_axes(B, psi, transposed_axes, psi_T)
# tf=time.time()

# print('cython took {:0.4f}'.format(tf-ti))


def psi_transpose(psi,transposed_axes):
	return psi.transpose(transposed_axes)


ti=time.time()
for i in range(B):
	psi_T[i] = psi_transpose(psi[i],transposed_axes[i])
tf=time.time()

print('python took {:0.4f}'.format(tf-ti))


#######

from jax import vmap, jit
import jax.numpy as jnp 


@jit
def dynamic_transpose(x, axes):
	axes = jnp.asarray(axes)
	# assert len(set(x.shape)) == 1
	# assert axes.shape == (x.ndim,)
	# create indices that correspond to slides of x along different dimensions
	indices = jnp.mgrid[tuple(slice(s) for s in x.shape)]
	# re-shuffle indices according to axes
	indices = indices[axes] 
	# do tranpose on indices of x
	return x[tuple(indices[i] for i in range(indices.shape[0]))]


#result_slow = jnp.array([x.transpose(p) for x, p in zip(data, transposed_axes)])

vmap_transpose = vmap(dynamic_transpose)
vmap_transpose(psi, transposed_axes)


ti=time.time()
result_vmap = vmap_transpose(psi, transposed_axes)
tf=time.time()

print('jax took {:0.4f}'.format(tf-ti))

#np.testing.assert_allclose(result_slow, result_vmap)

