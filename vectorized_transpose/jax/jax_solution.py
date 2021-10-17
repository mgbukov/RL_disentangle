from jax import vmap
import jax.numpy as jnp 
import numpy as np
from itertools import permutations 


seed=0
np.random.seed(0)

# define 3D arrays
B=5
data = np.random.uniform(size=(B,2,2,2) )

# define axes to transpose: each array has a different transpose axis. 
all_transposed_axes = np.array( list(permutations(range(3), )), dtype=np.int8 )
random_inds = np.random.choice(np.arange(all_transposed_axes.shape[0]), size=B)
transposed_axes = all_transposed_axes[random_inds]

# check shapes 
print(data.shape, transposed_axes.shape) # (5, 2, 2, 2) (5, 3)

# define function do transpose a single 3D array
def data_transpose(data,transposed_axes):
	return data.transpose(transposed_axes)

# check that function works on a single data points
result = data_transpose(data[0,:], transposed_axes[0,:]) # works

# use vmap to vectorize above function
#vec_transpose = vmap(data_transpose,in_axes=(0,0), )

# call vectorized function
#result = vec_transpose(data, transposed_axes) # throws error



def dynamic_transpose(x, axes):
	axes = jnp.asarray(axes)
	assert len(set(x.shape)) == 1
	assert axes.shape == (x.ndim,)
	# create indices that correspond to slices of x along different dimensions
	indices = jnp.mgrid[tuple(slice(s) for s in x.shape)]
	# re-shuffle indices according to axes
	indices = indices[axes] 
	# do tranpose on indices of x
	return x[tuple(indices[i] for i in range(indices.shape[0]))]


dynamic_transpose(data[0], transposed_axes[0])
exit()


result_slow = jnp.array([x.transpose(p) for x, p in zip(data, transposed_axes)])
result_vmap = vmap(dynamic_transpose)(data, transposed_axes)
np.testing.assert_allclose(result_slow, result_vmap)








