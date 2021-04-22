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

# define system size
L=10	
N_trials = 100	

# define subsystem
sub_sys_A = ([j for j in range(L//2)])
sub_sys_B = ([j for j in range(L) if j not in sub_sys_A])
system = sub_sys_A + sub_sys_B
system2 = (0,) + tuple(site+1 for site in system)

L_A = len(sub_sys_A)
L_B = len(sub_sys_B)


# define state
psi = np.random.uniform(size=(N_trials,2**L,))
norms = np.linalg.norm(psi, axis=1)
psi = np.einsum('ij,i->ij', psi, 1.0/norms)

########

@jit
def ent_entropy(psi,):

	# reshape state
	psi = psi.reshape((2,)*L)
	psi = psi.transpose(system)
	psi = psi.reshape(2**L_A, 2**L_B)

	# compute rdm
	rdm_A = psi @ psi.T.conj()

	# ### faster for L >= 14
	# # get eigenvalues of rdm
	# lmbda = jnp.linalg.eigvalsh(rdm_A)
	# lmbda += jnp.finfo(lmbda.dtype).eps

	# # compute entanglement entorpy
	# Sent = -1.0/L_A * ( lmbda @ jnp.log(lmbda) )

	### faster for L < 14
	lmbda = jnp.linalg.svd(psi, full_matrices=False, compute_uv=False)
	lmbda += jnp.finfo(lmbda.dtype).eps
	Sent = -2.0/L_A * ( lmbda**2 @ jnp.log(lmbda) )

	return Sent


@jit
def ent_entropies(psi,):

	# reshape state
	psi = psi.reshape((-1,)+(2,)*L)
	psi = psi.transpose(system2)
	psi = psi.reshape(N_trials, 2**L_A, 2**L_B,)

	# compute rdm
	rdm_A = jnp.einsum('aij,akj->aik', psi, psi.conj() )

	# ### faster for L >= 14
	# # get eigenvalues of rdm
	# lmbda = jnp.linalg.eigvalsh(rdm_A) 
	# lmbda += jnp.finfo(lmbda.dtype).eps

	# # compute entanglement entropy
	# Sent = -1.0/L_A * jnp.einsum('ai,ai->a', lmbda, jnp.log(lmbda) )

	### faster for L < 14
	lmbda = jnp.linalg.svd(psi, full_matrices=False, compute_uv=False)
	lmbda += jnp.finfo(lmbda.dtype).eps
	Sent = -2.0/L_A * jnp.einsum('ai,ai->a', lmbda**2, jnp.log(lmbda) )

	return Sent


# call once to take out jitting time
ent_entropy(psi[0], ).block_until_ready()
ent_entropies(psi, ).block_until_ready()


t_vec=[]
for j in range(N_trials):

	# # get new key for rng
	# rng,_ = random.split(rng) 
	# # define state
	# psi = random.uniform(rng,shape=(2**L,))
	# psi /= jnp.linalg.norm(psi)


	t_i=time.time()

	Sent = ent_entropy(psi[j], ).block_until_ready()

	t_f=time.time()

	t_vec.append(t_f-t_i)

	#print("iteration {0} with Sent = {1:0.4f} took {2:4f} secs.".format(j,Sent, t_f-t_i) )





print("\njax: total per-state calculation took {0:4f} secs.".format(np.sum(t_vec)) )

t_i=time.time()
Sent=ent_entropies(psi, ).block_until_ready()
t_f=time.time()


print("\njax: joint calcualtion took {0:4f} secs.".format(t_f-t_i) )
#print(Sent)


print("\n")


###############################################################################################


from quspin.basis import spin_basis_general

basis=spin_basis_general(L)

t_vec=[]

for j in range(N_trials):

	t_i=time.time()

	Sent = basis.ent_entropy(psi[j], sub_sys_A)['Sent_A']

	t_f=time.time()

	t_vec.append(t_f-t_i)

	#print("iteration {0} with Sent = {1:0.4f} took {2:4f} secs.".format(j,Sent, t_f-t_i) )

print("\nquspin: total per-state calculation took {0:4f} secs.".format(np.sum(t_vec)) )



t_i=time.time()
Sent = basis.ent_entropy(psi.T, sub_sys_A, enforce_pure=True)['Sent_A']
t_f=time.time()


print("\nquspin: joint calculation took {0:4f} secs.".format(t_f-t_i) )
#print(Sent)







