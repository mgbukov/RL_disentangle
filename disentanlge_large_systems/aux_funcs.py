
import numpy as np 
import jax.numpy as jnp 
from jax import random, jit, partial


def reshape_state(psi,L,L_A,subsys='half'):
	
	# reshape state
	psi = psi.reshape((2,)*L,)   # (2,2,2, | 2,...,2)

	if subsys=='half':

		#psi = psi.transpose(system) # sub_sus_A = [0,2,4] (transpose)--> (0,2,4, | 1,3,5) # shift sub_sys_A to the left
		psi = psi.reshape(2**L_A, 2**(L-L_A))
	
	elif subsys=='site':

		#psi = psi.transpose(system) # sub_sus_A = [0,2,4] (transpose)--> (0,2,4, | 1,3,5) # shift sub_sys_A to the left
		psi = psi.reshape(2, 2**(L-1))

	return psi



@partial(jit, static_argnums=(1,2))
def ent_entropy(psi,L,subsys):

	L_A=L//2

	psi = reshape_state(psi,L,L_A,subsys=subsys)

	# ### faster for L >= 18
	if L>=18:
		#compute rdm
		rdm_A = psi @ psi.T.conj()

		# get eigenvalues of rdm
		lmbda2 = jnp.linalg.eigvalsh(rdm_A)
		lmbda2 += jnp.finfo(lmbda2.dtype).eps

		# compute entanglement entorpy
		Sent = -1.0/L_A * ( lmbda2 @ jnp.log(lmbda2) )

	else:
		### faster for L < 14
		lmbda = jnp.linalg.svd(psi, full_matrices=False, compute_uv=False)
		lmbda += jnp.finfo(lmbda.dtype).eps # shift lmbda to be positive within machine precision
		Sent = -2.0/L_A * ( lmbda**2 @ jnp.log(lmbda) )

	return Sent



@partial(jit, static_argnums=(1,2))
def compute_rdm(psi,L,sub_sys_A):

	sub_sys_B = tuple([j for j in range(L) if j not in sub_sys_A])
	system = sub_sys_A + sub_sys_B

	# reshape state
	psi = psi.reshape((2,)*L)   # (2,2,2, | 2,...,2)
	psi = psi.transpose(system) # sub_sus_A = [0,2,4] (transpose)--> (0,2,4, | 1,3,5) # shift sub_sys_A to the left
	psi = psi.reshape(4, 2**(L-2))

	### faster for L >= 14
	#compute rdm
	rdm_A = psi @ psi.T.conj()

	return rdm_A



def apply_2q_unitary(psi,U,subsys,L):

	rest = tuple([j for j in range(L) if j not in subsys])
	
	system = subsys + rest
	inv_system = np.argsort(system)

	psi = psi.reshape((2,)*L)
	psi = psi.transpose(system)

	U = U.reshape(2,2, 2,2)

	Upsi = np.einsum('ij kl, kl ... -> ij...',U,psi)

	Upsi = Upsi.transpose(inv_system)
	Upsi = Upsi.ravel()

	return Upsi



def compute_Sent(p):
	lmbda = p[0]+p[1]
	if np.abs(lmbda) <= 1E-8:
		return 0.0
	else:
		return -(lmbda * np.log(lmbda) + (1.0-lmbda) * np.log(1.0-lmbda) )




def compute_Sent_qubit(rdm):

	rdm_A = np.trace(rdm.reshape(2,2, 2,2), axis1=1, axis2=3)

	lmbda = 0.5*(1.0 - np.sqrt(1.0 - 4.0*np.linalg.det(rdm_A))).real
	if np.abs(lmbda-1.0) <= 1E-13:
		return 0.0
	else:
		return -(lmbda * np.log(lmbda) + (1.0-lmbda) * np.log(1.0-lmbda) )
	