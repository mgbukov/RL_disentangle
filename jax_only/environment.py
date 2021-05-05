from functools import partial
import numpy as np 
from scipy.sparse.linalg import expm
import scipy.optimize as sp_opt

import jax.numpy as jnp 
import jax.scipy as jsp


import jax.scipy.optimize as jsp_opt

from jax import random, jit

from unitaries import create_unitaries

import time


class env(object):

	def __init__(self, L):

		# physics paramsters
		self.L=L

		# define subsystem
		sub_sys_A = [j for j in range(L//2)]
		sub_sys_B = [j for j in range(L) if j not in sub_sys_A]
		self.system = sub_sys_A + sub_sys_B
		self.system_vec = (0,) + tuple(site+1 for site in self.system)


		self.L_A = len(sub_sys_A)
		self.L_B = len(sub_sys_B)

		# RL params
		self.N_actions=9
		self.N_MC=8

		self._construct_actions()


		np.random.seed(0)

		psi = np.random.uniform(size=(self.N_MC,2**L,))
		norms = np.linalg.norm(psi, axis=1)
		psi = np.einsum('ij,i->ij', psi, 1.0/norms)

		states = psi
		#states[-1,...] = np.array( [ 0.37668964209437095, -0.2142433600384019,  0.705579958336652,  0.5606796042410662] )


		actions=jnp.array( np.random.randint(0,self.N_actions,size=self.N_MC) )
		
		
		
		self.compute_entropies(states, actions)
		self.compute_entropies2(states, actions)

		exit()

		

	def _construct_actions(self):
		self.hamiltonians = jnp.array( create_unitaries(self.L) )
		self.unitaries = lambda phi_vec: jnp.array( list( jsp.linalg.expm(-1j*phi_vec[j]*self.hamiltonians[j]) for j in range(self.N_actions) ) )

		self.angles_0=np.pi/np.exp(1)*jnp.ones(self.N_MC) # initial condition for solver
		

		# phis = [1.0,]*self.N_actions

		# print( self.unitaries(phis).shape)


	def step(self,action):
		pass

	
	


	def compute_entropies(self, states, actions):
	
		#unitaries = lambda phi_vec: jnp.array( list( jsp.linalg.expm(-1j*phi_vec[j]*self.hamiltonians[actions[j]]) for j in range(self.N_MC) ) )
		new_states = lambda phi_vec: jnp.array( list( jsp.linalg.expm(-1j*phi_vec[j]*self.hamiltonians[actions[j]]) @ states[j] for j in range(self.N_MC) ) )

		
		@jit
		def compute_angles(angles,):
			# new_states = jnp.einsum('ijk,ik->ij', unitaries(angles), states)
			# Sent=jnp.sum( self._ent_entropies(new_states) )	
			Sent=jnp.sum( self._ent_entropies( new_states(angles) ) )	
			return Sent 


		t_i=time.time()
		res = sp_opt.minimize(compute_angles, self.angles_0, args=(),  method='Nelder-Mead', tol=1e-7)
		t_f=time.time()

		self.angles=jnp.array(res.x)
		self.entropies = self._ent_entropies( new_states(self.angles) )

		print('\nentanglement calculation min(Sent) = {0:0.4f}: {1:d} evals took {2:0.6f} secs.\n'.format(np.sum(self.entropies), res.nfev, t_f-t_i))
		
		print(self.angles)
		print(self.entropies)

		return self.entropies

	def compute_entropies2(self, states, actions):
	
		@jit
		def compute_angle(angle,  H,psi):
			U = jsp.linalg.expm(-1j*angle*H)
			psi_new = U @ psi
			Sent = self._ent_entropy(psi_new)
			return Sent

		
		angles=[]
		entropies=[]

		t_i=time.time()

		fun_evals=0
		for j in range(self.N_MC):

			H=self.hamiltonians[actions[j]]
			psi=states[j]
			res = sp_opt.minimize(compute_angle, self.angles_0[0], args=(H,psi),  method='Nelder-Mead', tol=1e-4)
	
			angles.append( res.x[0] )
			entropies.append( res.fun )

			fun_evals+=res.nfev

		t_f=time.time()

		print('\nentanglement calculation min(Sent) = {0:0.4f}: {1:d} evals took {2:0.6f} secs.\n'.format(np.sum(entropies), fun_evals, t_f-t_i))
		
		print(jnp.array(angles))
		print(jnp.array(entropies))

		
		return entropies


	def _ent_entropy(self, psi,):

		# reshape state
		psi = psi.reshape((2,)*self.L)   # (2,2,2, | 2,...,2)
		psi = psi.transpose(self.system) # sub_sus_A = [0,2,4] (transpose)--> (0,2,4, | 1,3,5) # shift sub_sys_A to the left
		psi = psi.reshape(2**self.L_A, -1)

		# ### faster for L >= 14
		# #compute rdm
		# rdm_A = psi @ psi.T.conj()

		# # get eigenvalues of rdm
		# lmbda = jnp.linalg.eigvalsh(rdm_A)
		# lmbda += jnp.finfo(lmbda.dtype).eps

		# # compute entanglement entorpy
		# Sent = -1.0/L_A * ( lmbda @ jnp.log(lmbda) )

		### faster for L < 14
		lmbda = jnp.linalg.svd(psi, full_matrices=False, compute_uv=False)
		lmbda += jnp.finfo(lmbda.dtype).eps # shift lmbda to be positive within machine precision
		Sent = -2.0/self.L_A * ( lmbda**2 @ jnp.log(lmbda) )

		return Sent

	def _ent_entropies(self, states,):

		# reshape state
		states = states.reshape((-1,)+(2,)*self.L)
		states = states.transpose(self.system_vec)
		states = states.reshape(-1, 2**self.L_A, 2**self.L_B,)

		# ### faster for L >= 14

		# #compute rdm
		# rdm_A = jnp.einsum('aij,akj->aik', states, states.conj() )

		# # get eigenvalues of rdm
		# lmbda = jnp.linalg.eigvalsh(rdm_A) 
		# lmbda += jnp.finfo(lmbda.dtype).eps

		# # compute entanglement entropy
		# Sent = -1.0/L_A * jnp.einsum('ai,ai->a', lmbda, jnp.log(lmbda) )

		### faster for L < 14
		lmbda = jnp.linalg.svd(states, full_matrices=False, compute_uv=False)
		lmbda += jnp.finfo(lmbda.dtype).eps
		Sent = -2.0/self.L_A * jnp.einsum('ai,ai->a', lmbda**2, jnp.log(lmbda) )

		return Sent


	def reset(self):
		pass

env(2)
