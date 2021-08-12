import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from jax import jit 
# import jax.numpy as jnp 
# import jax.scipy as jsp


def compute_Sent(angle,  H,psi):
	# define gate
	U = expm(-1j * angle * H)
	# apply gate
	psi_new = U @ psi
	# compute rdm
	psi_new = psi_new.reshape(2,2)
	rdm_A = psi_new @ psi_new.T.conj()
	# compute e'values of rdm
	tr_rdm_A=(rdm_A[0,0]+rdm_A[1,1]).real
	det_rdm_A=(rdm_A[0,0]*rdm_A[1,1] - rdm_A[0,1]*rdm_A[1,0]).real
	lmbda = 0.5*( tr_rdm_A + np.sqrt(tr_rdm_A**2 - 4.0*det_rdm_A + np.finfo(det_rdm_A.dtype).eps) ) 
	# compute entanglement
	lmbda += np.finfo(lmbda.dtype).eps
	Sent = -lmbda*np.log(lmbda) - (1.0-lmbda)*np.log(1.0-lmbda)
	#
	return Sent

# @jit
# def compute_Sent(angle,  H,psi):
# 	# define gate
# 	U = jsp.linalg.expm(-1j * angle * H)
# 	# apply gate
# 	psi_new = U @ psi
# 	# compute rdm
# 	psi_new = psi_new.reshape(2,2)
# 	rdm_A = psi_new @ psi_new.T.conj()
# 	# compute e'values of rdm
# 	tr_rdm_A=(rdm_A[0,0]+rdm_A[1,1]).real
# 	det_rdm_A=(rdm_A[0,0]*rdm_A[1,1] - rdm_A[0,1]*rdm_A[1,0]).real
# 	lmbda = 0.5*( tr_rdm_A + jnp.sqrt(tr_rdm_A**2 - 4.0*det_rdm_A + jnp.finfo(det_rdm_A.dtype).eps) ) 
# 	# compute entanglement
# 	lmbda += jnp.finfo(lmbda.dtype).eps
# 	Sent = -lmbda*jnp.log(lmbda) - (1.0-lmbda)*jnp.log(1.0-lmbda)
# 	#
# 	return Sent



class TabularQubitEnv():
	"""
	Gym style environment for RL. You may also inherit the class structure from OpenAI Gym. 
	Parameters:
		seed:   int
				seed of the RNG (for reproducibility)
	"""
	
	def __init__(self, seed):
		"""
		Initialize the environment.
		
		"""
		
		# maximum number of time steps
		self.max_time_steps = 2

		### define action space variables
		# define Pauli matrices
		self.Id	    =np.array([[1.0,0.0  ], [0.0 ,+1.0]])
		self.sigma_x=np.array([[0.0,1.0  ], [1.0 , 0.0]])
		self.sigma_y=np.array([[0.0,-1.0j], [1.0j, 0.0]])
		self.sigma_z=np.array([[1.0,0.0  ], [0.0 ,-1.0]])
		

		generator=lambda sigma_1, sigma_2: np.kron(sigma_1,sigma_2)

		# self.action_space=[ generator(self.sigma_x,self.sigma_x), generator(self.sigma_x,self.sigma_y), generator(self.sigma_x,self.sigma_z),
		# 					generator(self.sigma_y,self.sigma_x), generator(self.sigma_y,self.sigma_y), generator(self.sigma_y,self.sigma_z),
		# 					generator(self.sigma_z,self.sigma_x), generator(self.sigma_z,self.sigma_y), generator(self.sigma_z,self.sigma_z),
		#   				]

		self.action_space=[ generator(self.sigma_x,self.sigma_x), 
							generator(self.sigma_x,self.sigma_y), 
							generator(self.sigma_x,self.sigma_z),
		  				]
		
		
		### define state space variables

		# define function to compute state coordinates
		self.get_indices = lambda S: np.array([np.searchsorted(self.theta_grid, S[0], side='right') - 1, 
											   np.searchsorted(self.phi_grid,   S[1], side='right') - 1,
											   np.searchsorted(self.r_grid,     S[2], side='right') - 1
											 ])

		# define grids
		self.theta_grid=np.linspace(0.0,np.pi+1E-12,51)
		self.phi_grid  =np.linspace(-np.pi,np.pi+1E-12,101)
		self.r_grid    =np.linspace(0.0,1.0+1E-12,101)

		# define target states
		self.cap_size = 1E-3 # size of polar cap around psi_target to define terminal states

		
		# set seed
		self.set_seed(seed)
		self.reset(random=False)


	
	
	def step(self, action):
		"""
		Interface between environment and agent. Performs one step in the environemnt.
		Parameters:
			action: int
					the index of the respective action in the action array
		Returns:
			output: ( object, float, bool)
					information provided by the environment about its current state:
					(state, reward, done)
		"""

		# get action
		H = self.action_space[action]

		# compute optimal angle and ent_entropy
		angle_0=np.pi/np.exp(1.0) # initial condition for solver
		res = minimize(compute_Sent, angle_0, args=(H,self.psi), method='Nelder-Mead', tol=1e-3*self.cap_size)
		angle=res.x[0]
		Sent=res.fun
		Sent_normed=Sent/np.log(2.0) - np.finfo(Sent.dtype).eps
		#print('action, angle, Sent, success:', action, res.x[0], res.fun, res.success)
		if not res.success:
			print('solver integraton unsuccessful')
			exit()


		# apply gate to quantum state
		self.psi = expm(-1j * angle * H) @ self.psi
		# compute RL state
		self.state = self.QM_to_RL_state(self.psi)


		# compute reward
		#reward = np.log(1.0 - Sent_normed) # resolves x ~ 0
		reward = -np.log(-np.log(1.0 - Sent_normed)) # resolves both x ~ 0 and x ~ 1
	   	
		# check if state is terminal or number of max steps is exceeded
		done=False
		if Sent_normed < self.cap_size or self.env_step >= self.max_time_steps:
			done=True

			# print messages
			if Sent_normed < self.cap_size:
				print('state prepared')
			else:
				print('ent entropy', Sent )
 

		# increment time step
		self.env_step += 1
		

		return self.state, reward, done

	
	
	def set_seed(self,seed=0):
		"""
		Sets the seed of the RNG.
		
		"""
		np.random.seed(seed)
	
	
	
	def reset(self, random=True):
		"""
		Resets the environment to its initial values.
		Returns:
			state:  object
					the initial state of the environment
			random: bool
					controls whether the initial state is a random state on the sphere or a fixed initial state.
		"""

		self.env_step = 0
		
		if random:
			self.psi=np.random.uniform(0.0,1.0,size=4) + 1j*np.random.uniform(0.0,1.0,size=4)
			self.psi/=np.linalg.norm(self.psi)
		else:
			# start from Bell state
			self.psi=np.array([0.0,1.0,1.0,0.0])/np.sqrt(2)
		

		# compute RL state
		self.state=self.QM_to_RL_state(self.psi) 
		
		return self.state

	
	
	def render(self):
		"""
		Plots the state of the environment. For visulization purposes only. 

		"""
		pass
	
	
	def RL_state_to_rdm(self,s):
		"""
		Take as input the RL state s, and return the quantum state |psi>
		"""
		theta, phi, r = self.theta_grid[s[0]], self.phi_grid[s[1]], self.r_grid[s[2]]
		rho = 0.5*( self.Id + r*np.sin(theta)*np.cos(phi)*self.sigma_x + r*np.sin(theta)*np.sin(phi)*self.sigma_y + r*np.cos(theta)*self.sigma_z )
		return rho
	
	
	def QM_to_RL_state(self,psi):
		"""
		Take as input the RL state s, and return the quantum state |psi><psi|
		"""

		# compute rdm
		psi = psi.reshape(2,2)
		rho = psi @ psi.T.conj()

		# compute projections
		r_x = np.trace(rho @ self.sigma_x).real
		r_y = np.trace(rho @ self.sigma_y).real
		r_z = np.trace(rho @ self.sigma_z).real

  
		# find Bloch sphere coords
		r = np.sqrt(r_x**2 + r_y**2 + r_z**2)
		theta = np.arccos(r_z/(r+np.finfo(r.dtype).eps)).real
		phi = np.arctan2(r_y,r_x)

		# print(rho)
		# print(r_x,r_y,r_z)
		# print('here', r)

		return self.get_indices( np.array([theta, phi, r]) )


