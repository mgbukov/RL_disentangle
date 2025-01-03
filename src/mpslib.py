import numpy as np 


def factor_site(j,Lambda_prev,Theta,chi_vec,d=2):
	"""
	This function performs the decomposition of a Theta tensor on bond j,j+1 into a Gamma tensor on site j, and a 
	(truncated/compressed) diagonal Lambda tensor on bond j,j+1; it also returns the new Theta_next tensor on
	bond j+1,j+2

	Theta = X @ S @ Y 

	Gamma = Lambda_prev^{-1} @ X
	Lambda =  truncate(S)
	Theta_next = Lambda @ Y 


	Parameters
	----------
	j: int
		spin/lattice site number
	Lambda_prev: np.array
		Lambda tensor between sites j-1, j
	Theta: np.array
		Theta tensor on sites j,j+1
	chi_vec: np.array([int])
		array with maximum bond dimensions: convention chi_vec[0]=1=chi_vec[-1]; controls compression via maximum
		possible entanglement
	d: int
		on-site Hilbert space dimension (d=2 for spins)

	Returns
	-------
	Gamma: np.array
		Gamma tensor on site j
	Lambda: np.array
		Lambda tensor on bond j,j+1
	Theta_next: np.array
		Theta tensor on bond j+1,j+2

	"""

	# decompose Theta using SVD
	X,Lambda,Y = np.linalg.svd( Theta.reshape(chi_vec[j]*d, -1), full_matrices=False)

	# identify diagonal tensor Lambda, left tensor X, and right tensor Y
	Lambda = Lambda[:chi_vec[j+1]]/np.sqrt(np.sum(np.abs(Lambda[:chi_vec[j+1]])**2))
	X = X[:,:chi_vec[j+1]].reshape(chi_vec[j]  ,d,chi_vec[j+1])	
	Y = Y[:chi_vec[j+1],:]

	# construct Gamma and new Theta
	Gamma = np.einsum('a,aib->aib',  np.divide(1.0, Lambda_prev, out=np.zeros_like(Lambda_prev), where=np.abs(Lambda_prev)>=1E-14), X.reshape(chi_vec[j], d, chi_vec[j+1]), )
	Theta_next = np.einsum('a,as->as',Lambda,Y)

	return Gamma, Lambda, Theta_next


def state_to_MPS(psi,chi_vec,L,d=2):
	"""
	This function takes a state `psi` represented as a vector in Hilbert space, and decomposes it into MPS 
	(in canonical form) according to the bond dimensions stored in `chi_vec`:

	psi = Lambda[0] @ Gamma[0] @ Lambda[1] @ Gamma[1] @ Lambda[2] @ ... @ Lambda[L-1] @ Gamma[L-1] @ Lambda[L]

	For example, for L=4: 

	psi = np.einsum('a,aib,b,bjc,c,ckd,d,dle,e->ijkl', Lambdas[0],Gammas[0],Lambdas[1],Gammas[1],Lambdas[2],Gammas[2],Lambdas[3],Gammas[3],Lambdas[4] ).reshape(-1,)


	Parameters
	----------
	psi: np.array 
		quantum state as vector in Hilbert space
	chi_vec: np.array([int])
		array with maximum bond dimensions: convention chi_vec[0]=1=chi_vec[-1]; controls compression via maximum
		possible entanglement
	L: int
		number of spins/qubits/lattice sites
	d: int
		on-site Hilbert space dimension (d=2 for spins)

	Returns
	-------
	Gammas: list[np.array]
		list of Gamma tensors, one for each lattice site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)

	"""

	# initialize MPS data
	Gammas =[]
	Lambdas=[np.array([1.0,]),]
	Theta=psi.copy()

	for j in range(L):
		Gamma, Lambda, Theta = factor_site(j,Lambdas[j],Theta,chi_vec,d=d)
		Gammas.append(Gamma)
		Lambdas.append(Lambda)

	return Gammas, Lambdas


def MPS_to_state(Tensors, Lambdas, canonical=0):
	"""
	This function takes a state in its MPS form (left-canonica, right-canonical, or canonical representation), and
	returns the state as a vector in Hilbert space.

	Parameters
	----------
	Tensors: list[np.array]
		list of on-site tensors (either A, B or Gamma), depending on the MPS canonical form, one for each site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)
	canonical: int=[-1,0,1]
		integer to determine canonical form of input state: 
			* canonical=-1: left canonical (Tensors=[A[0],A[1],...,A[L-1]]), psi = A[0] @ A[1] ... A[L-1]
			* canonical= 0: canonical, (Tensors=[Gamma[0],Gamma[1],...,Gamma[L-1]]), psi = Lambda[0] @ Gamma[0] @ Lambda[1] @ Gamma[1] ... @ Lambda[L-1] @ Gamma[L-1] @ Lambda[L]
			* canonical=+1: right canonical (Tensors=[B[0],B[1],...,B[L-1]]), psi = B[0] @ B[1] ... B[L-1]

	Returns
	-------
	psi: np.array
		state vector in Hilbert space corresponding to input MPS state

	"""

	if canonical==0: # canonical form, here Tensors = Gammas
		psi=Lambdas[0]
		for j in range(len(Tensors)):
			psi = np.einsum('...a,aib,b->...ib', psi, Tensors[j],Lambdas[j+1], )
	
	else: # left or right canonical form
		psi=np.array([1.0])
		for j in range(len(Tensors)):
			psi = np.einsum('...a,aib->...ib', psi, Tensors[j] )

	return psi.reshape(-1,)


def to_left_canonical(Gammas,Lambdas):
	"""
	Takes MPS in canonical form and transforms it to left-canonical form.

	A[0] @ A[1] ... A[L-1] = Lambda[0] @ Gamma[0] @ Lambda[1] @ Gamma[1] ... @ Lambda[L-1] @ Gamma[L-1] @ Lambda[L]
	
	Parameters
	----------
	Gammas: list[np.array]
		list of Gamma tensors, one for each lattice site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)

	Returns
	-------
	As: list[np.array]
		list of A tensors, one for each lattice site

	"""

	As=[]
	for j in range(len(Gammas)):
		A=np.einsum('a,aib->aib',Lambdas[j],Gammas[j])
		As.append(A)
	return As

def to_right_canonical(Gammas,Lambdas):
	"""
	Takes MPS in canonical form and transforms it to right-canonical form.

	B[0] @ B[1] ... B[L-1] = Lambda[0] @ Gamma[0] @ Lambda[1] @ Gamma[1] ... @ Lambda[L-1] @ Gamma[L-1] @ Lambda[L]
	
	Parameters
	----------
	Gammas: list[np.array]
		list of Gamma tensors, one for each lattice site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)

	Returns
	-------
	Bs: list[np.array]
		list of B tensors, one for each lattice site

	"""
	Bs=[]
	for j in range(len(Gammas)):
		B=np.einsum('aib,b->aib',Gammas[j],Lambdas[j+1])
		Bs.append(B)
	return Bs


def to_canonical(Tensors,Lambdas,chi_vec,L,canonical=-1):
	"""
	Transforms state frolm left/right canonical to canonical form:

	Tensor[0] @ Tensor[1] @ ... @ Tensor[L-1] = Lambda[0] @ Gamma[0] @ Lambda[1] @ Gamma[1] ... @ Lambda[L-1] @ Gamma[L-1] @ Lambda[L]
			

	Parameters
	----------
	Tensors: list[np.array]
		list of on-site tensors (either A, B or Gamma), depending on the MPS canonical form, one for each site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)
	chi_vec: np.array([int])
		array with maximum bond dimensions: convention chi_vec[0]=1=chi_vec[-1]; controls compression via maximum
		possible entanglement
	L: int
		number of spins/qubits/lattice sites
	canonical: int=[-1,1]
		integer to determine canonical form of input state: 
			* canonical=-1: left canonical (Tensors=[A[0],A[1],...,A[L-1]]), psi = A[0] @ A[1] ... A[L-1]
			* canonical=+1: right canonical (Tensors=[B[0],B[1],...,B[L-1]]), psi = B[0] @ B[1] ... B[L-1]

	Returns
	-------
	Gammas: list[np.array]
		list of Gamma tensors, one for each lattice site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)


	"""
	# this is a lazy implementation; there's probably a more efficient one
	psi=MPS_to_state(Tensors, Lambdas, canonical=canonical)
	return state_to_MPS(psi,chi_vec,L)



def compute_MPS_norm(Tensors,Lambdas,canonical=0):
	"""
	Computes norm of MPS state.


	Parameters
	----------
	Tensors: list[np.array]
		list of on-site tensors (either A, B or Gamma), depending on the MPS canonical form, one for each site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)
	canonical: int=[-1,0,1]
		integer to determine canonical form of input state: 
			* canonical=-1: left canonical (Tensors=[A[0],A[1],...,A[L-1]]), psi = A[0] @ A[1] ... A[L-1]
			* canonical= 0: canonical, (Tensors=[Gamma[0],Gamma[1],...,Gamma[L-1]]), psi = Lambda[0] @ Gamma[0] @ Lambda[1] @ Gamma[1] ... @ Lambda[L-1] @ Gamma[L-1] @ Lambda[L]
			* canonical=+1: right canonical (Tensors=[B[0],B[1],...,B[L-1]]), psi = B[0] @ B[1] ... B[L-1]

	Returns
	-------
	norm: float64
		norm of MPS state

	"""

	norm=1.0

	if canonical==0: # canonical form
		for j in range(len(Tensors)-1):
			Theta = np.einsum('a,aib,b->aib',Lambdas[j],Tensors[j],Lambdas[j+1],)
			norm*=np.sum(np.abs(Theta)**2)
	elif canonical==+1: # right canonical form
		Theta=np.eye(2)/2
		for j in range(len(Tensors)):
			Theta = np.einsum('ab,aic,bid->cd', Theta, Tensors[j].conj(), Tensors[j] )
		norm=Theta.squeeze()
	else: # left canonical form
		Theta=np.eye(2)/2
		for j in range(len(Tensors)-1,-1,-1):	
			Theta = np.einsum('aib,cid,bd->ac', Tensors[j].conj(), Tensors[j], Theta )
		norm=Theta.squeeze()

	return np.sqrt(norm)

def compute_Sent(Lambdas,bond):
	"""
	Computes the entanglement across a bond in the chain, from eigenvalues of the reduced density matrix Lambdas[bond].

	Parameters
	----------
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)
	bond: int
		bond in spin chain across which to compute the bipartite entanglement 

	Returns
	-------
	Sent, float64
		entanglement entropy across bond `bond`.

	"""
	
	Sent = -Lambdas[bond]**2 @ np.log(Lambdas[bond]**2)
	
	return Sent


def generate_random_MPS(L,d=2,chi_max=None):
	"""
	Constructs random state in MPS representation in left and right canonical form 
	(most likely this is Haar-random, need to check).

	
	Parameters
	----------
	L: int
		number of spins/qubits/lattice sites
	d: int
		on-site Hilbert space dimension (d=2 for spins)
	chi_max: int / None
		integer to set maximal bond dimensiona cross chain; must be a power of 2 and chi_max < 2**(L//2)


	Returns
	-------
	As: list[np.array]
		list of A tensors, one for each lattice site
	Bs: list[np.array]
		list of B tensors, one for each lattice site
	Lambdas: list[np.array]
		list of Lambda tensors, one for each bond (including leftmost and rightmost fictitious bonds)
	
	"""

	# compute MPS representation for fixed bond dimension chi
	if chi_max is None:
		chi_max==2**(L//2)

	# compute bond dimensions
	chi_vec = [min(chi_max, 2**(min(j,L-j)) ) for j in range(L+1)]


	# initialize data
	As=[]
	Bs=[]
	Lambdas=[np.array([1.0])]

	# for each lattice site j
	for j in range(L):

		# draw normal randm matrix
		M=np.random.normal(size=(chi_vec[j]*d,chi_vec[j+1]) )
		# do SVD and reshape
		X, S, Y = np.linalg.svd(M, full_matrices=False)
		X=X.reshape(chi_vec[j],d,chi_vec[j+1])

		# identify A, B ,and Lambda (truncated to bond requested dimension)
		As.append( np.einsum('a,aib->aib',Lambdas[j],X) )
		Lambdas.append( S[:chi_vec[j+1]]/np.sqrt(np.sum(np.abs(S[:chi_vec[j+1]])**2)) )
		Bs.append( np.einsum('aib,b->aib',X,Lambdas[j+1]) )

	# normalize MPS states
	norm_A=compute_MPS_norm(As,Lambdas,canonical=-1)
	norm_B=compute_MPS_norm(Bs,Lambdas,canonical=1)
	for j in range(L):
		As[j]/=norm_A**(1.0/L)
		Bs[j]/=norm_B**(1.0/L)

	return As, Bs, Lambdas





