import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

#from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis

import numpy as np # generic math functions
#from scipy.sparse.linalg import expm
#from scipy.linalg import logm, qr
from itertools import combinations
#from scipy.stats import unitary_group

from aux_funcs import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)


###########

# fix seed of RNG for reproducibility
seed=0
np.random.seed(seed)

np.set_printoptions(precision=6,suppress=True,) # print two decimals, suppress scientific notation

########################################



def disentangle(L,traj,psi):

	subsystems = list(combinations(range(L),2))

	basis=spin_basis_general(L)
	basis_red=spin_basis_general(2)

	M=traj.shape[0]


	Smin=np.zeros(M)

	for i in range(M):

		subsys = tuple(traj[i])

		rdm = compute_rdm(psi,L,subsys)
		lmbdas,U = np.linalg.eigh(rdm)

		psi = apply_2q_unitary(psi,U.conj().T,subsys,L)

		entropy = basis.ent_entropy(psi,density=True)
		Sent_half_chain=ent_entropy(psi,L,'half')
		#Sent_half_chain=compute_Sent(lmbdas)
		Smin[i]=Sent_half_chain
	
		#print(subsys, Sent_half_chain)
		#exit()


		#print(i)


	Sall=[]
	for _i, subsys in enumerate(subsystems):

		entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
		Sall.append( entropy['Sent_A'] )


	print("\nmin_ent={0:0.6f}, mean_ent={1:0.6f}, max_ent={2:0.6f}\n".format(np.min(Sall), np.mean(Sall), np.max(Sall)) )


	return i, Smin[:i], psi, traj

#exit()



# L=8 # two qubits
# M=200 # number of random disentangling gates

# subsystems = np.array(list(combinations(range(L),2)))
# inds = np.random.randint(0,subsystems.shape[0],size=M)
# traj = subsystems[inds]


# psi=np.random.normal(size=(2**L,)) + 1j*np.random.normal(size=(2**L,))
# psi/=np.linalg.norm(psi)

# i, Smin, psi, traj = disentangle(L,traj,psi,)

# plt.plot(np.arange(i), Smin)
# plt.show()


