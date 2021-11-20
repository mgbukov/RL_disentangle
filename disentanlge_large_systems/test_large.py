import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis
import numpy as np # generic math functions
from scipy.sparse.linalg import expm
from scipy.linalg import logm, qr
from itertools import combinations
from scipy.stats import unitary_group

from aux_funcs import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)


# fix seed of RNG for reproducibility
seed=0
np.random.seed(seed)

np.set_printoptions(precision=6,suppress=True,) # print two decimals, suppress scientific notation

########################################

L=4 # two qubits

basis=spin_basis_general(L)
basis_red=spin_basis_general(2)




subsystems = list(combinations(range(L),2))

#'''

N = 2 # trials

for i in range(N):

	# define random two-qubit state
	psi=np.random.normal(size=(basis.Ns,)) + 1j*np.random.normal(size=(basis.Ns,))
	#psi=np.random.uniform(size=(basis.Ns,)) + 1j*np.random.uniform(size=(basis.Ns,))
	psi/=np.linalg.norm(psi)

	#psi = unitary_group.rvs(2**L)[:,0]



	for subsys in subsystems:

		entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)

		Sent_A = basis_red.ent_entropy(entropy['rdm_A'], enforce_pure=False, return_rdm='A', return_rdm_EVs=True)

		print("trial={0:d}, subsys={1:}, 2rdm_e'vals={2:}, Smin={3:0.6f}, Srdm={4:0.6f}".format( i, subsys, entropy['p_A'], compute_Sent(entropy['p_A']), Sent_A['Sent_A'], )  )

	print('\n')

#exit()

#'''



#######################

print('\n')
print('\n')



#'''

subsys=(1,3)

psi=np.random.normal(size=(basis.Ns,)) + 1j*np.random.normal(size=(basis.Ns,))
psi/=np.linalg.norm(psi)

entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
Sent_A = basis_red.ent_entropy(entropy['rdm_A'], enforce_pure=False, return_rdm='A', return_rdm_EVs=True)

lmbda = entropy['p_A']
lmbda_swapped = lmbda.copy()
lmbda_swapped[[1,3]] = lmbda[[3,1]]

print(subsys, entropy['p_A'], compute_Sent(lmbda), Sent_A['Sent_A'],  )


rdm = entropy['rdm_A']
_,U = np.linalg.eigh(rdm)

print('\n')
print(U.conj().T @ rdm @ U )
print('\n')
psi_new = apply_2q_unitary(psi,U.conj().T,subsys,L)



entropy2 = basis.ent_entropy(psi_new,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
Sent_A2 = basis_red.ent_entropy(entropy2['rdm_A'], enforce_pure=False, return_rdm='A', return_rdm_EVs=True)

print(subsys, entropy2['p_A'], compute_Sent(entropy2['p_A']), Sent_A2['Sent_A'],  )



#'''

