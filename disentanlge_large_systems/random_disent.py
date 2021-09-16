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



def random_disentangle(L,M,psi,eps=1E-14):

	subsystems = list(combinations(range(L),2))

	basis=spin_basis_general(L)
	basis_red=spin_basis_general(2)



	Smin=np.zeros(M)
	traj = np.zeros((M,2),dtype=np.int8)


	system=np.zeros((L,L),dtype=np.int8)
	for k in range(L):
		system[k,:]=[k,]+[l for l in range(L) if l!=k]

	S = np.zeros(L)
	for j in range(L):
		#S[j]=basis.ent_entropy(psi,sub_sys_A=[j],)['Sent_A']
		S[j]=ent_entropy_site(psi,L,system[j])


	for i in range(M):

		subsys = subsystems[np.random.choice(range(len(subsystems))) ]

		# entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=False)
		# Sent_A = basis_red.ent_entropy(entropy['rdm_A'], enforce_pure=False, return_rdm='A', return_rdm_EVs=True)
		# print("trial={0:d}, subsys={1:}, 2rdm_e'vals={2:}, Smin={3:0.6f}, Srdm={4:0.6f}".format( i, subsys, entropy['p_A'], compute_Sent(entropy['p_A']), Sent_A['Sent_A'], )  )
		# rdm = entropy['rdm_A']


		rdm = compute_rdm(psi,L,subsys)
		_,U = np.linalg.eigh(rdm)

		# entropy = basis.ent_entropy(psi,density=True)
		# Sent_cutoff=entropy['Sent_A']
		# Sent_cutoff=ent_entropy(psi,L,'half')
		# Smin[i]=Sent_cutoff
		for j in subsys:
			#S[j]=basis.ent_entropy(psi,sub_sys_A=[j],)['Sent_A']
			S[j]=ent_entropy_site(psi,L,system[j])

		Sent_cutoff=np.sum(S)/L
		Smin[i]=Sent_cutoff
		

		traj[i,...]=subsys

		psi = apply_2q_unitary(psi,U.conj().T,subsys,L)

		if Sent_cutoff<eps:
			break

		#print(i)


	Sall=[]
	for _i, subsys in enumerate(subsystems):

		entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
		Sall.append( entropy['Sent_A'] )


	print("\nmin_ent={0:0.6f}, mean_ent={1:0.6f}, max_ent={2:0.6f}\n".format(np.min(Sall), np.mean(Sall), np.max(Sall)) )


	return i, Smin[:i+1], psi, traj

#exit()



# L=8 # two qubits
# M=200 # number of random disentangling gates


# psi=np.random.normal(size=(2**L,)) + 1j*np.random.normal(size=(2**L,))
# psi/=np.linalg.norm(psi)

# i, Smin, psi, traj = random_disentangle(L,M,psi)

# plt.plot(np.arange(i), Smin)
# plt.show()




