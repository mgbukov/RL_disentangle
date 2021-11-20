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

def chipoffsite_disentangle(L,M,psi,site=0,eps=1E-14):

	all_subsystems = list(combinations(range(L),2))
	subsystems = [(site,j) for j in range(L) if j!=site]

	basis=spin_basis_general(L)
	basis_red=spin_basis_general(2)



	Smin=np.zeros(M*len(subsystems))
	traj = np.zeros((M*len(subsystems),2),dtype=np.int8)


	for i, subsys in enumerate(M*subsystems):

		# entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=False)
		# Sent_A = basis_red.ent_entropy(entropy['rdm_A'], enforce_pure=False, return_rdm='A', return_rdm_EVs=True)
		# print("trial={0:d}, subsys={1:}, 2rdm_e'vals={2:}, Smin={3:0.6f}, Srdm={4:0.6f}".format( i, subsys, entropy['p_A'], compute_Sent(entropy['p_A']), Sent_A['Sent_A'], )  )
		# rdm = entropy['rdm_A']


		rdm = compute_rdm(psi,L,subsys)
		_,U = np.linalg.eigh(rdm)

		# entropy = basis.ent_entropy(psi,density=True)
		# Sent_cutoff=entropy['Sent_A']
		Sent_cutoff=ent_entropy(psi,L,'site',)
		Smin[i]=Sent_cutoff
		traj[i,...]=subsys


		psi = apply_2q_unitary(psi,U.conj().T,subsys,L)


		if Sent_cutoff<eps:
			break

		print(i)


	Sall=[]
	for _i, subsys in enumerate(all_subsystems):

		entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
		Sall.append( compute_Sent(entropy['p_A']) )

	print("\nmin_ent={0:0.6f}, mean_ent={1:0.6f}, max_ent={2:0.6f}\n".format(np.min(Sall), np.mean(Sall), np.max(Sall)) )

	return i, Smin[:i], psi, traj


# L=8 # two qubits
# M=200 # number of random disentangling gates
# site=0

# psi=np.random.normal(size=(2**L,)) + 1j*np.random.normal(size=(2**L,))
# psi/=np.linalg.norm(psi)

# i_f_min, Smin, psi, traj = chipoffsite_disentangle(L,M,psi, site, eps=1E-4)

# plt.plot(range(i_f_min), Smin)
# plt.show()

# exit()


"""


L=6 # two qubits
M=10 # number of random disentangling gates

all_subsystems = list(combinations(range(L),2))
subsystems = [(0,j) for j in range(L) if j!=0]

basis=spin_basis_general(L)
basis_red=spin_basis_general(2)


################


psi=np.random.normal(size=(basis.Ns,)) + 1j*np.random.normal(size=(basis.Ns,))
psi/=np.linalg.norm(psi)


Smin=[]

for i, subsys in enumerate(M*subsystems):
	
	entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)

	print("trial={0:d}, subsys={1:}, 2rdm_e'vals={2:}, Smin={3:0.6f}".format( m, subsys, entropy['p_A'], compute_Sent(entropy['p_A']) )  )
	
	Smin.append( compute_Sent(entropy['p_A']) )


	rdm = entropy['rdm_A']
	_,U = np.linalg.eigh(rdm)
	
	psi = apply_2q_unitary(psi,U.conj().T,subsys,L)

	


Sall=[]
for _, subsys in enumerate(all_subsystems):

	entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
	Sall.append( compute_Sent(entropy['p_A']) )

print("\nmin_ent={0:0.6f}, mean_ent={1:0.6f}, max_ent={2:0.6f}\n".format(np.min(Sall), np.mean(Sall), np.max(Sall)) )

print("single-site Sent={0:0.6f}.\n".format(basis.ent_entropy(psi,sub_sys_A = [0], return_rdm='A', return_rdm_EVs=True)['Sent_A'])  )

exit()





################

psi=np.random.normal(size=(basis.Ns,)) + 1j*np.random.normal(size=(basis.Ns,))
psi/=np.linalg.norm(psi)


Sent=[]

for m in range(M):

	Smin=[]
	Us=[]

	for _i, subsys in enumerate(subsystems):

		entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
		
		rdm = entropy['rdm_A']
		_,U = np.linalg.eigh(rdm)
		Us.append(U)

		Smin.append( compute_Sent(entropy['p_A']) )


	ind = np.argmin(Smin)
	U = Us[ind]
	subsys = subsystems[ind]


	psi = apply_2q_unitary(psi,U.conj().T,subsys,L)

	entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
	#Sent_A = basis_red.ent_entropy(entropy['rdm_A'], enforce_pure=False, return_rdm='A', return_rdm_EVs=True)

	
	print("trial={0:d}, subsys={1:}, 2rdm_e'vals={2:}, Smin={3:0.6f}".format( m, subsys, entropy['p_A'], compute_Sent(entropy['p_A']) )  )


	Sent.append( compute_Sent(entropy['p_A']) )

Sall=[]
for _, subsys in enumerate(all_subsystems):

	entropy = basis.ent_entropy(psi,sub_sys_A = subsys, return_rdm='A', return_rdm_EVs=True)
	Sall.append( compute_Sent(entropy['p_A']) )

print("\nmin_ent={0:0.6f}, mean_ent={1:0.6f}, max_ent={2:0.6f}\n".format(np.min(Sall), np.mean(Sall), np.max(Sall)) )

print("single-site Sent={0:0.6f}.\n".format(basis.ent_entropy(psi,sub_sys_A = [0], return_rdm='A', return_rdm_EVs=True)['Sent_A'])  )
#exit()

# plt.plot(range(M), Sent)
# plt.show()


"""

