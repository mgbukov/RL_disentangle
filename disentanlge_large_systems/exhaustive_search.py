import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general, spin_basis_1d # Hilbert space spin basis
import numpy as np 
from mps_lib import *
from disentangle import *
from generate_states import *

from itertools import combinations, combinations_with_replacement, permutations, product

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)


np.set_printoptions(suppress=True, precision=4)
seed=0
np.random.seed(seed)


###############################

save=False
figs_dir='./data/figs/'


L=4


chi_max=4
psi, psi_trunc = trucated_random_MPS_state(L,seed,chi_max=chi_max,)


psi  = np.random.normal(size=2**L) + 1j*np.random.normal(size=2**L)
psi /= np.linalg.norm(psi)

traj = list(combinations(range(L),2))


M=5
Smin=np.zeros( (L*(L-1)//2)**M )

all_traj=list(product(traj,repeat=M))

print((L*(L-1)//2)**M)



for j, t in enumerate(all_traj):

	_, S, _, _ = disentangle(L,np.array(t),psi)

	Smin[j]=S[-1]

	print(j, S[-1] )
	


best_ind = np.where(Smin==Smin.min() )[0]

print('\n\n')

for ind in best_ind:
	best_traj=all_traj[ind]
	print(best_traj)


print(Smin.min(), Smin.max() )
print(Smin[best_ind])


_, Sent, _, _ = disentangle(L,np.array(best_traj),psi)


print(Sent)

