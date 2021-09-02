import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

#from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis

import numpy as np # generic math functions
#from scipy.sparse.linalg import expm
#from scipy.linalg import logm, qr
from itertools import combinations
from scipy.stats import unitary_group

from aux_funcs import *
from random_entangle import *
from random_disent import *
from disentangle import *


import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)


###########

# fix seed of RNG for reproducibility
seed=2
np.random.seed(seed)

np.set_printoptions(precision=6,suppress=True,) # print two decimals, suppress scientific notation

#######################

L=6 # two qubits

M_ent=30 # number of random entangling gates
M_disent=30 # number of random entangling gates

psi_0 = np.zeros(2**L)
psi_0[0]=1.0

i_f_max, Smax, psi, traj_max = random_entangle(L,M_ent,psi_0)
print('last Smax', Smax[-1])
# np.random.seed(0)
#i_f_min, Smin, psi_f, traj_min = random_disentangle(L,M_disent,psi)
i_f_min, Smin, psi_f, traj_min = disentangle(L,traj_max[::-1],psi)

#Smax=np.append(Smax,Smin[0])

t_ent = np.arange(i_f_max)
t_disent = i_f_max + np.arange(i_f_min)

S_page = 2/L*( L/2*np.log(2) - 0.5 )

plt.plot(t_ent, Smax ,'r')
plt.plot(t_ent, S_page*np.ones_like(t_ent) ,'--k')
plt.plot(t_ent, np.log(2)*np.ones_like(t_ent), '.' ,color='k', markersize=0.5)
plt.plot(t_disent, Smin,'b')


plt.xlabel('$M$',fontsize=18)
#plt.ylabel('$S_\\mathrm{ent}(i,j)$',fontsize=18)
plt.ylabel('$S_\\mathrm{ent}^{L/2}$',fontsize=18)
plt.title('$L={}$'.format(L),fontsize=18)

plt.grid()
plt.tight_layout()

plt.show()
