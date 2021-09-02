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
from seq_disent import *
from chipoff_site import *

import time
import pickle


###########

# fix seed of RNG for reproducibility
seed=0
np.random.seed(seed)

np.set_printoptions(precision=6,suppress=True,) # print two decimals, suppress scientific notation

#######################

L=13 # spins
dtype=np.complex64 # np.complex128
eps=1E-4

#prss = 'random_disentangle'
#prss = 'seq_disentangle'
prss = 'chipoff_disentangle'

M_disent=100000 # number of random entangling gates

psi=np.random.normal(size=(2**L,),) + 1j*np.random.normal(size=(2**L,), )
psi = psi.astype(dtype)
psi/=np.linalg.norm(psi)

t_i=time.time()

if prss == 'random_disentangle':
	i_f_min, Smin, psi_f, traj = random_disentangle(L,M_disent,psi, eps=eps)
elif prss == 'seq_disentangle':
	i_f_min, Smin, psi_f, traj = seq_disentangle(L,M_disent,psi, eps=eps)
elif prss == 'chipoff_disentangle':
	site=0
	i_f_min, Smin, psi_f, traj = chipoffsite_disentangle(L,M_disent,psi, site, eps=eps)

t_disent = np.arange(i_f_min)

t_f=time.time()

print('simulation took {0:0.4} secs. for {1:d} gates at L={2:d}.'.format(t_f-t_i, i_f_min, L))

########
save_dir = './data/'+prss+'/'

file_name=save_dir+prss+"_L={0:d}_seed={1:d}.pkl".format(L,seed)

with open(file_name, 'wb') as handle:
    pickle.dump([Smin,t_disent,psi,psi_f,L,dtype,eps,seed], handle, protocol=pickle.HIGHEST_PROTOCOL)


