import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis
import numpy as np # generic math functions
from scipy.sparse.linalg import expm

# fix seed of RNG for reproducibility
seed=10
np.random.seed(seed)

np.set_printoptions(precision=2,suppress=True,) # print two decimals, suppress scientific notation

########################################

L=4 # 4 qubits

basis=spin_basis_general(L)


# define random two-qubit state
psi=np.random.uniform(size=basis.Ns) + 1j*np.random.uniform(size=basis.Ns)	
psi/=np.linalg.norm(psi)



# define single-particle gate generators
no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

# acting on qubits 0, 2

qubit_02=[[1.0,0,2]]
H_xx=hamiltonian([['xx',qubit_02],],[],basis=basis,**no_checks)

angle=np.pi/3.0

U_xx = expm(-1j*angle*H_xx.toarray())

Upsi = U_xx @ psi



#########################################


basis=spin_basis_general(2)
qubit_01=[[1.0,0,1]]
H_xx_small=hamiltonian([['xx',qubit_01],],[],basis=basis,**no_checks) # (4,4)

U_xx_small = expm(-1j*angle*H_xx_small.toarray()) # (4,4)

# 2 qubits  (4,4) @ (4,)  = (4,) || psi -> (2,2), U -> (4,4) = (2,2, 2,2) U @ psi: (2,2, 2,2) @ (2,2) = (2,2) <=> 'ij kl, kl -> ij'

psi_reshaped = psi.reshape((2,)*L) # (2,2,2,2,)

U_xx_small_reshaped = U_xx_small.reshape((2,)*4) # (2,2, 2,2)

Upsi_small = np.einsum('mn ik,ijkl -> mjnl',U_xx_small_reshaped,psi_reshaped) # (2,2, 2,2) @ (2,2,2,2) = (2,2,2,2)

Upsi_2 = Upsi_small.reshape(-1)

print(Upsi - Upsi_2)







