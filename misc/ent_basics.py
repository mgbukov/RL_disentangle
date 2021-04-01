import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
from scipy.sparse.linalg import expm

# fix seed of RNG for reproducibility
seed=10
np.random.seed(seed)

np.set_printoptions(precision=2,suppress=True,) # print two decimals, suppress scientific notation

########################################

L=2 # two qubits

basis=spin_basis_1d(L)

print(basis)


# define random two-qubit state
psi=np.random.uniform(size=basis.Ns)
psi/=np.linalg.norm(psi)

print(psi)

# compute enranglement of psi
Sent = basis.ent_entropy(psi,sub_sys_A=[0,],density=True)['Sent_A']
print(Sent)

# define single-particle gate generators
no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

# acting only on qubit 0
qubit_0=[[1.0,0]]
H_xI=hamiltonian([['x',qubit_0],],[],basis=basis,**no_checks)
H_yI=hamiltonian([['y',qubit_0],],[],basis=basis,**no_checks)
H_zI=hamiltonian([['z',qubit_0],],[],basis=basis,**no_checks)

# acting only on qubit 0
qubit_1=[[1.0,1]]
H_Ix=hamiltonian([['x',qubit_1],],[],basis=basis,**no_checks)
H_Iy=hamiltonian([['y',qubit_1],],[],basis=basis,**no_checks)
H_Iz=hamiltonian([['z',qubit_1],],[],basis=basis,**no_checks)

# define two-qubit/entangling gates
qubit_01=[[1.0,0,1]]
H_xx=hamiltonian([['xx',qubit_01],],[],basis=basis,**no_checks)
H_yy=hamiltonian([['yy',qubit_01],],[],basis=basis,**no_checks)
H_zz=hamiltonian([['zz',qubit_01],],[],basis=basis,**no_checks)


# define gates
angle=np.pi/8.0 # rotation angle: can be different for different gates

U_xI=expm(-1j*angle*H_Ix.toarray())
U_yI=expm(-1j*angle*H_Iy.toarray())
U_zI=expm(-1j*angle*H_Iz.toarray())

U_Ix=expm(-1j*angle*H_xI.toarray())
U_Iy=expm(-1j*angle*H_yI.toarray())
U_Iz=expm(-1j*angle*H_zI.toarray())

U_xx=expm(-1j*angle*H_xx.toarray())
U_yy=expm(-1j*angle*H_yy.toarray())
U_zz=expm(-1j*angle*H_zz.toarray())

print()
print(U_xI)
print()
print(U_Ix)
print()
print(U_xx)
print()

# apply gate on the quantum state
psi_new = U_Iy.dot(psi)

# compute enranglement of psi
Sent = basis.ent_entropy(psi_new,sub_sys_A=[0,],density=True)['Sent_A']
print(Sent)



# given a gate, find angle which minimizes entanglement

from scipy.optimize import minimize

def compute_Sent(angle,H,psi):
	U=expm(-1j*angle*H.toarray())
	psi_new=U.dot(psi)
	Sent = basis.ent_entropy(psi_new,sub_sys_A=[0,],density=True)['Sent_A']
	return Sent

angle_0=np.pi/np.exp(1) # initial condition for solver
res = minimize(compute_Sent, angle_0, args=(H_xx,psi_new), method='Nelder-Mead', tol=1e-6)

print(res.x[0], res.fun, res.success)




