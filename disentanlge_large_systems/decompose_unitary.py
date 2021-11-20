import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis

import numpy as np # generic math functions
from scipy.sparse.linalg import expm
from scipy.linalg import logm

#from scipy.sparse.linalg import expm
#from scipy.linalg import logm, qr
from itertools import combinations
from scipy.stats import unitary_group
from scipy.optimize import minimize

from aux_funcs import *

np.set_printoptions(suppress=True, precision=4)

seed=5
np.random.seed(seed)



L=2 # two qubits
basis=spin_basis_general(L)



# define single-particle gate generators
no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

# acting only on qubit 0
qubit_0=[[1.0,0]]
H_xI=hamiltonian([['x',qubit_0],],[],basis=basis,**no_checks).toarray()
H_yI=hamiltonian([['y',qubit_0],],[],basis=basis,**no_checks).toarray()
H_zI=hamiltonian([['z',qubit_0],],[],basis=basis,**no_checks).toarray()

# acting only on qubit 1
qubit_1=[[1.0,1]]
H_Ix=hamiltonian([['x',qubit_1],],[],basis=basis,**no_checks).toarray()
H_Iy=hamiltonian([['y',qubit_1],],[],basis=basis,**no_checks).toarray()
H_Iz=hamiltonian([['z',qubit_1],],[],basis=basis,**no_checks).toarray()

# define two-qubit/entangling gates
qubit_01=[[1.0,0,1]]
H_xx=hamiltonian([['xx',qubit_01],],[],basis=basis,**no_checks).toarray()
H_xy=hamiltonian([['xy',qubit_01],],[],basis=basis,**no_checks).toarray()
H_xz=hamiltonian([['xz',qubit_01],],[],basis=basis,**no_checks).toarray()

H_yx=hamiltonian([['yx',qubit_01],],[],basis=basis,**no_checks).toarray()
H_yy=hamiltonian([['yy',qubit_01],],[],basis=basis,**no_checks).toarray()
H_yz=hamiltonian([['yz',qubit_01],],[],basis=basis,**no_checks).toarray()

H_zx=hamiltonian([['zx',qubit_01],],[],basis=basis,**no_checks).toarray()
H_zy=hamiltonian([['zy',qubit_01],],[],basis=basis,**no_checks).toarray()
H_zz=hamiltonian([['zz',qubit_01],],[],basis=basis,**no_checks).toarray()




def polar(theta,phi):
	return np.array( [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])



def compute_U(angles):

	n1 = np.einsum('i,ijk->jk',  polar(angles[1],angles[2]),   np.array([H_xI,H_yI,H_zI]) )
	n2 = np.einsum('i,ijk->jk',  polar(angles[4],angles[5]),   np.array([H_Ix,H_Iy,H_Iz]) ) 
	n3 = np.einsum('i,ijk->jk',  polar(angles[7],angles[8]),   np.array([H_xI,H_yI,H_zI]) ) 
	n4 = np.einsum('i,ijk->jk',  polar(angles[10],angles[11]), np.array([H_Ix,H_Iy,H_Iz]) ) 


	A1 = (np.cos(angles[0])*np.eye(4) - 1j*np.sin(angles[0])*n1)  
	A2 = (np.cos(angles[3])*np.eye(4) - 1j*np.sin(angles[3])*n2)

	B1 = (np.cos(angles[6])*np.eye(4) - 1j*np.sin(angles[6])*n3) 
	B2 = (np.cos(angles[9])*np.eye(4) - 1j*np.sin(angles[9])*n4) 

	Uxx = np.cos(angles[12])*np.eye(4) - 1j*np.sin(angles[12])*H_xx
	Uyy = np.cos(angles[13])*np.eye(4) - 1j*np.sin(angles[13])*H_yy
	Uzz = np.cos(angles[14])*np.eye(4) - 1j*np.sin(angles[14])*H_zz 

	return np.exp(-1j*angles[15]) * ( (A1 @ A2) @ (Uxx @ Uyy @ Uzz) @ (B1 @ B2) )


# def compute_U(angles):

# 	Uxx = np.cos(angles[0])*np.eye(4) - 1j*np.sin(angles[0])*H_xx
# 	Uxy = np.cos(angles[1])*np.eye(4) - 1j*np.sin(angles[1])*H_xy
# 	Uxz = np.cos(angles[2])*np.eye(4) - 1j*np.sin(angles[2])*H_xz
	
# 	Uyx = np.cos(angles[3])*np.eye(4) - 1j*np.sin(angles[3])*H_yx
# 	Uyy = np.cos(angles[4])*np.eye(4) - 1j*np.sin(angles[4])*H_yy
# 	Uyz = np.cos(angles[5])*np.eye(4) - 1j*np.sin(angles[5])*H_yz
	
# 	Uzx = np.cos(angles[6])*np.eye(4) - 1j*np.sin(angles[6])*H_zx
# 	Uzy = np.cos(angles[7])*np.eye(4) - 1j*np.sin(angles[7])*H_zy 
# 	Uzz = np.cos(angles[8])*np.eye(4) - 1j*np.sin(angles[8])*H_zz

# 	return np.exp(-1j*angles[9]) * (Uxx @ Uxy @ Uxz) @ (Uyx @ Uyy @ Uyz) @ (Uzx @ Uzy @ Uzz)


def find_unitary(angles, U):
	U_varl = compute_U(angles)
	return np.linalg.norm(U_varl - U)
	#return np.linalg.norm(U_varl.T.conj()@U - np.eye(4))


U = unitary_group.rvs(4)

angles_0=1.0*np.pi*np.random.uniform(low=-1.0, high=1.0,size=16) # initial condition for solver


res = minimize(find_unitary, angles_0, args=(U,), method='Nelder-Mead', options=dict(maxiter=100000), tol=1e-12)

print(res.fun, res.success)

angles_opt = res.x

U_decomp = compute_U(angles_opt)

print(U - U_decomp)

print(angles_opt)






