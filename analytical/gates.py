import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis
import numpy as np # generic math functions
from scipy.sparse.linalg import expm

from sympy import *

# fix seed of RNG for reproducibility
seed=0
np.random.seed(seed)

np.set_printoptions(precision=2,suppress=True,) # print two decimals, suppress scientific notation

########################################

L=2 # two qubits

basis=spin_basis_general(L)

print(basis)
print('\n')


gate = [[1.0,0,1],]

no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)


XX = hamiltonian([['xx',gate],], [], basis=basis, **no_checks).toarray()
XY = hamiltonian([['xy',gate],], [], basis=basis, **no_checks).toarray()
XZ = hamiltonian([['xz',gate],], [], basis=basis, **no_checks).toarray()

YX = hamiltonian([['yx',gate],], [], basis=basis, **no_checks).toarray()
YY = hamiltonian([['yy',gate],], [], basis=basis, **no_checks).toarray()
YZ = hamiltonian([['yz',gate],], [], basis=basis, **no_checks).toarray()

ZX = hamiltonian([['zx',gate],], [], basis=basis, **no_checks).toarray()
ZY = hamiltonian([['zy',gate],], [], basis=basis, **no_checks).toarray()
ZZ = hamiltonian([['zz',gate],], [], basis=basis, **no_checks).toarray()


print(ZZ)

t = Symbol('t', real=True)


rho = Matrix([['rho_11','rho_12','rho_13','rho_14'],
			  [conjugate('rho_12'),'rho_22','rho_23','rho_24'],
			  [conjugate('rho_13'),conjugate('rho_23'),'rho_33','rho_34'],
			  [conjugate('rho_14'),conjugate('rho_24'),conjugate('rho_34'),'rho_44'],
	])

pprint(rho)



m_zz = -I * t * Matrix(ZZ) 

m_xx = -I * t * Matrix(XX) 


# print(ZZ)

U_xx = m_xx.exp().applyfunc(simplify)

U_zz = m_zz.exp().applyfunc(simplify)

pprint(U_zz)

#pprint(U_xx * rho * U_xx.T.conjugate() )
print('\n\n')
pprint(U_zz * rho * U_zz.T.conjugate() )









