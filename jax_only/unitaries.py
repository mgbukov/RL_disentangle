from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general 


def create_unitaries(L,):

	basis = spin_basis_general(L)

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

	return H_xx,H_xy,H_xz, H_yx,H_yy,H_yz, H_zx,H_zy,H_zz


