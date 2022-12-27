import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

from quspin.operators import hamiltonian, quantum_operator # Hamiltonians and operators
from quspin.basis import spin_basis_general, spin_basis_1d # Hilbert space spin basis
import numpy as np 
from mps_lib import *



###########################################################################

np.set_printoptions(suppress=True, precision=4)
seed=0
np.random.seed(seed)


###########################################################################

def trucated_random_MPS_state(L,seed,chi_max=None,):

	# fix seed
	np.random.seed(seed)

	# generate random state
	psi = np.random.normal(size=2**L) + 1j*np.random.normal(size=2**L)
	psi/=np.linalg.norm(psi)

	# compute MPS representation for fixed bond dimension chi
	if chi_max is None:
		chi_max==2**(L//2)

	# bond dimensions
	chi_vec=[min(chi_max, 2**(min(j,L-j)) ) for j in range(L+1)]

	# canonical MPS 
	Gammas, Lambdas = ED_to_MPS(psi,chi_vec,L,d=2)

	# compute full H-space representation
	psi_trunc=MPS_to_ED(Gammas, Lambdas, )

	return psi, psi_trunc



###########################################################################


class template_model:

	def __init__(self, L,pauli):

		# define basis
		self.basis=spin_basis_general(L,pauli=pauli)

		# definte site-coupling lists
		self.single_part_list = [[1.0,j] for j in range(L)]
		self.nn_int_list      = [[1.0,j,(j+1)%L] for j in range(L)]
		self.nnn_int_list     = [[1.0,j,(j+2)%L] for j in range(L)]
		

	def _construct_param_H(self,operator_dict):

		# construct parametric Hamiltonian
		no_checks=dict(check_herm=False, check_symm=False, check_pcon=False)
		self.H=quantum_operator(operator_dict,basis=self.basis,dtype=np.float64, **no_checks)


	def compute_ground_state(self,model_params):
	
		# compute GS
		E_GS,psi_GS = self.H.eigsh(pars=model_params, k=1, which='SA')
		
		E_GS=E_GS.squeeze()
		psi_GS=psi_GS.squeeze()

		return E_GS, psi_GS



class Ising_model(template_model):

	def __init__(self, L,pauli,):

		template_model.__init__(self,L,pauli)

		# define operators
		op_list_J  = [["zz", self.nn_int_list], ]
		op_list_hz = [["z",  self.single_part_list], ]
		op_list_hx = [["x",  self.single_part_list], ]

		operator_dict=dict(Jzz=op_list_J, hz=op_list_hz, hx=op_list_hx,)

		self._construct_param_H(operator_dict)



class Heisenberg_model(template_model):

	def __init__(self, L,pauli,):

		template_model.__init__(self,L,pauli)

		# define operators
		op_list_Jzz = [["zz", self.nn_int_list], ]
		op_list_Jxx = [["xx", self.nn_int_list], ]
		op_list_Jyy = [["yy", self.nn_int_list], ]

		op_list_hz = [["z",  self.single_part_list], ]

		operator_dict=dict(Jxx=op_list_Jxx, Jyy=op_list_Jyy, Jzz=op_list_Jzz, hz=op_list_hz)

		self._construct_param_H(operator_dict)



class J1J2_model(template_model):

	def __init__(self, L,pauli,):

		template_model.__init__(self,L,pauli)

		# define operators
		op_list_nn  = [ ["zz", self.nn_int_list],  ["xx", self.nn_int_list],  ["yy", self.nn_int_list],  ]
		op_list_nnn = [ ["zz", self.nnn_int_list], ["xx", self.nnn_int_list], ["yy", self.nnn_int_list], ]

		operator_dict=dict(J1=op_list_nn, J2=op_list_nnn)

		self._construct_param_H(operator_dict)



class MBL_model(template_model):

	def __init__(self, L,pauli,):

		template_model.__init__(self,L,pauli)

		# define operators
		op_list_Jzz  = [["zz", self.nn_int_list], ]
		op_list_Jxx  = [["xx", self.nn_int_list], ]

		op_list_hz = [["z",  self.single_part_list], ]
		op_list_hx = [["x",  self.single_part_list], ]

		operator_dict=dict(Jzz=op_list_Jzz, Jxx=op_list_Jxx, hz=op_list_hz, hx=op_list_hx,)

		# disordered z-field
		for i in range(L):
			op = [[1.0,i]]
			operator_dict["hz"+str(i)] = [["z",op]]

		self._construct_param_H(operator_dict)



###################################################################################################

if __name__ == "__main__":

	L=4
	pauli=True

	#################################################

	TFIM_model_params=dict(Jzz=1.0, hx=1.0, hz=1.0, )
	TFIM = Ising_model(L,pauli,)
	print(TFIM.compute_ground_state(TFIM_model_params))

	#################################################

	Heisenberg_model_params=dict(Jxx=1.0, Jyy=1.0, Jzz=1.0, hz=1.0 )
	HM = Heisenberg_model(L,pauli,)
	print(HM.compute_ground_state(Heisenberg_model_params))

	#################################################

	J1J2_model_params=dict(J1=1.0, J2=1.0, )
	J1J2M = J1J2_model(L,pauli,)
	print(J1J2M.compute_ground_state(J1J2_model_params))

	#################################################

	W=1.0
	hz = np.random.uniform(-1,1,size=L)
	disorder_params = {"hz"+str(i):W*hz[i] for i in range(L)}

	MBL_model_params = dict(Jzz=1.0, Jxx=1.0, hx=1.0, hz=1.0, )
	MBL_model_params.update( disorder_params )
			

	MBLM = MBL_model(L,pauli,)
	print(MBLM.compute_ground_state(MBL_model_params))

	#################################################


