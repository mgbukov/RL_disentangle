from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.Floquet import Floquet, Floquet_t_vec # Floquet Hamiltonian
import numpy as np # generic math functions
import matplotlib.pyplot as plt

np.random.seed(0)

#
##### define model parameters #####
L=14 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
Omega=4.5 # drive frequency
#
##### set up alternating Hamiltonians #####
# define time-reversal symmetric periodic step drive
def drive(t,Omega):
	return np.sign(np.sin(Omega*t))
drive_args=[Omega,]
# compute basis in the 0-total momentum and +1-parity sector
basis=spin_basis_1d(L=L,a=1,)
# define PBC site-coupling lists for operators
x_field_pos=[[+g,i]	for i in range(L)]
x_field_neg=[[-g,i]	for i in range(L)]
z_field=[[h*np.random.uniform(),i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
dynamic=[["zz",J_nn,drive,drive_args],
		 ["z",z_field,drive,drive_args],["x",x_field_neg,drive,drive_args]]
# compute Hamiltonian
H=0.5*hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
##### define time vector of stroboscopic times with 1 driving cycles and 10 points per cycle #####
times=Floquet_t_vec(Omega,5,len_T=10) # t.vals=times, t.i=initial time, t.T=drive period


psi_i=np.zeros(basis.Ns)
psi_i[-1]=1.0


psi_t=H.evolve(psi_i,times.i,times.vals,iterate=True)


Sent=np.zeros(times.len)

for j,psi in enumerate(psi_t):

	for k in range(L):
		ent=basis.ent_entropy(psi,sub_sys_A=[k],density=True, return_rdm_EVs=True)
		Sent[j]+=ent['Sent_A']
		print(ent['p_A'])

	Sent[j]/=L

	print("\n{}  --------------\n".format(times[j]))

plt.plot(times.vals, Sent)

for j in range(times.N):
	plt.axvline(x=times.strobo[j])

plt.axhline(y=np.log(2))

plt.show()
