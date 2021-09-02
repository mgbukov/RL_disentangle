import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis
import numpy as np # generic math functions
from scipy.sparse.linalg import expm
from scipy.linalg import logm

# fix seed of RNG for reproducibility
seed=9
np.random.seed(seed)

np.set_printoptions(precision=6,suppress=True,) # print two decimals, suppress scientific notation

########################################

L=2 # two qubits

basis=spin_basis_general(L)

print(basis)



# define random two-qubit state
psi=np.random.uniform(size=basis.Ns) + 1j*np.random.uniform(size=basis.Ns)
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

# acting only on qubit 1
qubit_1=[[1.0,1]]
H_Ix=hamiltonian([['x',qubit_1],],[],basis=basis,**no_checks)
H_Iy=hamiltonian([['y',qubit_1],],[],basis=basis,**no_checks)
H_Iz=hamiltonian([['z',qubit_1],],[],basis=basis,**no_checks)

# define two-qubit/entangling gates
qubit_01=[[1.0,0,1]]
H_xx=hamiltonian([['xx',qubit_01],],[],basis=basis,**no_checks)
H_xy=hamiltonian([['xy',qubit_01],],[],basis=basis,**no_checks)
H_xz=hamiltonian([['xz',qubit_01],],[],basis=basis,**no_checks)

H_yx=hamiltonian([['yx',qubit_01],],[],basis=basis,**no_checks)
H_yy=hamiltonian([['yy',qubit_01],],[],basis=basis,**no_checks)
H_yz=hamiltonian([['yz',qubit_01],],[],basis=basis,**no_checks)

H_zx=hamiltonian([['zx',qubit_01],],[],basis=basis,**no_checks)
H_zy=hamiltonian([['zy',qubit_01],],[],basis=basis,**no_checks)
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


U_xz=expm(-1j*angle*H_xz.toarray())

# print()
# print(U_xI)
# print()
# print(U_Ix)
# print()
# print(U_xx)
# print()

# print(U_zz)
# print()


# apply gate on the quantum state
psi_new = U_Iy.dot(psi)
#psi_new=psi
#psi_new = U_xx.dot(psi)


bell_1 = np.array([0.0,1.0,+1.0,0.0])/np.sqrt(2)
bell_2 = np.array([0.0,1.0,-1.0,0.0])/np.sqrt(2)
bell_3 = np.array([1.0,0.0,0.0,+1.0])/np.sqrt(2)
bell_4 = np.array([1.0,0.0,0.0,-1.0])/np.sqrt(2)

#psi_new = np.array([0.0,1.0,-1.0,0.0])/np.sqrt(2)
psi_new = np.array([1.0,0.0,0.0,1.0])/np.sqrt(2)

rho_new = np.outer(psi_new, psi_new.T.conj())

# rho_new = 0.25 * (np.outer(bell_1, bell_1.T.conj()) + \
# 				  np.outer(bell_2, bell_2.T.conj()) + \
# 				  np.outer(bell_3, bell_3.T.conj()) + \
# 				  np.outer(bell_4, bell_4.T.conj())
# 				 )


rho_new = 0.1*np.outer(bell_1, bell_1.T.conj()) + 1*np.outer(bell_2, bell_2.T.conj()) + 1*np.outer(bell_3, bell_3.T.conj()) + 0.2*np.outer(bell_4, bell_4.T.conj()) 

rho_new /= np.trace(rho_new)


E = np.linalg.eigvalsh(rho_new)

print(E)
#exit()

# print( rho_new )
# exit()
# print()

#rho_rdm = rho_new.reshape(2,8) @ rho_new.reshape(2,8).T.conj()

#print(rho_rdm)

#print(basis.ent_entropy(psi_new,sub_sys_A=[0,],density=True,return_rdm='A')['rdm_A'])


#rdm_ana = np.array([[rho_new[0,0]+rho_new[1,1], rho_new[0,2]+rho_new[1,3] ],[(rho_new[0,2]+rho_new[1,3]).conj(), 1 - (rho_new[0,0]+rho_new[1,1])]])

#print(rdm_ana)
#exit()

#f=4*np.linalg.det(rdm_ana).real

#S_2 = -np.log(1.0 - 0.5*f)

#print(S_2)



# compute enranglement of psi
Sent = basis.ent_entropy(psi_new,sub_sys_A=[0,],density=True, alpha=2)['Sent_A']
print('before opt:',Sent)

#exit()

alpha_opt = lambda rho: 0.25*(np.angle(rho[0,2]) - np.angle(rho[1,3]) )

#print('ana opt angle', alpha_opt)


#rdm_ana_alpha = np.array([[rho_new[0,0]+rho_new[1,1], rho_new[0,2]*np.exp(-2j*alpha_opt)+rho_new[1,3]*np.exp(+2j*alpha_opt) ],[(rho_new[0,2]*np.exp(-2j*alpha_opt)+rho_new[1,3]*np.exp(+2j*alpha_opt)).conj(), 1 - (rho_new[0,0]+rho_new[1,1])]])

#print(rdm_ana_alpha)

#f_alpha= 4*np.linalg.det(rdm_ana_alpha).real


#S_2_alpha = -np.log(1.0 - 0.5*f_alpha)

#print('S2_alpha', S_2_alpha)



#exit()

# given a gate, find angle which minimizes entanglement

from scipy.optimize import minimize

def compute_Sent(angle,  H,psi):
	U=expm(-1j*angle*H.toarray())
	psi_new=U.dot(psi)
	Sent = basis.ent_entropy(psi_new,sub_sys_A=[0,],density=True,alpha=1.0)['Sent_A']
	return Sent

def compute_Sent_rho(angle,  H,rho):
	U=expm(-1j*angle*H.toarray())
	rho_new=U @ rho @ U.conj().T
	Sent = basis.ent_entropy(rho_new,sub_sys_A=[0,],enforce_pure=False,density=True,alpha=1.0)['Sent_A']
	return Sent



angle_0=np.pi/np.exp(1) # initial condition for solver
res = minimize(compute_Sent, angle_0, args=(H_zz,psi_new), method='Nelder-Mead', tol=1e-12)

print(res.x[0], res.fun, res.success)

print('after opt:', res.fun)
print('optimal angle:', res.x[0], alpha_opt(rho_new) )




print('\n\n\n')




U_rot_yy = expm( +1j * np.pi/4 * (H_yI + H_Iy).toarray() )

U_rot_xx = expm( +1j * np.pi/4 * (H_xI + H_Ix).toarray() )

U_rot_yI = expm( +1j * np.pi/4 * (H_yI).toarray() )

# print(U_rot_y)


# print(U_xx)

# print(U_rot_y @ U_zz @ U_rot_y.conj().T)


# print('optimal angle:', alpha_opt(U_rot_yy @ rho_new @ U_rot_yy.conj().T), 
# 						alpha_opt(U_rot_yy @ rho_new @ U_rot_yy.conj().T) + np.pi/2,
# 						minimize(compute_Sent, angle_0, args=(H_xx,psi_new), method='Nelder-Mead', tol=1e-12).x[0],
# 						minimize(compute_Sent, angle_0, args=(H_xx,psi_new), method='Nelder-Mead', tol=1e-12).fun
# 						 )



# print('optimal angle:', alpha_opt(U_rot_xx @ rho_new @ U_rot_xx.conj().T), 
# 						alpha_opt(U_rot_xx @ rho_new @ U_rot_xx.conj().T) + np.pi/2,
# 						minimize(compute_Sent, angle_0, args=(H_yy,psi_new), method='Nelder-Mead', tol=1e-12).x[0],
# 						minimize(compute_Sent, angle_0, args=(H_yy,psi_new), method='Nelder-Mead', tol=1e-12).fun
# 						 )


# print('optimal angle:', alpha_opt(U_rot_yI @ rho_new @ U_rot_yI.conj().T), 
# 						alpha_opt(U_rot_yI @ rho_new @ U_rot_yI.conj().T) + np.pi/2,
# 						minimize(compute_Sent, angle_0, args=(H_xz,psi_new), method='Nelder-Mead', tol=1e-12).x[0],
# 						minimize(compute_Sent, angle_0, args=(H_xz,psi_new), method='Nelder-Mead', tol=1e-12).fun, 
						
# 	 )





print('Sent (2qubit):', -np.trace( rho_new @ logm(rho_new)).real  )

print('optimal Sent (1qubit):', '\n',
					'xx:',	minimize(compute_Sent_rho, angle_0, args=(H_xx,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'xy:',	minimize(compute_Sent_rho, angle_0, args=(H_xy,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'xz:',	minimize(compute_Sent_rho, angle_0, args=(H_xz,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'yx:',	minimize(compute_Sent_rho, angle_0, args=(H_yx,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'yy:',	minimize(compute_Sent_rho, angle_0, args=(H_yy,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'yz:',	minimize(compute_Sent_rho, angle_0, args=(H_yz,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'zx:',	minimize(compute_Sent_rho, angle_0, args=(H_zx,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'zy:',	minimize(compute_Sent_rho, angle_0, args=(H_zy,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'zz:',	minimize(compute_Sent_rho, angle_0, args=(H_zz,rho_new), method='Nelder-Mead', tol=1e-12).fun, '\n',
	 )

alpha_min = minimize(compute_Sent_rho, angle_0, args=(H_xy,rho_new), method='Nelder-Mead', tol=1e-12).x[0]
U_min = expm(-1j*alpha_min*H_xy.toarray())

rho_new_2 = U_min @ rho_new @ U_min.conj().T

print(rho_new_2)


print('optimal Sent (1qubit):', '\n',
					'xx:',	minimize(compute_Sent_rho, angle_0, args=(H_xx,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'xy:',	minimize(compute_Sent_rho, angle_0, args=(H_xy,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'xz:',	minimize(compute_Sent_rho, angle_0, args=(H_xz,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'yx:',	minimize(compute_Sent_rho, angle_0, args=(H_yx,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'yy:',	minimize(compute_Sent_rho, angle_0, args=(H_yy,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'yz:',	minimize(compute_Sent_rho, angle_0, args=(H_yz,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'zx:',	minimize(compute_Sent_rho, angle_0, args=(H_zx,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'zy:',	minimize(compute_Sent_rho, angle_0, args=(H_zy,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
					'zz:',	minimize(compute_Sent_rho, angle_0, args=(H_zz,rho_new_2), method='Nelder-Mead', tol=1e-12).fun, '\n',
	 )











