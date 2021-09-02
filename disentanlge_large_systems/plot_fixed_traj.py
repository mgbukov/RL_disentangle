import sys,os
# line 4 and line 5 below are for development purposes and can be removed
sys.path.append(os.path.expanduser("~") + '/quspin/QuSpin_dev/')

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general, spin_basis_1d # Hilbert space spin basis
import numpy as np 
from mps_lib import *
from disentangle import *
from generate_states import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)


np.set_printoptions(suppress=True, precision=4)
seed=0
np.random.seed(seed)


###############################

save=True
figs_dir='./data/figs/'


L=6


chi_max=4
psi, psi_trunc = trucated_random_MPS_state(L,seed,chi_max=chi_max,)

# basis=spin_basis_general(L,pauli=True)
# print(basis.ent_entropy(psi,)['Sent_A'])
# print(basis.ent_entropy(psi_trunc,)['Sent_A'])


#####################


# number of repetitions of trajectory
N_reps = 4 #10	


for prss in ['fixed-allcomb-traj', 'fixed-nnbond-traj',  'fixed-nbond-traj']:


	if prss=='fixed-nbond-traj':

		### simplest nearest-bond trajectory
		traj =  [(j,(j+1)%L) for j in range(0,L-1,1)]

	elif prss=='fixed-nnbond-traj':

		## nearest- + nn-bond trajectory
		traj=[]
		for j in range(0,L-2,1):
			traj.append( (j,j+1) )
			traj.append( (j,(j+2)%L) )
		traj.append((L-2,L-1))

	elif prss=='fixed-allcomb-traj':

		## all tow-body terms trajectory
		traj = list(combinations(range(L),2))



	full_traj = N_reps*traj



	i, Smin, psi, full_traj = disentangle(L,np.array(full_traj),psi_trunc)
	t_disent=np.arange(i)



	##############################################


	for _i in range(N_reps):
		t = _i*len(traj)
		plt.vlines(t, 0.0, np.log(2), colors='k', linewidth=1.0, linestyle='--', )


	plt.tick_params(labelsize=16)
	plt.plot(t_disent, Smin, '-x', label='fixed traj: ' + prss )
	plt.legend(fontsize=14,)
	plt.xlabel('$M$',fontsize=18)
	#ylabel_str='$S_\\mathrm{ent}^{[0]}$'
	ylabel_str='$S_\\mathrm{ent}^{[L/2]}$'
	plt.ylabel(ylabel_str,fontsize=18)
	# plt.yscale('log')
	# plt.xscale('log')
	plt.grid()
	plt.tight_layout()
	if save:
	    fig_name=prss+'_Sent_decay-L={0:d}_chimax={1:d}_Nreps={2:d}_seed={3:d}.pdf'.format(L,chi_max,N_reps,seed)
	    plt.savefig(figs_dir+fig_name)
	else:
	    plt.show()
	plt.clf()






