import sys, os
import numpy as np 

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)



import pickle
from scipy.optimize import curve_fit

###################

def linear_func(x, a, b):
    return a * x + b

def exp_func(x, a, b):
    return np.exp(a * x + b)




###################

save=True
figs_dir='./data/figs/'

#prss = 'random_disentangle'
#prss = 'seq_disentangle'
prss = 'chipoff_disentangle'

seed=0

#Ls=np.array([6,8,10,12,14,])
Ls=np.arange(6,15,1)

popts = np.zeros((Ls.shape[0],2), ) 
pcovs = np.zeros((Ls.shape[0],2,2), ) 

t_disent_tot=0 # total time to disentangle over all system sizes

for j,L in enumerate(Ls):

    load_dir = './data/'+prss+'/'
    file_name=load_dir+prss+"_L={0:d}_seed={1:d}.pkl".format(L,seed)

    with open(file_name, 'rb') as handle:
        Smin,t_disent,psi,psi_f,L,dtype,eps,seed = pickle.load(handle)

   
    # fit data
    popt, pcov = curve_fit(linear_func, t_disent, np.log(Smin), p0=[-10.0, np.log(2)], )

    popts[j,...] = popt
    pcovs[j,...] = pcov

    t_disent_tot+=t_disent[-1]

    p=plt.plot(t_disent, Smin,label='$L={0:d}$'.format(L))
    plt.plot(t_disent, exp_func(t_disent, *popt), '--', color=p[0].get_color(), )


if prss=='chipoff_disentangle':
    ylabel_str='$S_\\mathrm{ent}^{[0]}$'
    plt.vlines(t_disent_tot, 0.0, 1.0, colors='k', linewidth=1.0, linestyle='--', label='$t_\\mathrm{tot}'+'={0:d}$'.format(t_disent_tot) )
else:
    ylabel_str='$S_\\mathrm{ent}^{[L/2]}$'

plt.legend(fontsize=14,)
plt.xlabel('$M$',fontsize=18)

plt.ylabel(ylabel_str,fontsize=18)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.tight_layout()
if save:
    fig_name=prss+'_Sent_decay.pdf'
    plt.savefig(figs_dir+fig_name)
else:
    plt.show()
plt.clf()




plt.plot(Ls,-popts[:,0],'-x', linewidth=1.0, label='$a \\exp(-c(L) M)$')
plt.xlabel('$L$',fontsize=18)
plt.ylabel('$c(L)$',fontsize=18)
plt.tick_params(labelsize=16)
plt.yscale('log')
plt.legend(fontsize=16)
plt.grid()
plt.tight_layout()
if save:
    fig_name=prss+'_decay_rate_scaling.pdf'
    plt.savefig(figs_dir+fig_name)
else:
    plt.show()
plt.close()



