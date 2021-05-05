import numpy as np 
import scipy as sp
import scipy.optimize as sp_opt

import jax.numpy as jnp 
import jax.scipy as jsp
import jax.scipy.optimize as jsp_opt


import time


H = [[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
	 [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j],
	 [ 0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j],
	 [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]]

H2 = [[ 1.+0.j,  2.+0.j,  0.+0.j,  2.+0.j],
	 [ 2.+0.j,  3.+0.j,  -1.+0.j,  0.+0.j],
	 [ 0.+0.j,  -1.+0.j,  3.+0.j,  3.+0.j],
	 [ 2.+0.j,  0.+0.j,  3.+0.j,  -1.+0.j]]

psi = [0.45220585, 0.58929456, 0.49665892, 0.44896738]


H_jnp, psi_jnp, H_jnp2 = jnp.array(H, dtype=jnp.complex64), jnp.array(psi, dtype=jnp.complex64), jnp.array(H2, dtype=jnp.complex64)
H_np, psi_np, H_np2   =  np.array(H, dtype=np.complex64) ,  np.array(psi, dtype=np.complex64),  np.array(H2, dtype=np.complex64)



#@jit
def compute_energy_jnp(angle,  H,psi,H2):
	U=jsp.linalg.expm(-1j*angle*H)
	psi_new = U @ psi
	return (psi_new.conj() @ H2 @ psi_new).real


def compute_energy_np(angle,  H,psi,H2):
	U=sp.linalg.expm(-1j*angle*H)
	psi_new = U @ psi
	return (psi_new.conj() @ H2 @ psi_new).real



# initial solver condition
angle_0 = np.pi/np.exp(1.0)
angle_0_jnp, angle_0_np = jnp.array([angle_0,], dtype=jnp.float32), np.array([angle_0,], dtype=np.float32)


# scipy optimization

t_i = time.time()
res = sp_opt.minimize(compute_energy_np, angle_0_np, args=(H_np,psi_np,H_np2),  method='BFGS', tol=1e-3)
t_f = time.time()

print(res.x, res.fun, res.success, res.nfev, res.njev)
print('scipy calculation took {:0.6f} secs.\n'.format(t_f-t_i))


# jax.scipy optimization

compute_energy_jnp(angle_0_jnp, H_jnp, psi_jnp) #, H_jnp2), psi_np.conj() @ H_np2 @ psi_np


t_i = time.time()
res = jsp_opt.minimize(compute_energy_jnp, angle_0_jnp, args=(H_jnp,psi_jnp,H_jnp2),  method='BFGS', tol=1e-3)
t_f = time.time()

print(res.x, res.fun, res.success, res.nfev, res.njev)
print('JAX calculation took {:0.6f} secs.'.format(t_f-t_i))
