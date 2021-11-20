from transpose_lib import swap_axes
import numpy as np 
from itertools import permutations

import time

seed=0
np.random.seed(0)

B=2**3
M=3

psi = np.random.uniform(size=(B,)+(M,M), ) + 1j* np.random.uniform(size=(B,)+(M,M), )
psi_T = np.zeros_like(psi).reshape(-1,)

# sites=list(np.arange(L))
# all_transposed_axes = np.array( list(permutations(sites, )), dtype=np.uint8 )

# random_inds = np.random.choice(np.arange(all_transposed_axes.shape[0]), size=B)
# transposed_axes = all_transposed_axes[random_inds]


swap_axes(B, M, psi.flatten(), psi_T)

psi_T=psi_T.reshape(B,M,M)



m=-1
print(psi[m])
print()
print(psi_T[m])

print()
print(np.linalg.norm(psi.transpose(0,2,1) - psi_T))






