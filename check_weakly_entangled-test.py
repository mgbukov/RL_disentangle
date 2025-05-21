import numpy as np

from src.stategen import sample_haar_generalized
from src.util import sse

np.set_printoptions(suppress=True,precision=3)

L=12

psi = sample_haar_generalized(L, 2, 2, 2.5, 2.5, False)
#psi = sample_haar_generalized(L, 3, 3, 0.5, 0.5, False)

for i in range(1, L):
    A = tuple(range(i))
    print("Subsystem:", A)
    ent = sse(psi.reshape((2,) * L), A, batched=False)
    print(f"Entanglement: {ent:.3f}")


