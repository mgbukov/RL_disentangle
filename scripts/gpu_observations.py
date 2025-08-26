import itertools
import time

import torch
import numpy as np

import context
from src.quantum_env import rdm_2q_complex
from src.stategen import sample_haar_full


GPU_DEVICE = "cuda"


def gpu_rdm_2q_complex(states: np.ndarray):
    N = states.shape[0]
    Q = len(states.shape[1:])
    qubit_pairs = itertools.permutations(range(Q), 2)

    statevector = torch.from_numpy(states).to(device=GPU_DEVICE)
    statevector.requires_grad_ = False

    rdms = torch.zeros((Q*(Q-1), N, 4, 4),
                       dtype=torch.complex64,
                       requires_grad=False,
                       device=GPU_DEVICE)

    for i, qubits in enumerate(qubit_pairs):
        sysA = tuple(q+1 for q in qubits)
        sysB = tuple(q+1 for q in range(Q) if q not in qubits)
        permutation = (0,) + sysA + sysB
        psi = torch.permute(statevector, permutation).reshape(N, 4, -1)
        rdm = psi @ torch.permute(psi, (0, 2, 1)).conj()
        rdms[i, :, :, :] = rdm

    return torch.swapaxes(rdms, 0, 1).reshape(N, Q*(Q-1), 16)




if __name__ == "__main__":

    for B in (64, 128, 256):
        states = np.array([sample_haar_full(16) for _ in range(B)])
        tic = time.time()
        cpu_obs = rdm_2q_complex(states)
        toc = time.time()
        print(f"Elapsed time on CPU for batch size {B}: {toc-tic:.3f}", flush=True)

        tic = time.time()
        gpu_obs = gpu_rdm_2q_complex(states)
        torch.cuda.synchronize()
        toc = time.time()
        print(f"Elapsed time on GPU for batch size {B}: {toc-tic:.3f}", flush=True)

        assert np.allclose(cpu_obs, gpu_obs.cpu().numpy(), atol=1e-6)
