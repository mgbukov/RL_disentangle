import itertools
import time
import timeit
from typing import *

import torch
import numpy as np

import context
from src.quantum_env import rdm_2q_complex, rdm_2q_mean_real
from src.stategen import sample_haar_full
from src.observations import rdm2m
from src.util import sqe, cuda_sqe

GPU_DEVICE = "cuda"


def bench_obs(cpu_obs_fn, cuda_obs_fn):
    for B in (1, 4, 32, 64, 128):
        states = np.array([sample_haar_full(16) for _ in range(B)])
        tic = time.time()
        cpu_obs = cpu_obs_fn(states)
        toc = time.time()
        print(f"Elapsed time on CPU for batch size {B}: {toc-tic:.3f}", flush=True)

        x = torch.from_numpy(states).to(device=GPU_DEVICE, dtype=torch.complex64)
        tic = time.time()
        gpu_obs = cuda_obs_fn(x)
        torch.cuda.synchronize()
        toc = time.time()
        print(f"Elapsed time on GPU for batch size {B}: {toc-tic:.3f}", flush=True)

        assert np.allclose(cpu_obs, gpu_obs.cpu().numpy(), atol=1e-6)


def bench_svd():
    A = np.random.randn(32, 2, 32_768) + 1j * np.random.randn(32, 2, 32_768)
    t = timeit.repeat("lmbda = np.linalg.svd(A, full_matrices=False, compute_uv=False)",
                      number=10, repeat=10, globals=globals())
    print(f"Elapsed time for np.linalg.svd(): {np.mean(t):.2f} ± {np.std(t):.2f}")

    C = torch.from_numpy(A).to(dtype=torch.complex64, device="cuda")
    t = timeit.repeat("lmbda = torch.linalg.svd(C, full_matrices=False)",
                      number=10, repeat=10, globals=globals())
    print(f"Elapsed time for torch.linalg.svd(): {np.mean(t):.2f} ± {np.std(t):.2f}")


def bench_sqe(num_envs: Iterable[int], num_qubits: Iterable[int]):
    for B in num_envs:
        for L in num_qubits:
            print("-" * 80)
            print("Batch size:", B)
            print("Num qubits:", L, "\n")
            A = np.array([sample_haar_full(L) for _ in range(B)], dtype=np.complex64)
            C = torch.from_numpy(A).to(dtype=torch.complex64, device="cuda")

            vars = globals().copy()
            vars.update(locals())

            # Numpy
            t = timeit.repeat("x = sqe(A, batched=True)", number=1, repeat=10, globals=vars)
            print(f"Elapsed time for sqe()     : {np.mean(t):.2f} ± {np.std(t):.2f}")

            # CUDA
            t = timeit.repeat("x = cuda_sqe(C, batched=True)", number=1, repeat=10, globals=vars)
            print(f"Elapsed time for cuda_sqe(): {np.mean(t):.2f} ± {np.std(t):.2f}")


if __name__ == "__main__":
    # bench_sqe((1,4,16,32,64), (12,16))
    bench_obs(rdm_2q_mean_real, rdm2m)
