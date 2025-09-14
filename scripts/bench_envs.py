import argparse
import itertools
import time

import numpy as np
import torch

import context
from src.quantum_env import QuantumEnv
from src.qenv import QEnv


def benchmark_env(env, num_steps=10, sync=False):
    tic = time.time()
    N = env.num_envs
    A = env.single_action_space.n
    for i in range(num_steps):
        a = np.random.randint(0, A, size=N)
        o, r, t, tr, _ = env.step(a)
        if sync:
            torch.cuda.synchronize()
    toc = time.time()
    return toc - tic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default="cpu")
    device = parser.parse_args().device

    NUM_STEPS  = 10
    NUM_QUBITS = (6,8,10,12,16)
    NUM_ENVS   = (32, 64, 128)
    common_kwags = dict(
        epsi=1e-3, fast_obs=True, swaps=True, reward_fn="relative_delta"
    )

    for num_qubits, num_envs in itertools.product(NUM_QUBITS, NUM_ENVS):
        numpy_env = QuantumEnv(num_qubits, num_envs, obs_fn="rdm_2q_mean_real", **common_kwags)
        numpy_env.reset()
        torch_env = QEnv(num_qubits, num_envs, obs_fn="rdm2m", fast_ents=True, device=device, **common_kwags)
        numpy_elapsed = benchmark_env(numpy_env, NUM_STEPS, sync=False)
        torch_elapsed = benchmark_env(torch_env, NUM_STEPS, sync=True if device == "cuda" else False)
        print(f"Numpy env[{num_qubits}q, {num_envs}e, cpu]:".ljust(30) + f"{numpy_elapsed:.2f}s")
        print(f"Torch env[{num_qubits}q, {num_envs}e, {device}]:".ljust(30) + f"{torch_elapsed:.2f}s")
        print()