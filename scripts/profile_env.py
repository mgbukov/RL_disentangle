import argparse
import time
import cProfile

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
    parser.add_argument("env", type=str, choices=["numpy", "torch"])
    parser.add_argument("--device", "-d", type=str, choices=["cpu", "cuda"])
    parser.add_argument("-q", type=int)
    parser.add_argument("-e", type=int)
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()

    common_kwags = dict(
        epsi=1e-3, fast_obs=True, swaps=True, reward_fn="relative_delta"
    )
    if args.env == "numpy":
        env = QuantumEnv(args.q, args.e, obs_fn="rdm_2q_mean_real", **common_kwags)
        env.reset()
        sync = False
    else:
        env = QEnv(args.q, args.e, obs_fn="rdm2m", fast_ents=True, device=args.device, **common_kwags)
        sync = True if args.device == "cuda" else False

    cProfile.run("benchmark_env(env, args.n, sync)", sort="tottime")
