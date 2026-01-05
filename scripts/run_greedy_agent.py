import argparse
import json
import os
import time

from tqdm import tqdm

import context
from src.qenv import QEnv
from src.stategen import (
    StateGenerator,
    sample_haar_full,
    sample_haar_generalized,
    sample_haar_product
)
from search import GreedyAgent


EPSI = 1e-3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits", "-n", type=int)
    parser.add_argument("--num_tests", "-t", default=100, type=int)
    parser.add_argument("--max_iterations", "-i", type=int, default=10_000)
    parser.add_argument("--family", "-f", choices=["fullhaar", "weakly_entangled", "product"])
    parser.add_argument("--eta", "-e", type=float, default=4.1)
    parser.add_argument("--subsystem_size", "-s", type=int, default=4)
    parser.add_argument("--output", "-o", type=str)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.family == "fullhaar":
        stategen = StateGenerator(sample_haar_full, num_qubits=args.num_qubits)
    elif args.family == "product":
        stategen = StateGenerator(
            sample_haar_product,
            num_qubits=args.num_qubits,
            sample_params=dict(min_subsystem_size=args.subsystem_size,
                               max_subsystem_size=args.subsystem_size)
        )
    elif args.family == "weakly_entangled":
        stategen = StateGenerator(
            sample_haar_generalized,
            num_qubits=args.num_qubits,
            sample_params=dict(min_subsystem_size=args.subsystem_size,
                               max_subsystem_size=args.subsystem_size,
                               min_eta=args.eta,
                               max_eta=args.eta)
        )
    else:
        raise ValueError("Unexpected control flow")

    env = QEnv(num_qubits=args.num_qubits, num_envs=1, epsi=EPSI,
               act_space="reduced", reward_fn="relative_delta", obs_fn="rdm2m",
               state_generator=stategen, fast_ents=True, fast_obs=True,
               swaps=False, device="cpu")

    agent = GreedyAgent(epsi=EPSI)

    results = {
        "num_qubits":       args.num_qubits,
        "num_tests":        args.num_tests,
        "max_iterations":   args.max_iterations,
        "family":           args.family,
        "eta":              args.eta,
        "subsystem_size":   args.subsystem_size,
        "results":          []
    }

    print("Starting Greedy Agent")
    for _ in tqdm(range(args.num_tests)):
        env.reset()
        psi = env.simulator.states[0]

        tic = time.time()
        path = agent.start(psi, env, num_iter=args.max_iterations)
        toc = time.time()

        if path is None:
            print("F", end="", flush=True)
        else:
            print(".", end="", flush=True)
            results["results"].append({
                "steps":    len(path),
                "actions":  path,
                "elapsed":  toc - tic,
            })

            print("\n\nSaving JSON")
            with open(args.output, mode="w") as f:
                json.dump(results, f, indent="  ")

            print("Done")