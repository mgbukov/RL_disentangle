import argparse
import itertools
import json
import pickle
import os

import numpy as np
import torch

from src.quantum_env import QuantumEnv
from src.quantum_state import sample_mps
from src.util import str2state


# Define the initial states on which we want to test the agent.
TEST_HAAR_RANDOM_STATES = {
    4: ["RR-R-R", "RR-RR", "RRR-R", "RRRR"],
    5: ["RR-R-R-R", "RR-RR-R", "RRR-R-R", "RRR-RR", "RRRR-R", "RRRRR"],
    6: ["RR-R-R-R-R", "RR-RR-R-R", "RR-RR-RR", "RRR-R-R-R", "RRR-RR-R",
        "RRRR-R-R", "RRRR-RR", "RRRRR-R", "RRRRRR"],
    8: ["RRR-RRR-RR", "RRRR-RRRR", "RRRRR-RRR", "RRRRRR-RR", "RRRRRR-R", "RRRRRRRR"]
}


def test_agent(agent, states, **env_kwargs):
    """Test the agent on a set of specifically generated quantum states.
    Note that this function will reset the numpy rng seed.
    """

    # Initialize the environment
    states = np.asarray(states)
    num_envs = env_kwargs.pop("num_envs", states.shape[0])
    num_qubits = env_kwargs.pop("num_qubits", states.ndim - 1)
    shape = (num_envs,) + (2,) * num_qubits
    env = QuantumEnv(num_qubits=num_qubits, num_envs=num_envs, **env_kwargs)
    env.reset()
    env.simulator.states = states.reshape(shape)

    o = env.obs_fn(env.simulator.states)
    lengths = np.full(num_envs, np.nan, dtype=np.float32)
    done = np.full(num_envs, False, dtype=bool)
    solved = 0
    for _ in range(env.max_episode_steps):
        o = torch.from_numpy(o)
        pi = agent.policy(o)    # uses torch.no_grad
        acts = torch.argmax(pi.probs, dim=1).cpu().numpy() # greedy selection

        o, r, t, tr, infos = env.step(acts)
        if t.any():
            for k in range(env.num_envs):
                if t[k] and not done[k]:
                    lengths[k] = infos["episode"]["l"][k]
                    done[k] = True
                    solved += 1

        if np.all(done):
            break

    return {
        "avg_len": float(np.nanmean(lengths)),
        "95_percentile": float(np.nanpercentile(lengths, 95.)),
        "max_len": float(np.nanmax(lengths)),
        "ratio_solved": float(solved / num_envs),
    }


def test_on_haar_random(agent, num_qubits, n_tests=128, **env_kwargs):

    results = {}
    for L, names in TEST_HAAR_RANDOM_STATES.items():
        if L > num_qubits:
            continue
        results[L] = {}
        for name in names:
            states = np.array([str2state(name) for _ in range(n_tests)])
            results[L][name] = test_agent(agent, states, **env_kwargs)

    return results


def test_on_mps(agent, num_qubits, chi_max, n_tests=128, **env_kwargs):

    states = np.array([sample_mps(num_qubits, chi_max) for _ in range(n_tests)])
    return test_agent(agent, states, **env_kwargs)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str)
    parser.add_argument("--num_qubits", type=int)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--epsi", type=float, default=1e-3)
    parser.add_argument("--obs_fn", type=str, default="rdm_2q_mean_real")
    parser.add_argument("--n_tests", type=int, default=128)
    parser.add_argument("--chi_max", type=int, default=4)

    args = parser.parse_args()

    print(f"Loading agent from \"{args.agent}\"...")
    agent = torch.load(args.agent, map_location="cpu")
    print(f"Agent loaded successfuly!")
    agent.policy_network.eval()
    agent.value_network.eval()

    results = {}
    env_kwargs = dict(
        max_episode_steps=      args.max_steps,
        epsi=                   args.epsi,
        obs_fn=                 args.obs_fn,
    )
    results['agent'] = args.agent
    results["env_kwargs"] = env_kwargs
    results["chi_max"] = args.chi_max

    print("Testing on Haar random states...")
    results["haar_random"] = test_on_haar_random(
        agent,
        args.num_qubits,
        args.n_tests,
        **env_kwargs
    )
    print("Testing on Matrix product states...")
    results["mps"] = test_on_mps(
        agent,
        args.num_qubits,
        args.chi_max,
        args.n_tests,
        **env_kwargs
    )

    print("Testing completed! Writing results...")
    with open(args.output, mode='wt') as f:
        json.dump(results, f, indent=2)
    print("Done!")
