import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.environment_loop import test_agent
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state
from search import GreedyAgent, RandomAgent



ROOT = Path(__file__).parent.parent.parent.parent
PATH_4Q_AGENT = ROOT / "models" / "4q_agent.pt"
PATH_5Q_AGENT = ROOT / "models" / "5q_agent.pt"
PATH_6Q_AGENT = ROOT / "models" / "6q_agent.pt"
PATH_PROCESSED = ROOT / "data" / "processed" / "fig10"


def str2state(string_descr):
    psi = np.array([1.0], dtype=np.complex64)
    nqubits = 0
    for pair in string_descr.split('-'):
        q = pair.count('R')
        nqubits += q
        psi = np.kron(psi, random_quantum_state(q=q, prob=1.))
    return psi.reshape((2,) * nqubits)


def benchmark_agents(ntests=1000):
    TEST_STATES = {
        4: ["RR-R-R", "RR-RR", "RRR-R", "RRRR"],
        5: ["RR-R-R-R", "RR-RR-R", "RRR-R-R", "RRR-RR", "RRRR-R", "RRRRR"],
        6: ["RR-R-R-R-R", "RR-RR-R-R", "RR-RR-RR", "RRR-R-R-R", "RRR-RR-R",
            "RRRR-R-R", "RRRR-RR", "RRRRR-R", "RRRRRR"]
    }
    MAX_STEPS = 90

    # Load agents
    rl_agent6q = torch.load(PATH_6Q_AGENT, map_location='cpu')
    rl_agent5q = torch.load(PATH_5Q_AGENT, map_location='cpu')
    rl_agent4q = torch.load(PATH_4Q_AGENT, map_location='cpu')
    for enc in rl_agent6q.policy_network.net:
        enc.activation_relu_or_gelu = 1
    rl_agent6q.policy_network.eval()
    for enc in rl_agent5q.policy_network.net:
        enc.activation_relu_or_gelu = 1
    rl_agent5q.policy_network.eval()
    for enc in rl_agent4q.policy_network.net:
        enc.activation_relu_or_gelu = 1
    rl_agent4q.policy_network.eval()
    greedy = GreedyAgent(epsi=1e-3)
    random = RandomAgent(epsi=1e-3)

    results = {}
    # Do the tests
    for num_qubits in (4,5,6):
        print(f'Testing on {num_qubits} qubit system...')
        for state_str in TEST_STATES[num_qubits]:
            print(f'\tTesting states', state_str)
            # Generate test states
            initial_states = np.array(
                [str2state(state_str) for _ in range(ntests)])

            # Test RL agent
            if num_qubits == 6:
                rl_agent = rl_agent6q
            elif num_qubits == 5:
                rl_agent = rl_agent5q
            elif num_qubits == 4:
                rl_agent = rl_agent4q
            RL_res = test_agent(
                rl_agent, initial_states, num_envs=ntests,
                obs_fn="rdm_2q_mean_real", max_episode_steps=MAX_STEPS)
            print('\t\tDone testing RL agent.')

            # Test random agent
            env = QuantumEnv(num_qubits, 1, epsi=1e-3)
            random_len = []
            random_ent = []
            random_solves = []
            for s in initial_states:
                env.reset()
                env.simulator.states = np.expand_dims(s, 0)
                path, ents = random.start(s, env.simulator)
                if path is None and ents is None:
                    random_len.append(np.nan)
                    random_ent.append(np.nan)
                    random_solves.append(np.nan)
                else:
                    random_len.append(len(path))
                    random_ent.append(np.mean(ents[-1]))
                    random_solves.append(1)
            print('\t\tDone testing Random agent.')

            # Test greedy agent
            greedy_len = []
            greedy_ent = []
            greedy_solves = []
            for s in initial_states:
                env.reset()
                env.simulator.states = np.expand_dims(s, 0)
                path, ents = greedy.start(s, env.simulator)
                if path is None and ents is None:
                    greedy_len.append(np.nan)
                    greedy_ent.append(np.nan)
                    greedy_solves.append(np.nan)
                else:
                    greedy_len.append(len(path))
                    greedy_ent.append(np.mean(ents[-1]))
                    greedy_solves.append(1)
            print('\t\tDone testing Greedy agent.')

            results[state_str] = {
                "rl": {
                    "avg_steps": float(np.nanmean(RL_res["lengths"][RL_res["done"]])),
                    "std_steps": float(np.nanstd(RL_res["lengths"][RL_res["done"]])),
                    "avg_final_ent": float(np.nanmean(RL_res["entanglements"][RL_res["done"]])),
                    "success": float(np.nanmean(RL_res["done"]))
                },
                "random": {
                    "avg_steps": float(np.nanmean(random_len)),
                    "std_steps": float(np.nanstd(random_len)),
                    "avg_final_ent": float(np.nanmean(random_ent)),
                    "success": float(np.nanmean(random_solves))
                },
                "greedy": {
                    "avg_steps": float(np.nanmean(greedy_len)),
                    "std_steps": float(np.nanstd(greedy_len)),
                    "avg_final_ent": float(np.nanmean(greedy_ent)),
                    "success": float(np.nanmean(greedy_solves))
                }
            }
    return results



if __name__ == '__main__':

    # Benchmark agents
    results = benchmark_agents(1000)
    with open(PATH_PROCESSED / "_agents-benchmark-final.json", mode='w') as f:
        json.dump(results, f, indent=2)
