import argparse
import json
import os
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from run import test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', action='store', type=str,
                        help='Path to agent')
    parser.add_argument('--num_qubits', action='store', type=int,
                        help='Number of qubits')
    parser.add_argument('--num_envs', action='store', type=int, default=1024,
                        help='Number of environments')
    parser.add_argument('--steps_limit', action='store', type=int,
                        help='Steps limit')
    parser.add_argument('--obs_fn', action='store', type=str,
                        help='Observation function')
    parser.add_argument('--epsi', action='store', type=float, default=1e-3,
                        help='Disentanglement threshold')
    parser.add_argument('--seed', action='store', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output', action='store', type=str,
                        help='Savename')
    args = parser.parse_args()

    # Load agent
    agent = torch.load(args.agent, map_location='cpu')
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()
    agent.value_network.eval()

    results = test(agent, args)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
