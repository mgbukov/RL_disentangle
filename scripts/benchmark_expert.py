"""
python3 benchmark_expert.py -s 0 -q 5 --epsi 1e-3 --beam_size 100 -t 100
"""

import argparse
import os
import time
import sys
sys.path.append("..")

import numpy as np

from src.agents.expert import SearchExpert
from src.envs.rdm_environment import QubitsEnvironment
from src.infrastructure.util_funcs import fix_random_seeds
from src.infrastructure.logging import logText, log_test_stats


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-q", "--num_qubits", dest="num_qubits", type=int,
    help="Number of qubits in the quantum system", default=2)
parser.add_argument("--epsi", dest="epsi", type=float,
    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("--beam_size", dest="beam_size", type=int,
    help="Beam size for beam search expert", default=100)
parser.add_argument("-t", "--num_test", dest="num_test", type=int,
    help="Number of iterations to run the test for", default=1)
args = parser.parse_args()


fix_random_seeds(args.seed)


# Create direcotry for logging.
log_dir = os.path.join("..", "logs", f"{args.num_qubits}qubits", "benchmark_expert")
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, "bench.log")


# Create environment, policy and agent.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=1)
expert = SearchExpert(env, beam_size=args.beam_size)


# Test expert.
tic = time.time()
num_actions = []
for _ in range(args.num_test):
    env.set_random_states()
    _, actions = expert.rollout(env.states[0])
    num_actions.append(len(actions))
toc = time.time()


# Log the results.
total_num_steps = sum(num_actions)
total_time = toc - tic
logText(f"Testing the expert on {args.num_qubits} qubits for {args.num_test} iterations " +
    f"took {toc-tic:.3f} seconds.", logfile)
logText(f"    Average number of steps to disentangle a {args.num_qubits}-qubit system: " +
    f"{total_num_steps / args.num_test: .1f} steps", logfile)
logText(f"    Average time to disentangle a {args.num_qubits}-qubit system: " +
    f"{total_time / args.num_test: .3f} seconds", logfile)

#