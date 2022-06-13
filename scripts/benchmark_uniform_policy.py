"""
python3 benchmark_uniform_policy.py -s 0 -q 5 --env_batch 100 --steps 30 --max_steps 5000 --epsi 1e-3 -t 100
"""

import argparse
import os
import time
import sys
sys.path.append("..")

from src.envs.rdm_environment import QubitsEnvironment
from src.policies.uniform_policy import UniformPolicy
from src.agents.pg_agent import PGAgent
from src.infrastructure.util_funcs import fix_random_seeds
from src.infrastructure.logging import logText, log_test_stats


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-q", "--num_qubits", dest="num_qubits", type=int,
    help="Number of qubits in the quantum system", default=2)
parser.add_argument("--env_batch", dest="env_batch", type=int,
    help="Number of states in the environment batch", default=1)
parser.add_argument("--steps", dest="steps", type=int,
    help="Number of steps in an episode. Use this parameter to set the number " +
    "of steps used to test the accuracy.", default=10)
parser.add_argument("--max_steps", dest="max_steps", type=int,
    help="Maximum allowed number of steps in an episode. Use this parameter to set the " +
    "maximum number of steps used to test the average steps to disentangle.", default=1000)
parser.add_argument("--epsi", dest="epsi", type=float,
    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("-t", "--num_test", dest="num_test", type=int,
    help="Number of iterations to run the test for", default=1)
args = parser.parse_args()


fix_random_seeds(args.seed)


# Create direcotry for logging.
log_dir = os.path.join("..", "logs", f"{args.num_qubits}qubits", "benchmark_uniform_policy")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "bench.log")


# Create environment, policy and agent.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=args.env_batch)
policy = UniformPolicy(env.num_actions)
agent = PGAgent(env, policy)


# Test agent accuracy.
tic = time.time()
entropies, returns, nsolved, nsteps = agent.test_accuracy(args.num_test, args.steps, greedy=False)
stats = {"entropies":entropies, "returns":returns, "nsolved":nsolved, "nsteps":nsteps}
toc = time.time()
logText(f"Testing agent accuracy took {toc-tic:.3f} seconds.", log_file)
logText(f"Average accuracy for {args.steps} number of steps:", log_file)
log_test_stats(stats, log_file)


# Test the average steps to disentangle.
tic = time.time()
entropies, returns, nsolved, nsteps = agent.test_accuracy(args.num_test, args.max_steps, greedy=False)
stats = {"entropies":entropies, "returns":returns, "nsolved":nsolved, "nsteps":nsteps}
toc = time.time()
logText(f"Testing average steps to disentangle took {toc-tic:.3f} seconds.", log_file)
log_test_stats(stats, log_file)

#