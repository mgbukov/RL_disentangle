"""
python3 test_random.py -s 0 -n 5 -b 100 --steps 30  --epsi 1e-3 -t 100 --verbose
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
from src.infrastructure.logging import logTxt, log_test_stats


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-n", "--num_qubits", dest="num_qubits", type=int,
                    help="Number of qubits in the quantum system", default=2)
parser.add_argument("-b", "--batch_size", dest="batch_size", type=int,
                    help="Number of states in the environment batch", default=1)
parser.add_argument("--steps", dest="steps", type=int,
                    help="Number of steps in an episode", default=10)
parser.add_argument("--epsi", dest="epsi", type=float,
                    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("-t", "--num_test", dest="num_test", type=int,
                    help="Number of iterations to run the test for", default=1)
parser.add_argument("--verbose", dest="verbose", action='store_true', default=False)
args = parser.parse_args()


# Create direcotry for logging.
log_dir = f"../logs/{args.num_qubits}qubits/test/uniform_policy"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Create environment, policy and agent.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=args.batch_size)
policy = UniformPolicy(env.num_actions)
agent = PGAgent(env, policy, log_dir=log_dir)


# Run the tests.
tic = time.time()
fix_random_seeds(args.seed)
entropies, returns, nsolved = agent.test_accuracy(args.num_test, args.steps, greedy=False)
toc = time.time()
log_text_file = os.path.join(log_dir, "test_reward_v2.txt")
logTxt(f"Testing took {toc-tic:.3f} seconds.", log_text_file, args.verbose, create=True)
log_test_stats((entropies, returns, nsolved), log_text_file, args.verbose)

#