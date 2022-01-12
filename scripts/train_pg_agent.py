"""
python3 train_pg_agent.py -n 5 -b 1024 --steps 30  -i 4001 --verbose
"""

import argparse
import os
import pickle
import time
import sys
sys.path.append("..")

from src.agents.pg_agent import PGAgent
from src.envs.rdm_environment import QubitsEnvironment
from src.policies.fcnn_policy import FCNNPolicy
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions
from src.infrastructure.logging import (logTxt, plot_entropy_curves, plot_loss_curve,
                                        plot_nsolved_curves, plot_return_curves)


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
parser.add_argument("-i", "--num_iter", dest="num_iter", type=int,
                    help="Number of iterations to run the training for", default=1)
parser.add_argument("--lr", dest="learning_rate", type=float,
                    help="Learning rate", default=1e-4)
parser.add_argument("--reg", dest="reg", type=float,
                    help="L2 regularization", default=0.0)
parser.add_argument("--ereg", dest="entropy_reg", type=float,
                    help="Entropy regularization", default=0.0)

parser.add_argument("--verbose", dest="verbose", action='store_true', default=False)
args = parser.parse_args()


# Fix the random seeds for numpy and pytorch.
fix_random_seeds(args.seed)


# Create the environment.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=args.batch_size)


# Initialize the policy.
input_size = 2 ** (args.num_qubits + 1)
hidden_dims = [4096, 4096, 512]
output_size = env.num_actions
dropout_rate = 0.0
policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate)


# # Train the policy-gradient agent.
# lr_decay = 1.0 # np.power(0.1, 1.0/num_iter)
# clip_grad = 10.0
# log_every = 1
# test_every = 10
# verbose = True
# log_dir = "../logs/5qubits/general/traj_{}_iters_{}".format(args.batch_size, args.num_iter, hidden_dims)
log_dir = "test"
if not os.path.exists(log_dir + "/probs"):
    os.makedirs(log_dir + "/probs")
agent = PGAgent(env, policy, log_dir=log_dir)

# tic = time.time()
# agent.train(args.num_iter, args.steps, args.learning_rate, lr_decay, clip_grad,
#             args.reg, args.entropy_reg, log_every, test_every, args.verbose)
# toc = time.time()
# logTxt(f"Training took {toc-tic:.3f} seconds.", os.path.join(log_dir, "train_history.txt"), verbose)


# # Plot the results.
# with open(os.path.join(log_dir, "train_history.pickle"), "rb") as f:
#     train_history = pickle.load(f)
# with open(os.path.join(log_dir, "test_history.pickle"), "rb") as f:
#     test_history = pickle.load(f)

# plot_entropy_curves(train_history, os.path.join(log_dir, "final_entropy.png"))
# plot_loss_curve(train_history, os.path.join(log_dir, "loss.png"))
# plot_return_curves(train_history, test_history, os.path.join(log_dir, "returns.png"))
# plot_nsolved_curves(train_history, test_history, os.path.join(log_dir, "nsolved.png"))

#