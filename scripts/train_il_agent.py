"""
python3 train_il_agent.py -n 5 -b 64 -e 100001
"""

import argparse
import os
import pickle
import sys
import time
sys.path.append("..")


import numpy as np
import torch


from src.agents.il_agent import ILAgent
from src.envs.rdm_environment import QubitsEnvironment
from src.infrastructure.logging import (plot_distribution, plot_entropy_curves, plot_loss_curve,
                                        plot_nsolved_curves, plot_return_curves)
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions
from src.policies.fcnn_policy import FCNNPolicy


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-n", "--num_qubits", dest="num_qubits", type=int,
                    help="Number of qubits in the quantum system", default=2)
parser.add_argument("--epsi", dest="epsi", type=float,
                    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int,
                    help="Number of epochs to run the training for", default=1)
parser.add_argument("-b", "--batch_size", dest="batch_size", type=int,
                    help="Batch size parameter for policy network optimization", default=1)
parser.add_argument("--lr", dest="learning_rate", type=float,
                    help="Learning rate", default=1e-4)
parser.add_argument("--lr_decay", dest="lr_decay", type=float, default=1.0)
parser.add_argument("--reg", dest="reg", type=float,
                    help="L2 regularization", default=0.0)
parser.add_argument("--clip_grad", dest="clip_grad", type=float, default=10.0)
parser.add_argument("--dropout", dest="dropout", type=float, default=0.0)
parser.add_argument("--log_every", dest="log_every", type=int, default=1)
parser.add_argument("--test_every", dest="test_every", type=int, default=10)
args = parser.parse_args()


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(args.seed)
set_printoptions(precision=5, sci_mode=False)


# Create file to log output during training.
log_dir = "../logs/5qubits/imitation_100k"
os.makedirs(log_dir, exist_ok=True)
stdout = open(os.path.join(log_dir, "train_history_100k.txt"), "w")
# stdout = sys.stdout


# Log hyperparameters information.
# args.lr_decay = np.power(0.1, 1.0/num_iter)
print(f"""##############################
Training parameters:
    Minimum system entropy (epsi):  {args.epsi}
    Number of epochs:               {args.num_epochs}
    Batch size:                     {args.batch_size}
    Learning rate:                  {args.learning_rate}
    Learning rate decay:            {args.lr_decay}
    Final learning rate:            {round(args.learning_rate * (args.lr_decay ** args.num_epochs), 7)}
    Weight regularization:          {args.reg}
    Grad clipping threshold:        {args.clip_grad}
    Neural network dropout:         {args.dropout}
##############################\n""", file=stdout)


# Create the environment.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=1)


# Initialize the policy.
input_size = 2 ** (args.num_qubits + 1)
hidden_dims = [4096, 4096, 512]
output_size = env.num_actions
policy = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)


# Load the dataset.
data_path = "../data/5qubits/beam_size=100/100000_episodes.pickle"
with open(data_path, "rb") as f:
    dataset = pickle.load(f)
dataset["states"] = torch.from_numpy(dataset["states"])
dataset["actions"] = torch.from_numpy(dataset["actions"])


# Train the imitation learning agent.
agent = ILAgent(env, policy)
tic = time.time()
agent.train(dataset, args.num_epochs, args.batch_size, args.learning_rate, args.lr_decay,
            args.clip_grad, args.reg, args.log_every, args.test_every, stdout)
toc = time.time()
agent.save_policy(log_dir)
agent.save_history(log_dir)
print(f"Training took {toc-tic:.3f} seconds.", file=stdout)


# Close the logging file.
stdout.close()


# Plot the results.
with open(os.path.join(log_dir, "train_history.pickle"), "rb") as f:
    train_history = pickle.load(f)
with open(os.path.join(log_dir, "test_history.pickle"), "rb") as f:
    test_history = pickle.load(f)
plot_entropy_curves(test_history, os.path.join(log_dir, "final_entropy.png"))
plot_entropy_curves(train_history, os.path.join(log_dir, "train_entropy.png"))
plot_loss_curve(train_history, os.path.join(log_dir, "loss.png"))
# plot_return_curves(train_history, test_history, os.path.join(log_dir, "returns.png"))

#