"""
python3 il_train_agent.py -q 5 -b 128 -e 101
"""

import argparse
import os
import pickle
import sys
import time
sys.path.append("..")

import torch

from src.agents.il_agent import ILAgent
from src.envs.rdm_environment import QubitsEnvironment
from src.infrastructure.logging import logText
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions
from src.policies.fcnn_policy import FCNNPolicy


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-q", "--num_qubits", dest="num_qubits", type=int,
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
parser.add_argument("--save_every", dest="save_every", type=int, default=10)
parser.add_argument("--data_file", dest="data_file", type=str, default="beam_100_episodes_100000.pickle",
    help="Name of a pickle file containing training data located inside /data/Nqubits/")
args = parser.parse_args()


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(args.seed)
set_printoptions(precision=5, sci_mode=False)


# Load the dataset.
data_path = os.path.join("..", "data", f"{args.num_qubits}qubits", args.data_file)
with open(data_path, "rb") as f:
    dataset = pickle.load(f)
dataset["states"] = torch.from_numpy(dataset["states"])
dataset["actions"] = torch.from_numpy(dataset["actions"])


# Create file to log output during training.
# args.data_file = "../data/5qubits/beam_size_100/1000000_episodes.pickle"
num_epochs = int(args.data_file.split("/")[-1].split("_")[0])
log_dir = os.path.join("..", "logs", f"{args.num_qubits}qubits",
    f"il_epochs_{num_epochs // 1000}k_batch_{args.batch_size}")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")


# Log hyperparameters information.
# args.lr_decay = np.power(0.1, 1.0/num_iter)
logText(f"""##############################
Training parameters:
    Dataset size:                   {dataset["states"].shape[0]}
    Minimum system entropy (epsi):  {args.epsi}
    Number of epochs:               {args.num_epochs}
    Batch size:                     {args.batch_size}
    Learning rate:                  {args.learning_rate}
    Learning rate decay:            {args.lr_decay}
    Final learning rate:            {round(args.learning_rate * (args.lr_decay ** args.num_epochs), 7)}
    Weight regularization:          {args.reg}
    Grad clipping threshold:        {args.clip_grad}
    Neural network dropout:         {args.dropout}
##############################\n""", log_file)


# Create the environment.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=1)


# Initialize the policy.
input_size = 2 ** (args.num_qubits + 1)
hidden_dims = [4096, 4096, 512]
output_size = env.num_actions
policy = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)


# Train the imitation learning agent.
agent = ILAgent(env, policy)
tic = time.time()
agent.train(dataset, args.num_epochs, args.batch_size, args.learning_rate, args.lr_decay,
            args.clip_grad, args.reg, args.log_every, args.test_every, args.save_every,
            log_dir, log_file)
toc = time.time()
agent.save_policy(log_dir)
agent.save_history(log_dir)
logText(f"Training took {toc-tic:.3f} seconds.", log_file)

#