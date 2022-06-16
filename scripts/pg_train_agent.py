"""
python3 pg_train_agent.py -q 5 --env_batch 1024 --steps 40 -i 10001 --ereg 0.01
"""

import argparse
import os
import sys
import time
sys.path.append("..")

from src.agents.pg_agent import PGAgent
from src.envs.rdm_environment import QubitsEnvironment
from src.infrastructure.logging import logText, plot_reward_function
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions
from src.policies.fcnn_policy import FCNNPolicy


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-q", "--num_qubits", dest="num_qubits", type=int,
    help="Number of qubits in the quantum system", default=2)
parser.add_argument("--env_batch", dest="env_batch", type=int,
    help="Number of states in the environment batch", default=1)
parser.add_argument("--steps", dest="steps", type=int,
    help="Number of steps in an episode", default=10)
parser.add_argument("--epsi", dest="epsi", type=float,
    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("-i", "--num_iter", dest="num_iter", type=int,
    help="Number of iterations to run the training for", default=1)
parser.add_argument("--lr", dest="learning_rate", type=float,
    help="Learning rate", default=1e-4)
parser.add_argument("--lr_decay", dest="lr_decay", type=float, default=1.0)
parser.add_argument("--reg", dest="reg", type=float,
    help="L2 regularization", default=0.0)
parser.add_argument("--ereg", dest="entropy_reg", type=float,
    help="Entropy regularization", default=0.0)
parser.add_argument("--clip_grad", dest="clip_grad", type=float, default=10.0)
parser.add_argument("--dropout", dest="dropout", type=float, default=0.0)
parser.add_argument("--log_every", dest="log_every", type=int, default=100)
parser.add_argument("--test_every", dest="test_every", type=int, default=1000)
parser.add_argument("--save_every", dest="save_every", type=int, default=1000)
parser.add_argument("--model_path", dest="model_path", type=str,
    help="File path to load the parameters of a saved policy", default=None)
args = parser.parse_args()


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(args.seed)
set_printoptions(precision=5, sci_mode=False)


# Create file to log output during training.
pretrain = ""
if args.model_path is not None:
    # args.model_path = "../logs/5qubits/imitation_100k/policy_80.bin"
    pretrain = "_pretrain_" + args.model_path.split("_")[-1].split(".")[0]
log_dir = os.path.join("..", "logs", f"{args.num_qubits}qubits",
    f"pg_traj_{args.env_batch}_iters_{args.num_iter}_entreg_{args.entropy_reg}{pretrain}")
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, "train.log")


# Log hyperparameters information.
# args.lr_decay = np.power(0.1, 1.0/num_iter)
logText(f"""##############################
Training parameters:
    Number of trajectories:         {args.env_batch}
    Maximum number of steps:        {args.steps}
    Minimum system entropy (epsi):  {args.epsi}
    Number of iterations:           {args.num_iter}
    Learning rate:                  {args.learning_rate}
    Learning rate decay:            {args.lr_decay}
    Final learning rate:            {round(args.learning_rate * (args.lr_decay ** args.num_iter), 7)}
    Weight regularization:          {args.reg}
    Entropy regularization:         {args.entropy_reg}
    Grad clipping threshold:        {args.clip_grad}
    Neural network dropout:         {args.dropout}
##############################\n""", logfile)


# Create the environment.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=args.env_batch)
plot_reward_function(env, os.path.join(log_dir, "reward_function.png"))


# Initialize the policy.
input_size = 2 ** (args.num_qubits + 1)
hidden_dims = [4096, 4096, 512]
output_size = env.num_actions
policy = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)


# Maybe load a pre-trained model.
if args.model_path is not None:
    logText(f"Loading pre-trained model from {args.model_path}...")
    policy = FCNNPolicy.load(args.model_path)


# Train a policy-gradient agent.
agent = PGAgent(env, policy)
tic = time.time()
agent.train(args.num_iter, args.steps, args.learning_rate, args.lr_decay, args.clip_grad,
    args.reg, args.entropy_reg, args.log_every, args.test_every, args.save_every,
    log_dir, logfile)
toc = time.time()
agent.save_policy(log_dir)
agent.save_history(log_dir)
logText(f"Training took {toc-tic:.3f} seconds.", logfile)

#