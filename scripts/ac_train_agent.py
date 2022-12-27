"""
python3 ac_train_agent.py -q 5 --env_batch 1024 --steps 40 -i 10001 -b 128 #--ereg 0.01
"""

import argparse
import os
import sys
import time
sys.path.append("..")

from src.agents.ac_agent import ACAgent
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
parser.add_argument("-b", "--batch_size", dest="batch_size", type=int,
    help="Batch size for training value network", default=1)
parser.add_argument("--policy_lr", dest="policy_lr", type=float,
    help="Learning rate for training the policy network", default=1e-4)
parser.add_argument("--value_lr", dest="value_lr", type=float,
    help="Learning rate for training the value network", default=1e-4)
parser.add_argument("--policy_reg", dest="policy_reg", type=float,
    help="L2 regularization for the policy network", default=0.0)
parser.add_argument("--value_reg", dest="value_reg", type=float,
    help="L2 regularization for the value network", default=0.0)
# parser.add_argument("--ereg", dest="entropy_reg", type=float,
#     help="Entropy regularization", default=0.0)
parser.add_argument("--clip_grad", dest="clip_grad", type=float, default=10.0)
parser.add_argument("--dropout", dest="dropout", type=float, default=0.0)
parser.add_argument("--log_every", dest="log_every", type=int, default=100)
parser.add_argument("--test_every", dest="test_every", type=int, default=1000)
parser.add_argument("--save_every", dest="save_every", type=int, default=1000)
# parser.add_argument("--model_path", dest="model_path", type=str,
#     help="File path to load the parameters of a saved policy", default=None)
args = parser.parse_args()


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(args.seed)
set_printoptions(precision=5, sci_mode=False)


# Create file to log output during training.
log_dir = os.path.join("..", "logs", f"{args.num_qubits}qubits",
    f"ac_traj_{args.env_batch}_iters_{args.num_iter}")
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
    Policy Learning rate:           {args.policy_lr}
    Value Learning rate:            {args.value_lr}
    Policy L2 regularization:       {args.policy_reg}
    Value L2 regularization:        {args.value_reg}
    Batch size for value net:       {args.batch_size}
    Grad clipping threshold:        {args.clip_grad}
    Neural network dropout:         {args.dropout}
##############################\n""", logfile)


# Create the environment.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=args.env_batch)
plot_reward_function(env, os.path.join(log_dir, "reward_function.png"))


# Initialize the policy network.
input_size = 2 ** (args.num_qubits + 1)
hidden_dims = [4096, 4096, 512]
output_size = env.num_actions
policy_network = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)


# Initialize the value network.
input_size = 2 ** (args.num_qubits + 1)
hidden_dims = [4096, 4096, 512]
output_size = 1
value_network = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)


# Train an actor-critic agent.
agent = ACAgent(env, policy_network, value_network)
tic = time.time()
agent.train(args.num_iter, args.steps, args.policy_lr, args.value_lr, args. batch_size,
    args.clip_grad, args.policy_reg, args.value_reg, args.log_every, args.test_every,
    args.save_every, log_dir, logfile)
toc = time.time()
agent.save_policy(log_dir)
agent.save_history(log_dir)
logText(f"Training took {toc-tic:.3f} seconds.", logfile)

#