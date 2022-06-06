"""
python3 pg_train_agent.py -n 5 -b 4096 --steps 40  -i 10001 --ereg 0.01
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
parser.add_argument("--lr_decay", dest="lr_decay", type=float, default=1.0)
parser.add_argument("--reg", dest="reg", type=float,
                    help="L2 regularization", default=0.0)
parser.add_argument("--ereg", dest="entropy_reg", type=float,
                    help="Entropy regularization", default=0.0)
parser.add_argument("--clip_grad", dest="clip_grad", type=float, default=10.0)
parser.add_argument("--dropout", dest="dropout", type=float, default=0.0)
parser.add_argument("--log_every", dest="log_every", type=int, default=100)
parser.add_argument("--test_every", dest="test_every", type=int, default=1000)
parser.add_argument("--model_path", dest="model_path", type=str,
    help="File path to load the parameters of a saved policy", default=None)
args = parser.parse_args()


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(args.seed)
set_printoptions(precision=5, sci_mode=False)


# Create file to log output during training.
log_dir = "../logs/5qubits/traj_{}_iters_{}_entreg_{}".format(
    args.batch_size, args.num_iter, args.entropy_reg)
log_probs_dir = os.path.join(log_dir, "probs")
os.makedirs(log_probs_dir, exist_ok=True)
logfile = os.path.join(log_dir, "train.log")


# Log hyperparameters information.
# args.lr_decay = np.power(0.1, 1.0/num_iter)
logText(f"""##############################
Training parameters:
    Number of trajectories:         {args.batch_size}
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
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=args.batch_size)
plot_reward_function(env, os.path.join(log_dir, "reward_function.png"))


# Initialize the policy.
if args.model_path is None:
    input_size = 2 ** (args.num_qubits + 1)
    hidden_dims = [4096, 4096, 512]
    output_size = env.num_actions
    policy = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)
else:
    policy = FCNNPolicy.load(args.model_path)

# Train the policy-gradient agent.
agent = PGAgent(env, policy)
tic = time.time()
agent.train(args.num_iter, args.steps, args.learning_rate, args.lr_decay, args.clip_grad,
            args.reg, args.entropy_reg, args.log_every, args.test_every, logfile)
toc = time.time()
agent.save_policy(log_dir)
agent.save_history(log_dir)
logText(f"Training took {toc-tic:.3f} seconds.", logfile)

#