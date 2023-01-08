"""
python3 generate_data.py -s 0 -q 5 --beam_size 100 --epsi 1e-3 --num_episodes 100000 --env rdm
"""
import argparse
import os
import pickle
import sys
import time
sys.path.append("..")

import numpy as np
from tqdm import tqdm

from src.agents.expert import SearchExpert
from src.envs.rdm_environment import QubitsEnvironment as RDMEnvironment
from src.envs.chipoff_environment import QubitsEnvironment as ChipOffEnvironment
from src.infrastructure.logging import logText
from src.infrastructure.util_funcs import fix_random_seeds


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-q", "--num_qubits", dest="num_qubits", type=int,
                    help="Number of qubits in the quantum system", default=2)
parser.add_argument("--beam_size", dest="beam_size", type=int,
                    help="Size of the beam for beam search", default=10)
parser.add_argument("--epsi", dest="epsi", type=float,
                    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("--num_episodes", dest="num_episodes", type=int,
                    help="Number of episodes to be generated", default=1)
parser.add_argument("--env", dest="env", type=str, default="rdm",
                    help="The type of the environment: rdm or chip-off.")
args = parser.parse_args()


assert args.env in ["rdm", "chip-off"], f"unknown environment {args.env}"


# Fix seeds
fix_random_seeds(args.seed)


# Create file to log output during training.
log_dir = os.path.join("..", "data", f"{args.num_qubits}qubits", args.env)
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, "generate.log")

# Create the environment and the expert.
if args.env == "rdm":
    env = RDMEnvironment(args.num_qubits, epsi=args.epsi, batch_size=1)
else:
    env = ChipOffEnvironment(args.num_qubits, epsi=args.epsi, batch_size=1)
expert = SearchExpert(env, args.beam_size)


# Run the expert to generate data.
logText(f"Solving {args.num_episodes} {args.num_qubits}-qubits systems", logfile)
dataset = {"states":[], "actions":[]}
tic = time.time()
for _ in tqdm(range(args.num_episodes)):
    flag = False
    while not flag:
        env.set_random_states()
        psi = env.states[0]
        states, actions = expert.rollout(psi, num_iter=1000, verbose=False)

        # If (states, actions) = (None, None) then the expert could not solve the state.
        if states is None or actions is None:
            continue

        dataset["states"].append(states)
        dataset["actions"].append(actions)
        flag = True

toc = time.time()
logText(f"Data generation took {toc-tic:.3f} seconds", logfile)

dataset["states"] = np.vstack(dataset["states"])
dataset["actions"] = np.hstack(dataset["actions"])


# Save the dataset.
data_file = os.path.join(
    log_dir, f"beam_{args.beam_size}_episodes_{args.num_episodes}.pickle")

with open(data_file, "wb") as f:
    pickle.dump(dataset, f)

#