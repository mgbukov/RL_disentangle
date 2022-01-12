import argparse
import os
import pickle
import sys
import time
sys.path.append("..")

from tqdm import tqdm

from src.agents.expert import SearchExpert
from src.envs.rdm_environment import QubitsEnvironment


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-n", "--num_qubits", dest="num_qubits", type=int,
                    help="Number of qubits in the quantum system", default=2)
parser.add_argument("--beam_size", dest="beam_size", type=int,
                    help="Size of the beam for beam search", default=10)
parser.add_argument("--epsi", dest="epsi", type=float,
                    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("-i", "--num_iter", dest="num_iter", type=int,
                    help="Number of iterations to run the training for", default=1)
args = parser.parse_args()


# Run the expert to generate data.
env = QubitsEnvironment(args.num_qubits, epsi=args.epsi, batch_size=1)
expert = SearchExpert(env, args.beam_size)
dataset = {"states":[], "actions":[]}

print(f"Solving {args.num_iter} {args.num_qubits}-qubits systems")
tic = time.time()
for _ in tqdm(range(args.num_iter)):
    env.set_random_states()
    psi = env.states[0]
    states, actions = expert.rollout(psi, num_iter=1000, verbose=False)
    dataset["states"].append(states)
    dataset["actions"].append(actions)
toc = time.time()
print(f"Data generation took {toc-tic:.3f} seconds")


# Save the dataset.
log_dir = f"../data/{args.num_qubits}qubits/beam_size={args.beam_size}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(os.path.join(log_dir, f"{args.num_iter}_trajectories.pickle"), "wb") as f:
    pickle.dump(dataset, f)

#