"""
python3 test_beam_search.py -n 5 --beam_size 100 -t 10 --verbose
"""


import argparse
import os
import sys
import time
sys.path.append("..")

import numpy as np

from src.agents.expert import SearchExpert
from src.envs.rdm_environment import QubitsEnvironment
from src.infrastructure.beam_search import BeamSearch
from src.infrastructure.logging import logBarchart
from src.infrastructure.util_funcs import fix_random_seeds


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-n", "--num_qubits", dest="num_qubits", type=int,
                    help="Number of qubits in the quantum system", default=2)
parser.add_argument("--epsi", dest="epsi", type=float,
                    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("--beam_size", dest="beam_size", type=int,
                    help="Size of the beam for beam search", default=10)
parser.add_argument("-t", "--num_test", dest="num_test", type=int,
                    help="Number of iterations to run the test for", default=1)
parser.add_argument("--verbose", dest="verbose", action='store_true', default=False)
args = parser.parse_args()



np.set_printoptions(formatter={'all':lambda x: str(x)})
L = args.num_qubits
num_test = args.num_test

env = QubitsEnvironment(L, args.epsi, batch_size=1)
beam_search = BeamSearch(beam_size=args.beam_size)

tic = time.time()
# # Test single-qubit disentanglement.
# fix_random_seeds(args.seed)
# entropies = np.ndarray((num_test, L, L))
# path_lens = np.ndarray((num_test, L))
# for i in range(num_test):
#     env.set_random_states()
#     psi = env.states

#     for q0 in range(L):
#         path = beam_search.start(np.squeeze(psi,axis=0), env, qubit=q0,
#                                  num_iters=100, verbose=False)
#         env.states = psi
#         if args.verbose:
#             print("====================================================")
#             print(f"Trying to disentangle a single qubit: qubit {q0}")
#             print(f"found a path with length {len(path)}: {path}")
#             print(" entropy at step 0:", env.Entropy(env.states))
#         for idx, a in enumerate(path):
#             _ = env.step(a)
#             if args.verbose:
#                 print(f" entropy at step {idx+1}: {env.Entropy(env.states)}")
#         entropies[i][q0] = env.Entropy(env.states)[0]
#         path_lens[i][q0] = len(path)

# for i in range(L):
#     figname = f"/home/cacao-macao/Projects/RL_disentangle/logs/5qubits/single_qubit_disent_q{i}"
#     figtitle = f"Disentangling qubit {i}\nAverage entropy after {path_lens.mean(axis=0)[i]:.1f} steps"
#     logBarchart(figname, np.arange(L), entropies.mean(axis=0)[i], figtitle,
#         labels={},ylim=(0, 0.7))


# # Test entire system disentanglement.
# fix_random_seeds(args.seed)
# for _ in range(num_test):
#     env.set_random_states()
#     psi = env.states
#     path = beam_search.start(np.squeeze(psi, axis=0), env, num_iters=100, verbose=False)
#     env.states = psi
#     if args.verbose:
#         print("====================================================")
#         print(f"Trying to disentangle the entire system")
#         print(f"found a path with length {len(path)}: {path}")
#         print(" entropy at step 0:", env.Entropy(env.states))
#     for idx, a in enumerate(path):
#         _ = env.step(a)
#         if args.verbose:
#             print(f" entropy at step {idx+1}: {env.Entropy(env.states)}")


# Test entire system disentanglement qubit-by-qubit.
path_lens = []
s = SearchExpert(env, 100)
fix_random_seeds(args.seed)
for nt in range(num_test):
    print("i:", nt)
    env.set_random_states()
    psi = env.states
    states, actions = s.rollout(np.squeeze(psi, axis=0), num_iter=100, verbose=False)
    env.states = psi
    if args.verbose:
        print("====================================================")
        print(f"Trying to disentangle the entire system")
        print(f"found a path with length {len(actions)}: {actions}")
        print(" entropy at step 0:", env.Entropy(env.states))
    for idx, a in enumerate(actions):
        _ = env.step(a)
        if args.verbose:
            print(f" entropy at step {idx+1}: {env.Entropy(env.states)}")
    
    if not env.disentangled():
        print("ERROR: THE ENVIRONMENT IS NOT DISENTANGLED")
    else:
        print("OK")
    path_lens.append(len(actions))


toc = time.time()
print(f"{num_test} iters took {toc-tic:.3f} seconds")
print(f"average steps to disentangle: {np.array(path_lens).mean()}")

#

path = beam_search.start(np.squeeze(psi, axis=0), env, 0, verbose=False)
print(path)