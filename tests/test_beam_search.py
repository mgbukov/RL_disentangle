"""
python3 test_beam_search.py -n 5 --beam_size 100 -t 10 --verbose
"""

import argparse
import sys
import time

import numpy as np

sys.path.append("..")
from common import (environment_generator, make_semientagled_state,
                    pack_complex_state, unpack_complex_state)

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


# Global constants
EPSI = args.epsi
SEED = args.seed
L = args.num_qubits


np.set_printoptions(formatter={'all':lambda x: str(x)})
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

def test_disent_first_qubit():
    _test_disent_first_qubit(4, 100, 10)
    _test_disent_first_qubit(5, 100, 10)
    _test_disent_first_qubit(6, 100, 10)
    _test_disent_first_qubit(7, 100, 10)
    _test_disent_first_qubit(8, 100, 10)

def _test_disent_first_qubit(nqubits=None, beam_size=100, repeats=10):
    """ Tests if Beam Search disentangles the first (leftmost) qubit."""
    beam_search = BeamSearch(beam_size)
    for env in environment_generator(nqubits, 1, SEED, EPSI):
        for _ in range(repeats):
            print(env.L)
            env.set_random_states()
            psi = env.states.copy()
            solution = beam_search.start(env.states[0], env, 0, verbose=False)
            assert solution
            env.states = psi
            for a in solution:
                env.step(a)
            assert env.entropy()[0,0] <= env.epsi


def test_disent_single_qubit(nqubits=4, beam_size=100, repeats=10):
    """ Tests if Beam Search disentangles i-th qubit if the (i - 1)-th are
    already disentangled. """
    fix_random_seeds(SEED)
    env = QubitsEnvironment(nqubits, epsi=EPSI, batch_size=1)
    beam_search = BeamSearch(beam_size)
    for _ in range(repeats):
        for i in range(nqubits - 1):
            psi = make_semientagled_state(nqubits, i).reshape((2,) * nqubits)
            env.states = np.expand_dims(psi, axis=0)
            env.set_random_states()
            entropies = env.entropy()[0]
            # assert np.all(entropies[:i+1] < EPSI)
            path = beam_search.start(env.states[0], env, qubit=i+1, verbose=False)
            env.states = np.expand_dims(psi, axis=0)
            for a in path:
                env.step(a)
            # print('\tAfter Beam Search:', np.round(env.entropy()[0], 4))
            assert env.entropy()[0, i+1] <= env.epsi


def test_disent_all_qubits():
    """ Tests if Search Expert disentangles all qubits in a quantum system."""
    _test_disent_all_qubits(4, repeats=10)
    _test_disent_all_qubits(5, repeats=10)
    _test_disent_all_qubits(6, repeats=10)
    _test_disent_all_qubits(7, repeats=10)

def _test_disent_all_qubits(nqubits=None, beam_size=100, repeats=10):
    envgen = environment_generator(nqubits, 1, SEED, EPSI)
    for env in envgen:
        expert = SearchExpert(env)
        for _ in range(repeats):
            env.set_random_states()
            psi = env.states.copy()
            states, solution = expert.rollout(psi[0].copy())
            if solution is None:
                continue
            terminal = np.expand_dims(pack_complex_state(states[-1]), 0)
            assert env.Disentangled(terminal, epsi=env.epsi)[0]
            assert np.all(pack_complex_state(states[0]).ravel() == psi.ravel())
            env.states = psi.copy()
            for i, a in enumerate(solution, 1):
                next_state, _, _ = env.step(a)
                s = pack_complex_state(states[i])
                try:
                    overlap = np.abs(s.conj() @ next_state[0].ravel()) ** 2
                    assert np.isclose(overlap, 1)
                except AssertionError as ex:
                    print(np.round(s, 4))
                    print(np.round(next_state[0].ravel(), 4))
                    raise ex
            try:
                assert env.disentangled()[0]
            except:
                env.states = psi.copy()
                for a in solution:
                    env.step(a)
                raise ex
            print('ok', env.L, len(solution))


if __name__ == '__main__':
    test_disent_first_qubit()
    test_disent_single_qubit(args.num_qubits, repeats=args.num_test)
    test_disent_all_qubits()

toc = time.time()
print(f"{num_test} iters took {toc-tic:.3f} seconds")

#