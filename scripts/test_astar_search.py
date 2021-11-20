import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import sys
import time

FILEPATH = os.path.abspath(str(pathlib.Path(__file__).resolve()))
PROJECT_PATH = str(pathlib.Path(FILEPATH).parent.parent.resolve())
sys.path.append(PROJECT_PATH)
from tree_search import do_astar_search
from src.envs.rdm_environment import (QubitsEnvironment,
                                      _random_pure_state,
                                      _ent_entropy)

np.random.seed(34)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=10)

parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, default=5, action='store')
parser.add_argument('-N', type=int, default=1000, action='store')
parser.add_argument('--max_iter', '-m', type=int, default=500, action='store')
args = parser.parse_args()

L = args.L
N = args.N
MAX_ITER = args.max_iter

solved_states = []
solved_paths = []
unsolved_paths = []
unsolved_states = []
path_entropies = []

tick = time.time()
for n in range(N):
    print(f'Starting search in state {n+1}/{N}')
    psi = _random_pure_state(L)
    final_state, final_path = do_astar_search(psi, MAX_ITER)
    E = QubitsEnvironment(L, epsi=1e-3)
    E.state = psi
    path_entropy = [E.Entropy(E._state)]
    for a in final_path:
        E.state = E.next_state(a)
        path_entropy.append(E.Entropy(E._state))
    path_entropies.append(path_entropy)
    if E.disentangled()[0]:
        solved_paths.append(final_path)
        solved_states.append(final_state)
    else:
        unsolved_paths.append(final_path)
        unsolved_states.append(final_state)
tock = time.time()

total_time = int(tock - tick)
hours = total_time // 3600
minutes = (total_time - (hours * 60)) // 60
seconds = total_time % 60


os.chdir(PROJECT_PATH)
SAVEPATH = f'data/astar-{L}qubits-{N}iter/'
os.makedirs(SAVEPATH, exist_ok=True)

print('Saving output stats and states...')
with open(os.path.join(SAVEPATH, 'output.txt'), mode='w') as f:
    print(f'N = {N}\nL = {L}', file=f)
    print(f'Solved = {len(solved_paths)}', file=f)
    print(f'Iteration limit = {MAX_ITER}', file=f)
    shortest = min(map(len, solved_paths))
    longest = max(map(len, solved_paths))
    print(f'Minimum solution length = {shortest}', file=f)
    print(f'Maximum solution length = {longest}', file=f)
    print(f'Elapsed time = {hours:2>0}h:{minutes:2>0}m:{seconds:2>0}s', file=f)

np.save(os.path.join(SAVEPATH, 'solved-states.npy'), np.array(solved_states))
np.save(os.path.join(SAVEPATH, 'unsolved-states.npy'), np.array(unsolved_states))
np.save(os.path.join(SAVEPATH, 'solved-paths.npy'), np.array(solved_paths))
np.save(os.path.join(SAVEPATH, 'unsolved-paths.npy'), np.array(unsolved_paths))

with open(os.path.join(SAVEPATH, 'unique-solved-paths.txt'), mode='w') as f:
    for p in sorted(set(map(tuple, solved_paths)), key=len):
        print(p, file=f)

with open(os.path.join(SAVEPATH, 'unique-unsolved-paths.txt'), mode='w') as f:
    for p in sorted(set(map(tuple, unsolved_paths)), key=len):
        print(p, file=f)


# Distribution of single qubit entropies in unsolved states
# -----------------------------------------------------------------------------
print('Plotting unsolved states...')
if unsolved_states:
    unsolved_entropies = []
    for state in unsolved_states:
        unsolved_entropies.append([_ent_entropy(state, [i], L) for i in range(L)])
    unsolved_entropies = np.array(unsolved_entropies)
    with open(os.path.join(SAVEPATH, 'unsolved-entropies.txt'), mode='w') as f:
        for ent in unsolved_entropies:
            print(ent.round(4).ravel(), file=f)
    fig, axs = plt.subplots(2, 3,
                            sharex=True, sharey=True, figsize=(12, 8),
                            gridspec_kw=dict(hspace=0.25, wspace=0.25))
    fig.suptitle('Distribution of single qubit entropies in unsolved states')
    for i in range(L):
        axs.flat[i].set_title(f'Qubit {i}')
        axs.flat[i].hist(unsolved_entropies[:, i].ravel(), bins=20)
    fig.savefig(os.path.join(SAVEPATH, 'unsolved-entropy-distribution.pdf'))


# Evolution of single vs rest qubit entropies
# -----------------------------------------------------------------------------
print('Plotting solved states...')
fig, axs = plt.subplots(2, 3,
                        sharex=True, sharey=True, figsize=(12, 8),
                        gridspec_kw=dict(hspace=0.25, wspace=0.25))
fig.suptitle('Evolution of single qubit entropies in solved states')
for i in range(L):
    axs.flat[i].set_title(f'Qubit {i}')
    for p in path_entropies:
        p = np.array(p)
        axs.flat[i].set_yscale('log')
        axs.flat[i].plot(p[:, 0, i], alpha=0.2, color='tab:blue', linewidth=1)
fig.savefig(os.path.join(SAVEPATH, 'solved-entropy-evolution.pdf'))

print('Done!')
