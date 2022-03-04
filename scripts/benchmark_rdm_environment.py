import numpy as np
import pickle
import time
import os
import pathlib
import matplotlib.pyplot as plt
import sys

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.split(PATH)[0])
from src.envs.rdm_environment import QubitsEnvironment

BENCHMARK_NAME = 'benchmark-rdm-cython2'
NSTEPS = 100
REPEATS = 1
results = {}
Ls = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
Bs = (32, 64, 128, 256)

os.makedirs('data/benchmarks/', exist_ok=True)
if os.path.exists(f'data/benchmarks/{BENCHMARK_NAME}.pkl'):
    with open(f'data/benchmarks/{BENCHMARK_NAME}.pkl', mode='rb') as f:
        results = pickle.load(f)
else:
    for L in Ls:
        for B in Bs:
            print(f'Rollout with L={L}, B={B}...')
            E = QubitsEnvironment(num_qubits=L, batch_size=B)
            for _ in range(REPEATS):
                E.set_random_state(copy=False)
                tick = time.time()
                for a in np.random.uniform(0, E.num_actions, (NSTEPS, B)).astype(np.int32):
                    state, rewards, done = E.step2(a)
                tock = time.time()
                results.setdefault((L, B), []).append(tock - tick)
    print('Saving benchmark results...')
    with open(f'data/benchmarks/{BENCHMARK_NAME}.pkl', mode='wb') as f:
        pickle.dump(results, f)

print('Creating figure...')
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title(f'{BENCHMARK_NAME}')
ax.set_xlabel('batch size')
ax.set_ylabel('seconds')
markers = '.o^+sxDv'
linestyles = ['-', '--', '-.', ':']

for i, L in enumerate(Ls):
    xs, ys, errs = [], [], []
    for B in Bs:
        xs.append(B)
        ys.append(np.mean(results[(L, B)]))
        errs.append(np.std(results[(L, B)]))
    ax.errorbar(xs, ys, errs,
                marker=markers[i % len(markers)],
                linewidth=1,
                linestyle=linestyles[i % len(linestyles)],
                label=f'{L} qubits')

ax.set_xticks(Bs)
ax.set_ylim(0, 5)
ax.grid(True)
ax.legend(loc='upper left')
fig.savefig(f'data/benchmarks/{BENCHMARK_NAME}.pdf')