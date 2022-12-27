import itertools
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


RE_ITERATION = r'\s*Iteration \(?(\d+).*'
RE_RETURN = r'\s*Mean return:\s*(\S+)'
RE_ENTROPY = r'\s*Mean final entropy:\s*(\S+)'
RE_95_ENTROPY = r'\s*95 percentile entropy:\s*(\S+)'
RE_STEPS = r'\s*Avg steps to disentangle:\s*(\S+)'
RE_SOLVED_T = r'Solved trajectories:\s*(\d+) / (\d+)'
RE_SOLVED_S = r'Solved states:\s*(\d+) / (\d+).*'


def extract_stats(text, istest=False):
    iteration = int(re.search(RE_ITERATION, text).group(1))
    returns = float(re.search(RE_RETURN, text).group(1))
    entropy = float(re.search(RE_ENTROPY, text).group(1))
    entropy95 = float(re.search(RE_95_ENTROPY, text).group(1))
    steps = float(re.search(RE_STEPS, text).group(1))
    if istest:
        solved = int(re.search(RE_SOLVED_S, text).group(1))
        total = int(re.search(RE_SOLVED_S, text).group(2))
    else:
        solved = int(re.search(RE_SOLVED_T, text).group(1))
        total = int(re.search(RE_SOLVED_T, text).group(2))
    return {
        'iteration':    iteration,
        'return':       returns,
        'entropy':      entropy,
        'entropy95':    entropy95,
        'steps':        steps,
        'solved':       solved / total,
        'test':         istest,
    }


def read_text_logs(filepath):
    with open(filepath) as f:
        trainlog = '\n'.join(line.rstrip() for line in f)
        stats = []
        for section in trainlog.split('\n\n'):
            if section.lstrip().startswith('Iteration'):
                istest = 'Testing agent accuracy for ' in section
                stats.append(extract_stats(section, istest))
    return stats


# Read text logs
# trainlog1 = read_text_logs('logs/5qubits-chipoff/chipoff-train-25steps-1e-3.log')
# trainlog2 = read_text_logs('logs/5qubits-chipoff/chipoff-train-40steps-1e-3.log')
# trainlog3 = read_text_logs('logs/5qubits-chipoff/chipoff-train-40steps-1e-6.log')
# trainlog4 = read_text_logs('logs/5qubits/rdm-train-40steps-1e-3.log')
trainlog1 = read_text_logs('experiments/chipoff_pretrained.log')
trainlog2 = read_text_logs('experiments/chipoff_pretrained_1e-6.log')
trainlog3 = read_text_logs('experiments/chipoff_pretrained_reward.log')

trainstats = [
    trainlog1,
    trainlog2,
    trainlog3,
    # trainlog4
]
names = [
    # 'chipoff, 25 steps, 1e-3',
    # 'chipoff, 40 steps, 1e-3',
    # 'chipoff, 40 steps, 1e-6',
    # 'rdm, 40 steps, 1e-3'
    'chipoff, 40 steps, 1e-3',
    'chipoff, 40 steps, 1e-6',
    'chipoff, 40 steps, 1e-3, reward=10'
]

# SAVEDIR = 'data/rdm-chipoff-pg-compare/'
SAVEDIR = 'data/chipoff-pretrained'
os.makedirs(SAVEDIR, exist_ok=True)


# Entropy figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_yscale('log')
ax.set_ylim(1e-6, 1e-1)
colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
for name, stats in zip(names, trainstats):
    color = next(colors)
    # Train
    xs = [x['iteration'] for x in stats if not x['test']]
    ys = [x['entropy'] for x in stats if not x['test']]
    ax.plot(xs, ys, color=color, label=name)
    # Test
    xs = [x['iteration'] for x in stats if x['test']]
    ys = [x['entropy'] for x in stats if x['test']]
    ax.plot(xs, ys, color=color, linestyle='--', marker='.')
ax.legend(loc='lower right')
ax.set_xlabel('iteration')
ax.set_ylabel('mean entropy')
# ax.set_title('PG Comparison on RDM and Chipoff Environments\nMean Entropy')
ax.set_title('IL + PG, Chipoff Environment\nMean Entropy')
fig.savefig(os.path.join(SAVEDIR, 'entropy.png'), dpi=120)


# Returns figure
fig, ax = plt.subplots(figsize=(12, 6))
colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
for name, stats in zip(names, trainstats):
    color = next(colors)
    # Train
    xs = [x['iteration'] for x in stats if not x['test']]
    ys = [x['return'] for x in stats if not x['test']]
    ax.plot(xs, ys, color=color, label=name)
    # Test
    xs = [x['iteration'] for x in stats if x['test']]
    ys = [x['return'] for x in stats if x['test']]
    ax.plot(xs, ys, color=color, linestyle='--', marker='.')
ax.legend(loc='lower right')
ax.set_xlabel('iteration')
ax.set_ylabel('mean return')
ax.grid(True)
# ax.set_title('PG Comparison on RDM and Chipoff Environments\nMean Returns')
ax.set_title('IL + PG, Chipoff Environment\nMean Return')
fig.savefig(os.path.join(SAVEDIR, 'return.png'), dpi=120)


# Solved figure
fig, ax = plt.subplots(figsize=(12, 6))
colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
for name, stats in zip(names, trainstats):
    color = next(colors)
    # Train
    xs = [x['iteration'] for x in stats if not x['test']]
    ys = [100 * x['solved'] for x in stats if not x['test']]
    ax.plot(xs, ys, color=color, label=name)
    # Test
    xs = [x['iteration'] for x in stats if x['test']]
    ys = [100 * x['solved'] for x in stats if x['test']]
    ax.plot(xs, ys, color=color, linestyle='--', marker='.')
ax.legend(loc='lower right')
ax.set_xlabel('iteration')
ax.set_ylabel('% solved')
ax.set_yticks(list(range(0,101, 5)))
ax.grid('on')
# ax.set_title('PG Comparison on RDM and Chipoff Environments\nSolved States')
ax.set_title('IL + PG, Chipoff Environment\nSolved States')
fig.savefig(os.path.join(SAVEDIR, 'solved.png'), dpi=120)


# Steps figure
fig, ax = plt.subplots(figsize=(12, 6))
colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
for name, stats in zip(names, trainstats):
    color = next(colors)
    # Train
    xs = [x['iteration'] for x in stats if not x['test']]
    ys = [x['steps'] for x in stats if not x['test']]
    ax.plot(xs, ys, color=color, label=name)
    # Test
    xs = [x['iteration'] for x in stats if x['test']]
    ys = [x['steps'] for x in stats if x['test']]
    ax.plot(xs, ys, color=color, linestyle='--', marker='.')
ax.legend(loc='upper right')
ax.set_xlabel('iteration')
ax.set_ylabel('# steps to solve')
lo, hi = ax.get_ylim()
ax.set_yticks(list(range(int(lo), int(hi), 1)))
ax.grid(True)
# ax.set_title('PG Comparison on RDM and Chipoff Environments\nSolution Length')
ax.set_title('IL + PG, Chipoff Environment\nSolution Length')
fig.savefig(os.path.join(SAVEDIR, 'steps.png'), dpi=120)


# Entropy 95 figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_yscale('log')
colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
for name, stats in zip(names, trainstats):
    color = next(colors)
    # Train
    xs = [x['iteration'] for x in stats if not x['test']]
    ys = [x['entropy95'] for x in stats if not x['test']]
    ax.plot(xs, ys, color=color, label=name)
    # Test
    xs = [x['iteration'] for x in stats if x['test']]
    ys = [x['entropy95'] for x in stats if x['test']]
    ax.plot(xs, ys, color=color, linestyle='--', marker='.')
ax.legend(loc='lower right')
ax.set_xlabel('iteration')
ax.set_ylabel('entropy')
# ax.set_title('PG Comparison on RDM and Chipoff Environments\n95% Entropy')
ax.set_title('IL + PG, Chipoff Environment\n95% Entropy')
fig.savefig(os.path.join(SAVEDIR, 'entropy95.png'), dpi=120)
