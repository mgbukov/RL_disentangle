import numpy as np
import torch
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import os
import sys

file_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(file_path, os.pardir)))
from src.quantum_env import QuantumEnv
from src.quantum_state import VectorQuantumState
from src.quantum_state import random_quantum_state


# TODO 
# Use LateX font
# Use lower case 'q' instead of 'Q'

# TODO Figure 1
# Add dashed line inbetween actions - histogram bottom
# Add name of quantum state to bottom right
# Remove action names, add line overlay of qubits on which they don't act
# Histogram should be between gates
# Add probability text on histogram
# Skip columns that are 0
# Add label for policy (right or top)
# Remove all axes splines on axA + ylabels

# TODO Figure 3
# Remove rectangles
# Remove axes splines, keep only step index
# Add horizontal lines for each qubit
# Change gate notation
# Change entanglement notation
# Add average entanglement per step



PATH_4Q_AGENT = ''
PATH_5Q_AGENT = 'logs/5q_pGen_0.9_attnHeads_4_tLayers_4_ppoBatch_512_entReg_0.1_embed_256_mlp_512/agent.pt'
PATH_6Q_AGENT = ''


def figure3(initial_state):
    AW = 0.1   # action line: horizonal size
    AT = 4     # action line: thickness
    CR = 0.4   # circle radius (qubit)
    RW = 0.6   # rectangle width
    RH = 0.8   # rectangle height
    RT = 8     # rectangle text size
    SPA = 10   # steps per axes

   # Initialize environment
    num_qubits = int(np.log2(initial_state.size))
    shape = (2,) * num_qubits
    initial_state = initial_state.reshape(shape)
    env = QuantumEnv(num_qubits, 1, obs_fn='rdm_2q_mean_real')
    env.reset()
    env.simulator.states = np.expand_dims(initial_state, 0)

    # Load agent
    if num_qubits == 6:
        agent = torch.load(PATH_6Q_AGENT, map_location='cpu')
    elif num_qubits == 5:
        agent = torch.load(PATH_5Q_AGENT, map_location='cpu')
    elif num_qubits == 4:
        agent = torch.load(PATH_4Q_AGENT, map_location='cpu')
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()

    # Rollout a trajectory
    actions, entanglements, probabilities = [], [], []
    for i in range(30):
        ent = env.simulator.entanglements.copy()
        observation = torch.from_numpy(env.obs_fn(env.simulator.states))
        policy = agent.policy(observation).probs[0].cpu().numpy()
        a, p = np.argmax(policy), np.max(policy)
        actions.append(env.simulator.actions[a])
        entanglements.append(ent.ravel())
        probabilities.append(p)
        o, r, t, tr, i = env.step([a])
        if np.all(t):
            break

    # Initialize figure
    steps = len(actions)
    nrows = int(np.ceil(steps / SPA))
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(16,9*nrows), squeeze=False)

    # Draw qubit circles
    for ax in axs.flat:
        qubits_xs, qubits_ys = np.full(num_qubits, -1), np.arange(num_qubits)
        for x, y in zip(qubits_xs, qubits_ys):
            ax.add_patch(patches.Circle((x, y), CR, edgecolor='k', linewidth=5,
                                        fill=True, facecolor='#f4c3b8'))
            ax.text(x, y, f'Q{y+1}',
                    fontdict=dict(fontsize=16, horizontalalignment='center',
                                  verticalalignment='center'))

    # Draw actions & entanglements
    for i in range(len(actions)):
        k, j = divmod(i, SPA)
        ax = axs[k, 0]
        # Draw entanglements
        squares = []
        for n in range(num_qubits):
            e = entanglements[i][n]
            if e < env.epsi:
                color = 'k'
                t = ''
            elif env.epsi < e < 1e-2:
                color = 'tab:cyan'
                t = str(np.round(e * 1e2, 2))
            elif 1e-2 < e < 1e-1:
                color = 'tab:pink'
                t = str(np.round(e * 1e1, 2))
            else:
                color = 'tab:purple'
                t = str(np.round(e, 2))
            sq = patches.Rectangle(
                (j - RW/2, n - RH/2),
                RW, RH, color=color, alpha=0.4)
            ax.add_patch(sq)
            ax.text(j, n, t, fontdict=dict(fontsize=RT, horizontalalignment='center',
                                           verticalalignment='center'))

        # Draw action
        q0, q1 = actions[i]
        ax.plot([j + RW/2, j + RW/2 + AW], [q0, q0], color='k', linewidth=AT)
        ax.plot([j + RW/2, j + RW/2 + AW], [q1, q1], color='k', linewidth=AT)
        ax.plot([j + RW/2 + AW, j + RW/2 + AW], [q0, q1], color='k', linewidth=AT)
        ax.scatter([j + RW/2, j + RW/2], [q0, q1], s=5*AT, color='k')

        # Draw probability
        p = int(probabilities[i] * 100)
        ax.text(j + RW/2 + 0.2, (q0 + q1) / 2, f'{p}%',
                fontdict=dict(fontsize=10, rotation='vertical',
                              verticalalignment='center'))

    # Set limits
    for i, ax in enumerate(axs.flat, 1):
        ax.set_aspect('equal')
        xbegin, xend = (i-1) * SPA, min(i * SPA, steps)
        ax.set_xticks(list(range(xend - xbegin)), list(range(1 + xbegin, 1 + xend)))
        ax.set_yticks(list(range(num_qubits)), list(range(1, num_qubits + 1)))
        ax.set_xlim(-2, SPA + 1)
        ax.set_ylim(-1, num_qubits)
        ax.set_xlabel('Agent step')
        ax.set_ylabel('Qubit')

    return fig


def figure1(initial_state):

    CR = 0.35   # circle radius (qubit)
    RW = 0.1   # rectangle extra width
    RH = 0.6   # rectangle height
    PW = 0.9   # policy bar width
    RT = 16     # rectangle text size

   # Initialize environment
    num_qubits = int(np.log2(initial_state.size))
    shape = (2,) * num_qubits
    initial_state = initial_state.reshape(shape)
    env = QuantumEnv(num_qubits, 1, obs_fn='rdm_2q_mean_real')
    env.reset()
    env.simulator.states = np.expand_dims(initial_state, 0)

    # Load agent
    if num_qubits == 6:
        agent = torch.load(PATH_6Q_AGENT, map_location='cpu')
    elif num_qubits == 5:
        agent = torch.load(PATH_5Q_AGENT, map_location='cpu')
    elif num_qubits == 4:
        agent = torch.load(PATH_4Q_AGENT, map_location='cpu')
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()

    # Rollout a trajectory
    actions, entanglements, policies = [], [], []
    for i in range(30):
        ent = env.simulator.entanglements.copy()
        observation = torch.from_numpy(env.obs_fn(env.simulator.states))
        policy = agent.policy(observation).probs[0].cpu().numpy()
        a = np.argmax(policy)
        actions.append(env.simulator.actions[a])
        entanglements.append(ent.ravel())
        policies.append(policy)
        o, r, t, tr, i = env.step([a])
        if np.all(t):
            break

    # Initialize figure
    steps = len(actions)
    fig, axs = plt.subplots(1, 2, figsize=(16,5 + 1*steps))
    axA, axB = axs[0], axs[1]

    # Draw qubit circles
    qubits_ys, qubits_xs = np.full(num_qubits, -1), np.arange(num_qubits)
    for x, y in zip(qubits_xs, qubits_ys):
        axA.add_patch(patches.Circle((x, y), CR, edgecolor='k', linewidth=5,
                                    fill=True, facecolor='#f4c3b8', zorder=10))
        axA.text(x, y, f'Q{x+1}',
                fontdict=dict(fontsize=14, horizontalalignment='center',
                                verticalalignment='center', zorder=11))

    # Draw vertical lines
    for i in range(num_qubits):
        axA.plot([i, i], [-1, steps + 2], zorder=0, color='k', linewidth=6, alpha=0.5)
    
    # Draw actions & policies
    for n in range(steps):
        # Draw action
        q0, q1 = sorted(actions[n])
        rect = patches.Rectangle((q0 - RW, n), q1 - q0 + 2 * RW, RH,
                                 facecolor='#b8f4eb', edgecolor='k', linewidth=3,
                                 zorder=5)
        axA.add_patch(rect)
        axA.text(
            (q0 + q1) / 2, n + RH/2,
            f'{q0 + 1}, {q1 + 1}',
            fontdict=dict(horizontalalignment='center',
                          verticalalignment='center', fontsize=RT, zorder=6
        ))
        # Draw policy
        policy = policies[n].ravel()
        pmax = np.max(policy)
        axB.plot([-0.5, len(policy) + 0.5], [n, n], color='k', linewidth=1)
        for x, p in enumerate(policy):
            color = 'tab:red' if p == pmax else 'tab:blue'
            bar = patches.Rectangle((x - PW/2, n), PW, p * 0.8, facecolor=color)
            axB.add_patch(bar)
            q0, q1 = env.simulator.actions[x]
            if n > 0:
                continue
            axB.text(x, n - 0.15, f'({q0+1}, {q1+1})',
                     fontdict=dict(horizontalalignment='center', verticalalignment='center'))

    # Set limits
    axA.set_aspect('equal')
    # axB.set_aspect(axA.get_aspect())
    axB.set_ylim(-2, steps + 1)
    axA.set_ylim(-2, steps + 1)
    axB.set_xlim(-1, num_qubits * (num_qubits - 1) / 2 + 1)
    axB.set_axis_off()
    axA.xaxis.set_tick_params(labelbottom=False)
    axA.set_xticks([])
    axA.set_yticks(list(range(0, steps)), list(range(1, steps+1)))
    axA.spines['top'].set_visible(False)
    # axA.spines['right'].set_visible(False)
    # axA.spines['bottom'].set_visible(False)
    # axA.spines['left'].set_visible(False)
    return fig


if __name__ == '__main__':

    np.random.seed(5)
    initial_5q_states = {
            "|RR-R-R-R>": np.kron(
                random_quantum_state(q=2, prob=1.),
                np.kron(
                    np.kron(
                        random_quantum_state(q=1, prob=1.),
                        random_quantum_state(q=1, prob=1.),
                    ),
                    random_quantum_state(q=1, prob=1.),
                ),
            ).reshape((2,) * 5).astype(np.complex64),

            "|RR-RR-R>": np.kron(
                random_quantum_state(q=2, prob=1.),
                np.kron(
                    random_quantum_state(q=2, prob=1.),
                    random_quantum_state(q=1, prob=1.),
                ),
            ).reshape((2,) * 5).astype(np.complex64),

            "|RRR-R-R>": np.kron(
                random_quantum_state(q=3, prob=1.),
                np.kron(
                    random_quantum_state(q=1, prob=1.),
                    random_quantum_state(q=1, prob=1.),
                ),
            ).reshape((2,) * 5).astype(np.complex64),

            "|RRR-RR>": np.kron(
                random_quantum_state(q=3, prob=1.),
                random_quantum_state(q=2, prob=1.),
            ).reshape((2,) * 5).astype(np.complex64),

            "|RRRR-R>": np.kron(
                random_quantum_state(q=4, prob=1.),
                random_quantum_state(q=1, prob=1.),
            ).reshape((2,) * 5).astype(np.complex64),

            "|RRRRR>": random_quantum_state(q=5, prob=1.),
        }

    initial_state = random_quantum_state(5)
    fig = figure3(initial_state)
    fig.savefig('test_figure3.pdf')
    fig = figure1(initial_5q_states["|RRR-R-R>"])
    fig.suptitle("|RRR-R-R>")
    fig.savefig('test_figure1.pdf')
