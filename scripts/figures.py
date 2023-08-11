import itertools
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import sys

file_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(file_path, os.pardir)))
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'


PATH_4Q_AGENT = ''
PATH_5Q_AGENT = 'logs/5q_pGen_0.9_attnHeads_4_tLayers_4_ppoBatch_512_entReg_0.1_embed_256_mlp_512/agent.pt'
PATH_6Q_AGENT = ''


def figure3(initial_state, selected_actions=None):
    AW = 0.2   # action line: horizonal size
    AT = 4     # action line: thickness
    CR = 0.4   # circle radius (qubit)
    RW = 0.6   # rectangle width
    RH = 0.8   # rectangle height
    RT = 18    # rectangle text size
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
    max_steps = len(selected_actions) if selected_actions is not None else 30
    for i in range(max_steps):
        ent = env.simulator.entanglements.copy()
        observation = torch.from_numpy(env.obs_fn(env.simulator.states))
        policy = agent.policy(observation).probs[0].cpu().numpy()
        if selected_actions is None:
            a, p = np.argmax(policy), np.max(policy)
        else:
            a, p = selected_actions[i], policy[i]
        actions.append(env.simulator.actions[a])
        entanglements.append(ent.ravel())
        probabilities.append(p)
        o, r, t, tr, i = env.step([a], reset=False)
        if np.all(t):
            break

    # Append final entangments
    entanglements.append(env.simulator.entanglements.copy().ravel())

    # Initialize figure
    steps = len(actions)
    nrows = int(np.ceil(steps / SPA))
    fig, axs = plt.subplots(nrows, 1, figsize=(16,9*nrows), squeeze=False)

    # Draw qubit circles
    qubits_fontdict = dict(fontsize=20, horizontalalignment='center',
                           verticalalignment='center')
    avg_ent_fontdict = dict(fontsize=18, horizontalalignment='center',
                            verticalalignment='center', color='k')
    for ax in axs.flat:
        qubits_xs, qubits_ys = np.full(num_qubits, -1), np.arange(num_qubits)
        for x, y in zip(qubits_xs, qubits_ys):
            circle = patches.Circle((x, y), CR, edgecolor='k', linewidth=3,
                                    fill=True, facecolor='white')
            ax.add_patch(circle)
            ax.text(x, y, f'q{y+1}', fontdict=qubits_fontdict)
        # Draw average entanglement label
        ax.text(-1, -0.8, 'Avg. ent.', fontdict=avg_ent_fontdict)

    # Draw actions & entanglements
    entanglement_fontdict = dict(fontsize=RT, horizontalalignment='center',
                                 verticalalignment='center',
                                 weight='bold')
    for i in range(len(actions) + 1):
        k, j = divmod(i, SPA)
        ax = axs[k, 0]
        # Draw entanglements
        for n in range(num_qubits):
            e = entanglements[i][n] / np.log(2)
            if e < (env.epsi / np.log(2)):
                color = 'k'
                t = str(np.round(e * 1e3, 2))
            elif env.epsi < e < 1e-2:
                color = 'blue'
                t = str(np.round(e * 1e2, 2))
            elif 1e-2 < e < 1e-1:
                color = 'magenta'
                t = str(np.round(e * 1e1, 2))
            else:
                color = 'red'
                t = str(np.round(e, 2))
            bg = patches.Rectangle(
                (j - RW/2, n - RH/2),
                RW, RH, facecolor='white', zorder=1)
            ax.add_patch(bg)
            # Add circles behind text
            circle = patches.Circle((j, n), RW/2, facecolor=color, alpha=0.1, zorder=2)
            ax.add_patch(circle)
            entanglement_fontdict.update(color=color)
            ax.text(j, n, t, fontdict=entanglement_fontdict)
        # Draw average entanglement
        avg_e = np.mean(entanglements[i]) / np.log(2)
        ax.text(j, -0.8, str(np.round(avg_e, 3)),
                fontdict=dict(fontsize=RT, horizontalalignment='center',
                              verticalalignment='center', color='k'))

        if i == len(actions):
            continue

        # Draw action
        q0, q1 = actions[i]
        ax.plot([j + RW/2 + AW, j + RW/2 + AW], [q0, q1], color='k', linewidth=AT)
        ax.scatter([j + RW/2 + AW, j + RW/2 + AW], [q0, q1], s=120, color='k')

        # Draw probability
        p = int(probabilities[i] * 100)
        text_x = j + RW/2 - AW/2
        text_y = (q0 + q1) / 2
        if text_y - int(text_y) < 0.4:
            text_y += 0.5
        ax.text(text_x, text_y, f'{p}%', fontdict=dict(fontsize=18,
                                                       rotation='vertical',
                                                       verticalalignment='center'))

    for i, ax in enumerate(axs.flat, 1):
        ax.set_aspect('equal')
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        # Set limits
        ax.set_xlim(-2, SPA + 1)
        ax.set_ylim(-2, num_qubits)
        ax.set_xlabel('Agent step', fontsize=18)
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Add horizontal arrow indicating agent timesteps
        xbegin, xend = (i-1) * SPA, min(i * SPA, steps)
        # Add horizontal line per qubit
        for q in range(num_qubits):
            ax.plot([-1, xend - xbegin], [q, q], linewidth=4, color='k', zorder=-10)
        xticks = np.arange(0.5, 0.5 + xend - xbegin)
        xticklabels = (i-1)*SPA + xticks.astype(np.int32) + 1
        ax.plot(xticks, [-1.5] * len(xticks), markevery=1, marker='|',
                markersize=10, markeredgewidth=2, linewidth=0.5, color='k')
        ax.arrow(0.5, -1.5, (xend - xbegin), 0, width=0.01, head_width=0.1, color='k')
        # Add text labels per step
        for x, lab in zip(xticks, xticklabels):
            ax.text(x, -1.8, str(lab), fontsize=18, horizontalalignment='center')

    return fig


def figure1(initial_state, state_name=''):

    CR = 0.45   # circle radius (qubit)
    RW = 0.1    # rectangle extra width
    RH = 0.6    # rectangle height
    PW = 0.9    # policy bar width
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

    policies = np.array(policies)
    policy_row_max = np.max(policies, axis=0)
    action_labels = list(itertools.combinations(range(num_qubits), 2))
    action_labels = np.array(action_labels)[policy_row_max > 0.05]
    policies = policies[:, policy_row_max > 0.05]

    righthalf_xmin = num_qubits + 2
    righthalf_xmax = righthalf_xmin + policies.shape[1]

    # Initialize figure
    steps = len(actions)
    fig, ax = plt.subplots(1, figsize=(16,5 + 1*steps))

    # Draw qubit circles
    qubits_ys, qubits_xs = np.full(num_qubits, -2), np.arange(num_qubits)
    for x, y in zip(qubits_xs, qubits_ys):
        ax.add_patch(patches.Circle((x, y), CR, edgecolor='k', linewidth=2,
                                    fill=True, facecolor='white', zorder=10))
        ax.text(x, y, f'q{x+1}',
                fontdict=dict(fontsize=16, horizontalalignment='center',
                                verticalalignment='center', zorder=11))

    # Draw vertical lines
    for i in range(num_qubits):
        ax.plot([i, i], [-2, steps + 2], zorder=0, color='k', linewidth=6)
    
    # Draw actions & policies
    for n in range(steps):
        # Draw action
        q0, q1 = sorted(actions[n])
        rect = patches.Rectangle((q0 - RW, 2 * n), q1 - q0 + 2 * RW, RH,
                                 facecolor='#85caff', edgecolor='k', linewidth=2,
                                 zorder=5)
        ax.add_patch(rect)
        for q in range(q0 + 1, q1):
            ax.plot([q, q], [2 * n, 2 * n + RH], color='k',
                    linewidth=6, zorder=6)
        # Draw horizontal line connecting policy and gate
        ax.plot([-1, righthalf_xmin - PW], [2*n - 1, 2*n - 1],
                linestyle='--', color='k')

        # Draw policy
        policy = policies[n].ravel()
        pmax = np.max(policy)
        ax.plot([righthalf_xmin - PW/2, righthalf_xmax], [2*n - 1, 2*n - 1],
                color='k', linewidth=2)
        ax.text(righthalf_xmax, 2 * n, f'$\pi_{n}$', fontdict=dict(fontsize=18))
        for x, p in enumerate(policy):
            color = 'tab:red' if p == pmax else 'tab:blue'
            x_coord = righthalf_xmin + x
            y_coord = 2*n - 1
            bar = patches.Rectangle((x_coord - PW/2, y_coord), PW, p * 0.9,
                                    facecolor=color)
            ax.add_patch(bar)
            q0, q1 = action_labels[x]
            ax.text(x_coord, y_coord - 0.3, f'({q0+1}, {q1+1})',
                     fontdict=dict(horizontalalignment='center',
                                   verticalalignment='center', fontsize=16))
            ax.text(x_coord, y_coord + p + 0.1, f'{int(p * 100)}%',
                    fontdict=dict(horizontalalignment='center', fontsize=14))

    # Add state name text
    ax.text(2, -3.5, state_name,
            fontdict=dict(fontsize=18, horizontalalignment='center'))
    # Set limits
    ax.set_aspect('equal')
    ax.set_ylim(-4, 2 * steps + 1)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
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
    fig = figure1(initial_5q_states["|RRR-RR>"], r'RRR-RR')
    fig.savefig('test_figure1.pdf')
