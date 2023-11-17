import itertools
import json
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import pickle
import sys
from matplotlib.gridspec import GridSpec

file_path = os.path.split(os.path.abspath(__file__))[0]
project_dir = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.append(project_dir)
from src.environment_loop import test_agent
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state
from search import GreedyAgent, RandomAgent

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'


PATH_4Q_AGENT = os.path.join(project_dir, "logs/4q_10000_iters_haar_unif2_1024envs/agent.pt")
PATH_5Q_AGENT = os.path.join(project_dir, "logs/5q_20000_iters_haar_unif2_128envs/agent.pt")
PATH_6Q_AGENT = os.path.join(project_dir, "logs/6q_4000iters_haar_unif3_512envs_seed7_3rd/agent.pt")


def str2state(string_descr):
    psi = np.array([1.0], dtype=np.complex64)
    nqubits = 0
    for pair in string_descr.split('-'):
        q = pair.count('R')
        nqubits += q
        psi = np.kron(psi, random_quantum_state(q=q, prob=1.))
    return psi.reshape((2,) * nqubits)


def str2latex(string_descr):
    numbers = "123456789"
    Rs = string_descr.split("-")
    name = []
    i = 0
    for r in Rs:
        s = r'|R_{' + f'{numbers[i:i+len(r)]}' + r'}\rangle'
        name.append(r'\mathrm{' + s + '}')
        i += len(r)
    return "$" + ''.join(name) + "$"

def rollout(initial_state, max_steps=30):
    """
    Returns action names, entanglements and policy probabilities for each step.
    """

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
    else:
        raise ValueError(f'Cannot find agent for {num_qubits}-qubit system')
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()

    # Rollout a trajectory
    actions, entanglements, probabilities = [], [], []
    for i in range(max_steps):
        ent = env.simulator.entanglements.copy()
        np.set_printoptions(precision=3, suppress=True)
        print(ent[0])
        observation = torch.from_numpy(env.obs_fn(env.simulator.states))
        probs = agent.policy(observation).probs[0].cpu().numpy()
        a = np.argmax(probs)
        actions.append(env.simulator.actions[a])
        entanglements.append(ent.ravel())
        probabilities.append(probs)
        o, r, t, tr, i = env.step([a], reset=False)
        if np.all(t):
            break
    # Append final entangments
    print(env.simulator.entanglements[0])
    assert np.all(env.simulator.entanglements <= env.epsi)
    entanglements.append(env.simulator.entanglements.copy().ravel())

    return np.array(actions), np.array(entanglements), np.array(probabilities)


def peek_policy(state):
    """Returns the agent probabilites for this state."""
    _, _, probabilities = rollout(state, max_steps=1)
    return probabilities[0]


def figure1a():

    # /// USER CONSTANTS
    num_qubits = 3
    nsteps = 2
    QCY = -1        # Y coordinate of qubits' circles
    QCX = 0         # Min X coordinate of qubits' circles
    QCR = 0.35      # radius of qubits' circles
    QFS = 13        # fontsize of qubits' text
    QLW = 1.5       # linewidth of qubits' circles
    WLW = 0.6       # linewidth of qubit wires
    GLW = 2         # gate wire linewidth
    GSS = 100       # gate wire connection scatter size

    # /// DERIVED LAYOUT CONSTANTS
    WIRES_BOTTOM = QCY + QCR
    WIRES_TOP = QCY + nsteps + 0.5
    FIGSIZE = (6, 2)

    # Initialize figure
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes((0.30, -0.05, 0.5, 1.2))
    # Initialize fontdict
    textdict = dict(fontsize=QFS, ha='center',
                    va='center', zorder=11)

    # Draw qubit circles with Y=`QCY`, X=[`QCX`, `QCX` + `num_qubits`]
    qubits_xs = np.arange(QCX, QCX + num_qubits)
    qubits_ys = np.full(num_qubits, QCY)
    for x, y in zip(qubits_xs, qubits_ys):
        ax.add_patch(patches.Circle((x, y), QCR, edgecolor='k', linewidth=QLW,
                                    fill=True, facecolor='white', zorder=10))
        ax.text(x, y, f'$q_{x+1}$', fontdict=textdict)

    # Draw base wires for gates (starting from qubit circles)
    for i in range(num_qubits):
        ax.plot([QCX + i, QCX + i], [WIRES_BOTTOM, WIRES_TOP],
                zorder=0, color='k', alpha=0.5, linewidth=WLW)

    # Draw gate wires for gates
    ax.scatter([0, 1], [0, 0], s=GSS, color='k')
    ax.plot([0, 1], [0, 0], linewidth=GLW, color='k')
    ax.text(0.5, 0.2, r"$U^{(1,2)}$", fontdict=textdict)
    #
    ax.scatter([1, 2], [1, 1], s=GSS, color='k')
    ax.plot([1, 2], [1, 1], linewidth=GLW, color='k')
    ax.text(1.5, 1.2, r"$U^{(2,3)}$", fontdict=textdict)

    # Draw text
    textdict.update(ha='left')
    ax.text(QCX + num_qubits - .5, -0.5, r"$|\psi_{1,2,3}\rangle$",
            fontdict=textdict)
    ax.text(QCX + num_qubits - .5, 0.5, r"$|\psi_1\rangle|\psi_{2,3}\rangle$",
            fontdict=textdict)
    ax.text(QCX + num_qubits - .5, 1.5,
            r"$|\psi_1\rangle|\psi_2\rangle|\psi_3\rangle$", fontdict=textdict)

    # Config axes
    ax.set_aspect(1.0)
    ax.set_ylim(-2, WIRES_TOP + 1)
    ax.set_xlim(QCX - 0.5, QCX + num_qubits + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return fig


def figure1bd():

    # /// USER CONSTANTS
    num_qubits = 4
    nsteps = 4
    QCY = -1        # Y coordinate of qubits' circles
    QCX = 0         # Min X coordinate of qubits' circles
    QCR = 0.40      # radius of qubits' circles
    QFS = 15        # fontsize of qubits' text
    QLW = 1.5       # linewidth of qubits' circles
    WLW = 0.6       # linewidth of qubit wires
    GLW = 2         # gate wire linewidth
    GSS = 100       # gate wire connection scatter size

    # /// DERIVED LAYOUT CONSTANTS
    WIRES_BOTTOM = QCY + QCR
    WIRES_TOP = QCY + nsteps + 0.5
    FIGSIZE = (6, 3.2)

    # Initialize figure
    fig = plt.figure(figsize=FIGSIZE, layout=None)
    axl = fig.add_axes((-0.06, 0.0, 0.65, 1.))
    axr = fig.add_axes((0.55, 0.0, 0.65, 1.))
    axs = (axl, axr)
    # Initialize fontdict
    textdict = dict(fontsize=QFS - 2, ha='center',
                    va='center', zorder=11)

    # Draw qubit circles with Y=`QCY`, X=[`QCX`, `QCX` + `num_qubits`]
    qubits_xs = np.arange(QCX, QCX + num_qubits)
    qubits_ys = np.full(num_qubits, QCY)
    for x, y in zip(qubits_xs, qubits_ys):
        for ax in axs:
            ax.add_patch(
                patches.Circle((x, y), QCR, edgecolor='k', linewidth=QLW,
                                fill=True, facecolor='white', zorder=10)
            )
            ax.text(x, y, f'$q_{x+1}$', fontdict=textdict)

    # Draw base wires for gates (starting from qubit circles)
    for i in range(num_qubits):
        for ax in axs:
            ax.plot([QCX + i, QCX + i], [WIRES_BOTTOM, WIRES_TOP],
                    zorder=0, color='k', alpha=0.5, linewidth=WLW)

    # Draw gate wires for gates
    axs[0].scatter([0, 1, 2, 3], [0, 0, 0, 0], s=GSS, color='k')
    axs[0].plot([0, 1], [0, 0], linewidth=GLW, color='k')
    axs[0].plot([2, 3], [0, 0], linewidth=GLW, color='k')
    axs[0].text(0.5, 0.2, r"$U^{(1,2)}$", fontdict=textdict)
    axs[0].text(2.5, 0.2, r"$U^{(3,4)}$", fontdict=textdict)
    #
    axs[0].scatter([0,2], [1,1], s=GSS, color='k')
    axs[0].plot([0,2], [1,1], linewidth=GLW, color='k')
    axs[0].text(0.5, 1.2, r"$U^{(1,3)}$", fontdict=textdict)
    #
    axs[0].scatter([1,3], [2,2], s=GSS, color='k')
    axs[0].plot([1,3], [2,2], linewidth=GLW, color='k')
    axs[0].text(1.5, 2.2, r"$U^{(1,3)}$", fontdict=textdict)
    #
    axs[0].scatter([2, 3], [3, 3], s=GSS, color='k')
    axs[0].plot([2, 3], [3, 3], linewidth=GLW, color='k')
    axs[0].text(2.5, 3.2, r"$U^{(3,4)}$", fontdict=textdict)
    #
    axs[1].scatter([0, 1, 2, 3], [0, 0, 0, 0], s=GSS, color='k')
    axs[1].plot([0, 1], [0, 0], linewidth=GLW, color='k')
    axs[1].plot([2, 3], [0, 0], linewidth=GLW, color='k')
    axs[1].text(0.5, 0.2, r"$U^{(1,2)}$", fontdict=textdict)
    axs[1].text(2.5, 0.2, r"$U^{(3,4)}$", fontdict=textdict)
    # CNOT gate
    axs[1].scatter([2], [1], s=GSS, color='k',)
    axs[1].scatter([0], [1], s=GSS, color='w', edgecolors='k', linewidths=QLW)
    axs[1].scatter([0], [1], s=GSS, color='k', marker='+', linewidths=QLW)
    axs[1].plot([0.14, 2], [1,1], linewidth=GLW, color='k')
    axs[1].text(0.2, 1.2, r"$\mathrm{CNOT^{(1,3)}}$", fontdict=dict(
        fontsize=QFS-2, ha='left', va='center'))
    # CNOT Gate
    axs[1].scatter([3], [2], s=GSS, color='k',)
    axs[1].scatter([1], [2], s=GSS, color='w', edgecolor='k', linewidths=QLW)
    axs[1].scatter([1], [2], s=GSS, color='k', marker='+', linewidths=QLW)
    axs[1].plot([1.14, 3], [2,2], linewidth=GLW, color='k')
    axs[1].text(1.2, 2.2, r"$\mathrm{CNOT^{(2,4)}}$", fontdict=dict(
        fontsize=QFS-2, ha='left', va='center'))
    #
    axs[1].scatter([2, 3], [3, 3], s=GSS, color='k')
    axs[1].plot([2, 3], [3, 3], linewidth=GLW, color='k')
    axs[1].text(2.5, 3.2, r"$U^{(3,4)}$", fontdict=textdict)

    # Draw text
    axs[0].text(QCX + num_qubits + 1.4, -0.5,
            r"$|\psi_{1,2,3,4}\rangle$", fontdict=textdict)
    axs[0].text(QCX + num_qubits + 1.4, 1.5,
            r"$|\psi_1\rangle|\psi_{2,3,4}\rangle$", fontdict=textdict)
    axs[0].text(QCX + num_qubits + 1.4, 2.5,
            r"$|\psi_1\rangle|\psi_2\rangle|\psi_{3,4}\rangle$",
            fontdict=textdict)
    axs[0].text(QCX + num_qubits + 1.4, 3.5,
            r"$|\psi_1\rangle|\psi_2\rangle|\psi_3\rangle|\psi_4\rangle$",
            fontdict=textdict)

    # Config axes
    for ax in axs:
        ax.set_aspect(1.0)
        ax.set_ylim(-2, WIRES_TOP + 1)
        ax.set_xlim(QCX - 0.5, QCX + num_qubits + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    fig.subplots_adjust(wspace=0.05)
    return fig


def figure2(path_to_search_stats):

    # Old font size
    old_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 14

    # /// User Constants
    AREC = (0.15, 0.10 + 0.48, 0.8, 0.37)      # top subplot (random agent)
    BREC = (0.62, 0.345 + 0.48, 0.3, 0.1)      # top inset (random agent)
    CREC = (0.15, 0.10, 0.8, 0.37)             # bottom subplot (greedy agent)
    DREC = (0.62, 0.345, 0.3, 0.1)             # bottom inset (gredy agent)

    with open(path_to_search_stats, mode="rb") as f:
        stats = pickle.load(f)

    fig = plt.figure(figsize=(6, 9))
    axA = fig.add_axes(AREC)
    axB = fig.add_axes(BREC)
    axC = fig.add_axes(CREC)
    axD = fig.add_axes(DREC)
    axA.set_yscale('log')
    axB.set_yscale('log')
    axC.set_yscale('log')
    axD.set_yscale('log')
    colors = ["black", "tab:red", "tab:orange", "tab:green", "tab:blue"]
    colors = list(reversed(colors))

    # Plot random agent
    step_index_disentangled = []
    for n, k in enumerate([4,5,6,7,8]):
        # Dimensions are (sample, state, single qubit entanglement)
        random_ent = stats[k]["random"]["entanglements"]
        random_maxsteps = max(len(x) for x in random_ent)
        avg_entanglements_per_step = []
        std_entanglements_per_step = []
        for i in range(random_maxsteps):
            ent_step_i = []
            for arr in random_ent:
                if len(arr) <= i:
                    ent_step_i.append(1e-4)
                else:
                    ent_step_i.append(np.mean(arr[i]))
            avg_entanglements_per_step.append(np.mean(ent_step_i))
            std_entanglements_per_step.append(np.std(ent_step_i))
        avg_entanglements_per_step = np.array(avg_entanglements_per_step)
        std_entanglements_per_step = np.array(std_entanglements_per_step)
        axA.plot(avg_entanglements_per_step, color=colors[n],
                 linewidth=1.5, label=f"L={k}")
        axA.fill_between(
            np.arange(avg_entanglements_per_step.shape[0]),
            avg_entanglements_per_step - std_entanglements_per_step,
            avg_entanglements_per_step + std_entanglements_per_step,
            linewidth=0,
            color=colors[n],
            alpha=0.1
        )
        step_index_disentangled.append(
            np.argmax(avg_entanglements_per_step < 1e-3)
        )

    axB.plot([4,5,6,7,8], step_index_disentangled, color='k', linestyle='--',
             marker='o', linewidth=0.5)

    # Plot greedy agent
    step_index_disentangled = []
    for n, k in enumerate([4,5,6,7,8]):
        # Dimensions are (sample, state, single qubit entanglement)
        random_ent = stats[k]["greedy"]["entanglements"]
        random_maxsteps = max(len(x) for x in random_ent)
        avg_entanglements_per_step = []
        std_entanglements_per_step = []
        for i in range(random_maxsteps):
            ent_step_i = []
            for arr in random_ent:
                if len(arr) <= i:
                    ent_step_i.append(1e-4)
                else:
                    ent_step_i.append(np.mean(arr[i]))
            avg_entanglements_per_step.append(np.mean(ent_step_i))
            std_entanglements_per_step.append(np.std(ent_step_i))
        avg_entanglements_per_step = np.array(avg_entanglements_per_step)
        std_entanglements_per_step = np.array(std_entanglements_per_step)
        axC.plot(avg_entanglements_per_step, color=colors[n],
                 linewidth=1.5, label=f"L={k}")
        axC.fill_between(
            np.arange(avg_entanglements_per_step.shape[0]),
            avg_entanglements_per_step - std_entanglements_per_step,
            avg_entanglements_per_step + std_entanglements_per_step,
            linewidth=0,
            color=colors[n],
            alpha=0.1
        )
        step_index_disentangled.append(
            np.argmax(avg_entanglements_per_step < 1e-3)
        )

    axD.plot([4,5,6,7,8], step_index_disentangled, color='k', linestyle='--',
             marker='o', linewidth=0.5)

    for ax in (axA, axC):
        ax.set_ylabel("$\mathrm{S_{avg}}$")
        ax.set_xlabel("step")
        ax.set_xlim(0, 800)
        ax.set_ylim(1e-4, 1)
    # axA.legend(loc='center right', ncols=2, fontsize=14)
    axC.legend(loc='lower right', ncols=2, fontsize=14)
    for ax in (axB, axD):
        ax.set_xticks([4,5,6,7,8])
        ax.set_yticks([10, 10**2, 10**3])

    # Restore old fontsize
    mpl.rcParams['font.size'] = old_fontsize
    return fig


def figure5(initial_state, selected_actions=None):

    # Save old fontsize
    old_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 18

    num_qubits = int(np.log2(initial_state.size))
    EPSI = 1e-3
    max_steps = 30 if selected_actions is None else len(selected_actions)
    
    # Rollout a trajectory
    actions, entanglements, _probabilities = rollout(initial_state, max_steps)
    if selected_actions is not None:
        probabilities = []
        for i, a in enumerate(selected_actions):
            probabilities.append(_probabilities[i, a])
        probabilities = np.array(probabilities)
    else:
        probabilities = np.array([np.max(prob) for prob in _probabilities])
    steps = len(actions)

    # /// USER CONSTANTS
    MSA = 20        # maximum steps per ax
    QCR = 0.4       # qubits' circles radius
    QCX = -1        # qubits' circles X coordinate
    QFS = 28        # qubits' circles font size
    QLW = 3         # qubits' circles linewidth
    WLW = 1         # qubit wires linewidth
    AFS = 22        # $S_{avg}$ text fontsize
    AEY = -0.8      # $S_{avg}$ + per step avg. entropies Y coordinate
    EFS = 20        # single qubit entropies fontsize
    ERS = 0.6       # single qubit entropies background rectangle size
    ECR = 0.3       # single qubit entropies circle radius
    ECA = 0.1       # single qubit entropies circle alpha
    GOX = 0.5       # gate's X axis offset from single qubit entanglement circle
    GSS = 150       # gate's wire circle scatter size
    GLW = 4         # gate's wire linewidth
    PFS = 20        # gate's probability fontsize
    TLX = -1.5      # step timeline starting X coordinate
    TLW = 1.2       # step timeline linewidth
    LFS = 26        # other labels font size

    # /// DERIVED LAYOUT CONSTANTS
    NAX = divmod(steps + 1, MSA)[0] + int(((steps+1) % MSA) > 0)    # number of axs
    FIGSIZE = (22, 9 * NAX)                                         # figsize

    # Initialize figure
    fig = plt.figure(figsize=FIGSIZE)
    fig, axs = plt.subplots(NAX, 1, figsize=FIGSIZE, squeeze=False)
    fig.tight_layout()

    # Draw qubit circles & "$S_{avg}$" text
    qubits_fontdict = dict(fontsize=QFS, ha='center',
                           va='center')
    avg_ent_fontdict = dict(fontsize=AFS, ha='center',
                            va='center', color='k')
    for ax in axs.flat:
        # Draw qubit circles 
        qubits_xs = np.full(num_qubits, QCX)
        qubits_ys = np.arange(num_qubits)
        for x, y in zip(qubits_xs, qubits_ys):
            circle = patches.Circle((x, y), QCR, edgecolor='k', linewidth=QLW,
                                    fill=True, facecolor='white')
            ax.add_patch(circle)
            ax.text(x, y, f'$q_{y+1}$', fontdict=qubits_fontdict)
        # Draw average entanglement text
        ax.text(QCX, AEY, '$S_{avg}$', fontdict=avg_ent_fontdict)

    # Draw actions & entanglements
    entanglement_fontdict = dict(fontsize=EFS, ha='center',
                                 va='center',
                                 weight='bold')
    for i in range(steps + 1):
        k, j = divmod(i, MSA)
        ax = axs[k, 0]
        # Draw single qubit entanglements
        for n in range(num_qubits):
            e = entanglements[i][n] / np.log(2)
            if e < (EPSI / np.log(2)):
                color = 'k'
                t = str(np.round(e * 1e3, 2))
            elif EPSI < e < 1e-2:
                color = 'blue'
                t = str(np.round(e * 1e2, 2))
            elif 1e-2 < e < 1e-1:
                color = 'magenta'
                t = str(np.round(e * 1e1, 2))
            else:
                color = 'red'
                t = str(np.round(e, 2))
            bg = patches.Rectangle((j - ERS/2, n - ERS/2),
                                   ERS, ERS, facecolor='white', zorder=1)
            ax.add_patch(bg)
            # Add circles behind text
            circle = patches.Circle((j, n), ECR, facecolor=color, alpha=ECA, zorder=2)
            ax.add_patch(circle)
            entanglement_fontdict.update(color=color)
            ax.text(j, n, t, fontdict=entanglement_fontdict)

        # Draw average entanglement on this step
        S_avg = np.mean(entanglements[i]) / np.log(2)
        ax.text(j, AEY, str(np.round(S_avg, 3)),
                fontdict=dict(fontsize=AFS, ha='center',
                              va='center', color='k'))

        # Skip drawing of gate if we are at terminal step
        if i == len(actions):
            continue

        # Draw gate
        q0, q1 = actions[i]
        ax.plot([j + GOX, j + GOX], [q0, q1], color='k', linewidth=GLW)
        ax.scatter([j + GOX, j + GOX], [q0, q1], s=GSS, color='k', zorder=1)

        # Draw probability
        p = int(probabilities[i] * 100)
        text_x = j + GOX - 0.3
        text_y = (q0 + q1) / 2
        if text_y - int(text_y) < 0.4:
            text_y += 0.5
        ax.text(text_x, text_y, f'{p}\\%', fontdict=dict(fontsize=PFS,
                rotation='vertical', va='center'))

    for i, ax in enumerate(axs.flat, 1):
        # Set aspect & remove ticks
        ax.set_aspect('equal')
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        # Set limits
        ax.set_xlim(-1.5, MSA)
        ax.set_ylim(-2, num_qubits)
        ax.set_xlabel('episode step', fontsize=LFS)
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        xbegin, xend = (i-1) * MSA, min(i * MSA, steps)
        # Add horizontal line per qubit
        for q in range(num_qubits):
            ax.plot([-1, xend - xbegin], [q, q], linewidth=WLW, color='k', zorder=-10)
        # Add horizontal arrow indicating agent timesteps
        # ax.plot([-0.5, 0.5 + xend - xbegin], [TLX, TLX], linewidth=TLW, color='k')
        xticks = np.arange(0.5, 0.5 + xend - xbegin, dtype=np.float32)
        xticks = np.concatenate([np.array([-.5]), xticks])
        xticklabels = (i-1)*MSA + xticks[1:] + 1
        xticklabels = [''] + [str(x) for x in xticklabels.astype(np.int32)]
        ax.plot(xticks, [TLX] * len(xticks), markevery=1, marker='|',
                markersize=10, markeredgewidth=2, linewidth=TLW, color='k')
        ax.arrow(0.5, TLX, (xend - xbegin), 0, width=0.0005, head_width=0.1, color='k')
        # Add text labels per step
        for x, lab in zip(xticks, xticklabels):
            ax.text(x, AEY - 1.1, str(lab), fontsize=LFS, ha='center')

    mpl.rcParams['font.size'] = old_fontsize
    return fig


def benchmark_agents(ntests=1000):

    TEST_STATES = {
        4: ["RR-R-R", "RR-RR", "RRR-R", "RRRR"],
        5: ["RR-R-R-R", "RR-RR-R", "RRR-R-R", "RRR-RR", "RRRR-R", "RRRRR"],
        6: ["RR-R-R-R-R", "RR-RR-R-R", "RR-RR-RR", "RRR-R-R-R", "RRR-RR-R",
            "RRRR-R-R", "RRRR-RR", "RRRRR-R", "RRRRRR"]
    }
    MAX_STEPS = 90

    # Load agents
    rl_agent6q = torch.load(PATH_6Q_AGENT, map_location='cpu')
    rl_agent5q = torch.load(PATH_5Q_AGENT, map_location='cpu')
    rl_agent4q = torch.load(PATH_4Q_AGENT, map_location='cpu')
    for enc in rl_agent6q.policy_network.net:
        enc.activation_relu_or_gelu = 1
    rl_agent6q.policy_network.eval()
    for enc in rl_agent5q.policy_network.net:
        enc.activation_relu_or_gelu = 1
    rl_agent5q.policy_network.eval()
    for enc in rl_agent4q.policy_network.net:
        enc.activation_relu_or_gelu = 1
    rl_agent4q.policy_network.eval()
    greedy = GreedyAgent(epsi=1e-3)
    random = RandomAgent(epsi=1e-3)

    results = {}
    # Do the tests
    for num_qubits in (4,5,6):
        print(f'Testing on {num_qubits} qubit system...')
        for state_str in TEST_STATES[num_qubits]:
            print(f'\tTesting states', state_str)
            # Generate test states
            initial_states = np.array(
                [str2state(state_str) for _ in range(ntests)])

            # Test RL agent
            if num_qubits == 6:
                rl_agent = rl_agent6q
            elif num_qubits == 5:
                rl_agent = rl_agent5q
            elif num_qubits == 4:
                rl_agent = rl_agent4q
            RL_res = test_agent(
                rl_agent, initial_states, num_envs=ntests,
                obs_fn="rdm_2q_mean_real", max_episode_steps=MAX_STEPS)
            print('\t\tDone testing RL agent.')

            # Test random agent
            env = QuantumEnv(num_qubits, 1, epsi=1e-3)
            random_len = []
            random_ent = []
            random_solves = []
            for s in initial_states:
                env.reset()
                env.simulator.states = np.expand_dims(s, 0)
                path, ents = random.start(s, env.simulator)
                if path is None and ents is None:
                    random_len.append(np.nan)
                    random_ent.append(np.nan)
                    random_solves.append(np.nan)
                else:
                    random_len.append(len(path))
                    random_ent.append(np.mean(ents[-1]))
                    random_solves.append(1)
            print('\t\tDone testing Random agent.')

            # Test greedy agent
            greedy_len = []
            greedy_ent = []
            greedy_solves = []
            for s in initial_states:
                env.reset()
                env.simulator.states = np.expand_dims(s, 0)
                path, ents = greedy.start(s, env.simulator)
                if path is None and ents is None:
                    greedy_len.append(np.nan)
                    greedy_ent.append(np.nan)
                    greedy_solves.append(np.nan)
                else:
                    greedy_len.append(len(path))
                    greedy_ent.append(np.mean(ents[-1]))
                    greedy_solves.append(1)
            print('\t\tDone testing Greedy agent.')

            results[state_str] = {
                "rl": {
                    "avg_steps": float(np.nanmean(RL_res["lengths"][RL_res["done"]])),
                    "std_steps": float(np.nanstd(RL_res["lengths"][RL_res["done"]])),
                    "avg_final_ent": float(np.nanmean(RL_res["entanglements"][RL_res["done"]])),
                    "success": float(np.nanmean(RL_res["done"]))
                },
                "random": {
                    "avg_steps": float(np.nanmean(random_len)),
                    "std_steps": float(np.nanstd(random_len)),
                    "avg_final_ent": float(np.nanmean(random_ent)),
                    "success": float(np.nanmean(random_solves))
                },
                "greedy": {
                    "avg_steps": float(np.nanmean(greedy_len)),
                    "std_steps": float(np.nanstd(greedy_len)),
                    "avg_final_ent": float(np.nanmean(greedy_ent)),
                    "success": float(np.nanmean(greedy_solves))
                }
            }
    return results


def figure6(benchmark_results):

    # Save old fontsize
    old_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 14

    # /// User Constants
    BWI = 0.5              # Bar width
    BOF = 0.8              # Bar offset

    # Initialize figure and create axes
    fig = plt.figure(figsize=(12, 9), layout='constrained')
    gridspec = GridSpec(2, 5, fig)
    ax4 = fig.add_subplot(gridspec[0, :2])
    ax5 = fig.add_subplot(gridspec[0, 2:])
    ax6 = fig.add_subplot(gridspec[1, :])
    axes = (ax4, ax5, ax6)

    # Scarlet #B85042, Light Olive #E7E8D1, Light Teal #A7BEAE
    # deep slate blue #3D405B medium blue #0077B6 light blue #90E0EF, pale blue #CAF0F8
    colors = ['#3D405B', '#A7BEAE', '#B85042']
    colors = ['tab:orange', 'tab:green', 'tab:blue']
    colors = ['#7abacc', '#0077B6', '#3D405B']
    
    # Plot 4q results
    keys4q = (k for k in benchmark_results if len(k.replace('-', '')) == 4)
    offset = 0
    xticks = []
    xticklabels = []
    for k in keys4q:
        res = benchmark_results[k]
        xs = [offset, offset + BWI, offset + 2*BWI]
        heights = [res['random']['avg_steps'], res['greedy']['avg_steps'], res['rl']['avg_steps']]
        stds = [res['random']['std_steps'], res['greedy']['std_steps'], res['rl']['std_steps']]
        xticks.append(offset + BWI)
        xticklabels.append(str2latex(k))
        rects = ax4.bar(xs, heights, BWI, color=colors, yerr=stds, ecolor='red', capsize=5)
        # labels = [f'{int(h)} ±{int(s)}' for h, s in zip(heights, stds)]
        labels = [int(h) for h in heights]
        ax4.bar_label(rects, labels, rotation=45)
        offset += 3 * BWI + BOF
    ax4.set_xticks(xticks, xticklabels, rotation=45)
    ax4.set_yticks([1, 10, 20, 30])

    # Plot 5q results
    keys5q = (k for k in benchmark_results if len(k.replace('-', '')) == 5)
    offset = 0
    xticks = []
    xticklabels = []
    for k in keys5q:
        res = benchmark_results[k]
        xs = [offset, offset + BWI, offset + 2*BWI]
        heights = [res['random']['avg_steps'], res['greedy']['avg_steps'], res['rl']['avg_steps']]
        stds = [res['random']['std_steps'], res['greedy']['std_steps'], res['rl']['std_steps']]
        xticks.append(offset + BWI)
        xticklabels.append(str2latex(k))
        rects = ax5.bar(xs, heights, BWI, color=colors, yerr=stds, ecolor='red', capsize=5)
        # labels = [f'{int(h)} ±{int(s)}' for h, s in zip(heights, stds)]
        labels = [int(h) for h in heights]
        ax5.bar_label(rects, labels, rotation=45)
        offset += 3 * BWI + BOF
    ax5.set_xticks(xticks, xticklabels, rotation=45)
    ax5.set_yticks([1, 20, 40, 60, 80])

    # Plot 6q results
    keys6q = (k for k in benchmark_results if len(k.replace('-', '')) == 6)
    offset = 0
    xticks = []
    xticklabels = []
    for k in keys6q:
        res = benchmark_results[k]
        xs = [offset, offset + BWI, offset + 2*BWI]
        heights = [res['random']['avg_steps'], res['greedy']['avg_steps'], res['rl']['avg_steps']]
        stds = [res['random']['std_steps'], res['greedy']['std_steps'], res['rl']['std_steps']]
        xticks.append(offset + BWI)
        xticklabels.append(str2latex(k))
        rects = ax6.bar(xs, heights, BWI, color=colors, yerr=stds,
                        ecolor='red', capsize=5,
                        label=['Random', 'Greedy', 'RL'])
        # labels = [f'{int(h)} ±{int(s)}' for h, s in zip(heights, stds)]
        labels = [int(h) for h in heights]
        ax6.bar_label(rects, labels, rotation=45)
        offset += 3 * BWI + BOF
    ax6.set_xticks(xticks, xticklabels, rotation=45)
    handles, labels = ax6.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax6.legend(by_label.values(), by_label.keys(), loc=(0.15, 1.0))
    ax6.set_yticks([1, 60, 120, 180])

    # Configure axes
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Restore old font size
    mpl.rcParams['font.size'] = old_fontsize
    return fig


def figure4(initial_state, state_name=''):
    #
    # Figure contains only 1 ax and almost everything drawn with
    # shape primitives. The left subfigure shows the application of gates
    # over the quantuam system, the right subfigure shows the policy
    # distribution.
    #
    #      +-----+    +-----+
    #      |     |    |     |
    #      +-----+    +-----+
    #       gates      policy
    #
    #   "gates" boundaries:
    #       top left: NA
    #       top right: NA
    #       bottom left: (0, -2)
    #       bottom right: (n_qubits, -2)
    #
    #   "policy" boundaries:
    #       top left: NA
    #       top right: NA
    #       bottom left: (n_qubits + 1, -2.2)
    #       bottom right: (n_qubits + 1 + C*n_actions_shown, -2.2)

    num_qubits = int(np.log2(initial_state.size))
    # Rollout
    actions, _, probabilities = rollout(initial_state)
    nsteps = len(actions)

    # Select actions that are to be shown in right subfigure
    masked_actions = np.max(probabilities, axis=0) > 0.05
    action_labels = list(itertools.combinations(range(num_qubits), 2))
    action_labels = np.array(action_labels)[masked_actions]
    # Plotted actions
    probs_main = probabilities[:, masked_actions]
    # Summarized actions (plotted in "rest" column)
    probs_rest = probabilities[:, ~masked_actions].sum(axis=1)

    # /// USER CONSTANTS
    QCY = -2        # Y coordinate of qubits' circles               (L subfig)
    QCX = 0         # Min X coordinate of qubits' circles           (L subfig)
    QCR = 0.45      # radius of qubits' circles                     (L subfig)
    QFS = 12        # fontsize of qubits' text                      (L subfig)
    QLW = 1.2       # linewidth of qubits' circles                  (L subfig)
    WLW = 1.0       # linewidth of qubit wires                      (L subfig)
    PBW = 0.9       # width of single bar in "policy" subfigure     (R subfig)
    GLW = 3         # gate wire linewidth                           (L subfig)
    GSS = 90        # gate wire connection scatter size             (L subfig)

    # /// DERIVED LAYOUT CONSTANTS
    R_SUBFIG_XMIN = num_qubits + 1.5
    R_SUBFIG_XMAX = R_SUBFIG_XMIN + 1 + probs_main.shape[1]
    WIRES_BOTTOM = QCY + QCR
    WIRES_TOP = 2*nsteps - 1
    FIGSIZE = (6, max(5, nsteps + 2.5))

    # Initialize figure
    fig, ax = plt.subplots(1, figsize=FIGSIZE, dpi=120)

    # Draw qubit circles with Y=`QCY`, X=[`QCX`, `QCX` + `num_qubits`]
    qubits_xs = np.arange(QCX, QCX + num_qubits)
    qubits_ys = np.full(num_qubits, QCY)
    for x, y in zip(qubits_xs, qubits_ys):
        ax.add_patch(patches.Circle((x, y), QCR, edgecolor='k', linewidth=QLW,
                                    fill=True, facecolor='white', zorder=10))
        ax.text(x, y, f'$q_{x+1}$',
                fontdict=dict(fontsize=QFS, ha='center', va='center', zorder=11))

    # Draw base wires for gates (starting from qubit circles)
    for i in range(num_qubits):
        ax.plot([QCX + i, QCX + i], [WIRES_BOTTOM, WIRES_TOP],
                zorder=0, color='k', linewidth=WLW)

    # Draw gates & policy
    for n in range(nsteps):
        # Draw gate
        q0, q1 = sorted(actions[n])
        ax.plot([q0, q1], [2*n, 2*n], linewidth=GLW, color='k')
        ax.scatter([q0, q1], [2*n, 2*n], s=GSS, color='k')
        
        # Draw horizontal line connecting policy and gate
        ax.plot([-1, R_SUBFIG_XMIN - PBW], [2*n - 1, 2*n - 1],
                linestyle='--', linewidth=0.5, color='k')

        # Draw main policy actions
        pmax = np.max(probs_main[n])
        ax.plot([R_SUBFIG_XMIN, R_SUBFIG_XMAX], [2*n - 1, 2*n - 1],
                color='k', linewidth=0.8)
        ax.text(R_SUBFIG_XMIN - 1.75, 2 * n, f'$\pi(a|s_{n})$',
                fontdict=dict(fontsize=12, ha='left'))
        for x, p in enumerate(probs_main[n]):
            color = 'tab:red' if p == pmax else 'tab:blue'
            x_coord = R_SUBFIG_XMIN + x
            y_coord = 2*n - 1
            bar = patches.Rectangle((x_coord, y_coord), PBW, p * 0.9,
                                    facecolor=color)
            ax.add_patch(bar)
            q0, q1 = action_labels[x]
            ax.text(x_coord + PBW/2, y_coord - 0.3, f'({q0+1},{q1+1})',
                     fontdict=dict(ha='center', va='center', fontsize=10))
            ax.text(x_coord + PBW/2, y_coord + p + 0.1, f'${int(p * 100)}\%$',
                    fontdict=dict(ha='center', fontsize=10))

        # Draw summarized "rest" actions
        p = probs_rest[n] * 0.9
        x_coord = R_SUBFIG_XMIN + len(probs_main[n])
        y_coord = 2*n - 1
        bar = patches.Rectangle((x_coord - PBW/2, y_coord), PBW, p,
                                facecolor='tab:cyan')
        ax.add_patch(bar)
        ax.text(x_coord + PBW/2, y_coord - 0.3, 'rest', fontdict=dict(
            ha='center', va='center', fontsize=12))
        ax.text(x_coord + PBW/2, y_coord + p + 0.1, f'${int(p * 100)}\%$',
                fontdict=dict(ha='center', fontsize=10))

    # Add "actions" text
    ax.text((R_SUBFIG_XMIN + R_SUBFIG_XMAX) / 2, QCY - QCR / 2, 'actions',
            fontdict=dict(ha='center', fontsize=12, va='bottom'))
    # Add state name text
    ax.text(2, -3.5, state_name,
            fontdict=dict(fontsize=16, ha='center'))

    # Set limits & ticks
    ax.set_aspect('equal')
    ax.set_ylim(-4, 2 * nsteps + 1)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig


if __name__ == '__main__':

    # Benchmark agents
    # results = benchmark_agents(1000)
    # with open("agents-benchmark.json", mode='w') as f:
    #     json.dump(results, f, indent=2)

    # Figure 1
    fig1a = figure1a()
    fig1a.savefig('../figures/circuit-3q.pdf')
    fig1bd = figure1bd()
    fig1bd.savefig('../figures/circuit-4q.pdf')

    # Figure 2
    # fig2 = figure2('../data/random-greedy-stats.pickle')
    # fig2.savefig('../figures/exponential-difficulty-both.pdf')

    # # Figure 4a
    # bell =  np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex64) / np.sqrt(2)
    # bell_bell = np.kron(bell, bell)
    # fig4a = figure4(bell_bell, r"$|Bell_{1,2}\rangle|Bell_{3,4}\rangle$")
    # fig4a.savefig('../figures/Figure_4a.pdf')

    # # Figure 4b
    # w = np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=np.complex64) / np.sqrt(3)
    # zero = np.array([1, 0], dtype=np.complex64)
    # zero_ghz = np.kron(zero, w)
    # fig4b = figure4(zero_ghz, r"$|0\rangle|GHZ_{2,3,4}\rangle$")
    # fig4b.savefig('../figures/Figure_4b.pdf')

    # Figure 4c
    np.random.seed(23)
    # s = np.kron(random_quantum_state(3, 1.0), random_quantum_state(1, 1.0))
    # s = np.einsum("ijk,l -> ijkl", random_quantum_state(3, 1.0), random_quantum_state(1, 1.0))
    # s = np.kron(random_quantum_state(1, 1.0), random_quantum_state(3, 1.0))
    # fig4c = figure4(s, r"$|R_{1,2,3}\rangle|R_4\rangle$")
    # fig4c = figure4(s, r"$|R_1\rangle|R_{2,3,4}\rangle$")
    # fig4c.savefig('../figures/Figure_4c.pdf')

    # # Figure 4d
    # np.random.seed(45)
    # fig4d = figure4(random_quantum_state(4, 1.0), r"$|R_{1,2,3,4}\rangle$")
    # fig4d.savefig('../figures/Figure_4d.pdf')

    # # Figure 5
    # np.random.seed(45)
    # fig5 = figure5(random_quantum_state(5, 1.0))
    # fig5.savefig('../figures/5q-trajectory.pdf')

    # # Figure 6
    # with open('../data/agents-benchmark.json') as f:
    #     results = json.load(f)
    #     fig6 = figure6(results)
    #     fig6.savefig('../figures/456q-agents.pdf')
