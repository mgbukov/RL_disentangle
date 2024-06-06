import itertools
import numpy as np
import torch
import matplotlib as mpl
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

from context import *
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state



mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

AGENTS_PATH = os.path.join(project_dir, "logs", "4q_2000iters_animation")
COLOR0 = "dodgerblue"
COLOR1 = "lightskyblue"
COLOR2 = "aqua"
COLOR3 = "lightgreen"
EPSI = 1e-3



def rollout(state, agent, max_steps=30):
    """Returns action pairs, entanglements and policy for each step."""

    # Initialize environment
    n_qubits = int(np.log2(state.size))
    env = QuantumEnv(n_qubits, 1, obs_fn='rdm_2q_mean_real',
                     max_episode_steps=max_steps+1)
    env.reset()
    shape = (1,) + (2,) * n_qubits
    env.simulator.states = state.reshape(shape)

    # Rollout
    actions, entanglements, probabilities = [], [], []
    # Append starting entanglement
    entanglements.append(env.simulator.entanglements.copy().ravel())
    for i in range(max_steps):
        observation = torch.from_numpy(env.obs_fn(env.simulator.states))
        probs = agent.policy(observation).probs[0].cpu().numpy()
        probabilities.append(probs)
        a = np.argmax(probs)
        actions.append(env.simulator.actions[a])
        _, _, t, _, _ = env.step([a], reset=False)
        ent = env.simulator.entanglements.copy()
        entanglements.append(ent.ravel())
        if np.all(t):
            break
    return np.array(actions), np.array(probabilities), np.array(entanglements)


def draw_frame(policy, actions, entanglements, n_qubits=4, draw_percentages=True,
               endswith="gate"):

    steps = len(actions)

    # /// USER CONSTANTS
    DPI = 240                        # dpi
    FIG = (1920/DPI, 1080/DPI)      # figure size in inches
    MSA = 16        # maximum steps per ax
    QCR = 0.4       # qubits' circles radius
    QCX = -1        # qubits' circles X coordinate
    QFS = 16        # qubits' circles font size
    QLW = 1         # qubits' circles linewidth
    WLW = 0.2       # qubit wires linewidth
    AEY = -0.8      # $S_{avg}$ + per step avg. entropies Y coordinate
    EFS = 9         # single qubit entropies fontsize
    ECR = 0.25      # single qubit entropies circle radius
    ECA = 0.9       # single qubit entropies circle alpha
    GOX = 0.5       # gate's X axis offset from single qubit entanglement circle
    GSS = 100       # gate's wire circle scatter size
    GLW = 2         # gate's linewidth
    PFS = 10        # gate's probability fontsize
    TLW = 0.5       # step timeline linewidth
    LFS = 12        # other labels font size

    # Initialize figure
    fig = plt.figure(figsize=FIG, dpi=DPI, layout="none")
    ax_circuit = fig.add_axes((0.02, 0.35, 0.85, 0.6))
    ax_avgent  = fig.add_axes((0.02, .14, 0.85, 0.15))

    # Draw qubit circles & "$S_{avg}$" text
    qubits_fontdict  = dict(fontsize=QFS, ha='center', va='center', color="white")
    circle_styledict = dict(edgecolor='k', linewidth=QLW, fill=True, facecolor='k')

    # Draw qubit circles
    qubits_xs = np.full(n_qubits, QCX)
    qubits_ys = np.arange(n_qubits)
    for x, y in zip(qubits_xs, qubits_ys):
        circle = patches.Circle((x, y), QCR, **circle_styledict)
        ax_circuit.add_patch(circle)
        ax_circuit.text(x, y, f'$q_{y+1}$', **qubits_fontdict)

    # Draw actions & entanglements on each step
    ent_fontdict = dict(fontsize=EFS, ha='center', va='center', color="darkblue", weight='bold')
    percentages = (100 * policy).astype(np.int32)
    action_i = np.argmax(policy, axis=1)
    percentages[np.arange(steps), action_i] += 100 - percentages.sum(axis=1)

    for i in range(steps + 1):
        if i == steps and endswith == "gate":
            break
        # Draw single qubit entanglements
        for n in range(n_qubits):
            e = entanglements[i][n] / np.log(2)
            if e < (EPSI / np.log(2)):
                color = COLOR3
                t = str(np.round(e * 1e3, 2))
            elif EPSI < e < 1e-2:
                color = COLOR2
                t = str(np.round(e * 1e2, 2))
            elif 1e-2 < e < 1e-1:
                color = COLOR1
                t = str(np.round(e * 1e1, 2))
            else:
                color = COLOR0
                t = str(np.round(e, 2))
            bg = patches.Rectangle((i - ECR, n - ECR),
                                   2*ECR, 2*ECR, facecolor='white', zorder=1)
            ax_circuit.add_patch(bg)
            # Add circles behind text
            circle = patches.Circle(
                (i, n), ECR, facecolor=color, edgecolor=None,
                alpha=ECA, zorder=2
            )
            ax_circuit.add_patch(circle)
            ent_fontdict.update(color=color)
            # ax_circuit.text(i, n, t, fontdict=ent_fontdict)

        # Skip drawing of gate if we are at terminal step
        if i == len(actions) or (i == steps and endswith == "ent"):
            break
        # Draw gate
        q0, q1 = actions[i]
        ax_circuit.plot([i + GOX, i + GOX], [q0, q1], color='k', linewidth=GLW)
        ax_circuit.scatter([i + GOX, i + GOX], [q0, q1], s=GSS, color='k', zorder=1)
        # Draw percentages
        if not draw_percentages:
            continue
        text_x = i + GOX - 0.3
        text_y = (q0 + q1) / 2
        if text_y - int(text_y) < 0.4:
            text_y += 0.5
        j = np.argmax(percentages[i])
        ax_circuit.text(text_x, text_y, f'{percentages[i][j]}\\%',
                        fontsize=PFS, rotation='vertical', va='center')

    # Draw S_ent fill curve
    ys = np.mean(entanglements, axis=1)
    xs = np.arange(len(entanglements))
    if endswith == "gate":
        xs, ys = xs[:-1], ys[:-1]
    ax_avgent.plot(xs, ys, color=COLOR0, linewidth=0.5, marker='.')
    ax_avgent.fill_between(xs, ys, color="lightblue", alpha=0.5)
    ax_avgent.text(5, 1e-6,
                   s=r"$S_\mathrm{avg} = \frac{1}{L}\sum_{j=1}^L S_\mathrm{ent}[\rho^{(j)}]$",
                   transform=ax_avgent.transData, fontsize=14, ha="center")
    ax_avgent.text(-1.1-QCR, 1e-2, s=r"$\frac{S_\mathrm{avg}}{\log(2)}$", fontsize=16, ha="left")
    # ax_avgent.plot([0, 10], [1e-4, 1e-4], color='k', linewidth=2)

    # Draw "episode step"
    ax_circuit.text(x=-1-QCR, y=AEY+0.1, s="episode step", ma="center", fontsize=11, va="top", ha="left")

    # Set aspect & remove ticks
    ax_circuit.set_aspect(1.0)
    ax_circuit.set_xticks([], [])
    ax_circuit.set_yticks([], [])
    ax_avgent.set_xticks([], [])
    # Set limits
    ax_circuit.set_xlim(-1.5, 11)
    ax_circuit.set_ylim(-1, n_qubits)
    ax_avgent.set_xlim(-1.5, 11)
    ax_avgent.set_yscale("log")
    ax_avgent.set_ylim(1e-4, 1.0)
    ax_avgent.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1],
                         ['', "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", ''], fontsize=14)

    # Remove spines
    ax_circuit.spines['top'].set_visible(False)
    ax_circuit.spines['left'].set_visible(False)
    ax_circuit.spines['right'].set_visible(False)
    ax_circuit.spines['bottom'].set_visible(False)
    ax_avgent.spines['top'].set_visible(False)
    # ax_avgent.spines['left'].set_visible(False)
    ax_avgent.spines['right'].set_visible(False)
    # ax_avgent.spines['bottom'].set_visible(False)
    ax_avgent.spines["left"].set_position(('data', 0.))
    ax_avgent.spines["bottom"].set_bounds(0.0, 10)
    # Draw grid
    ax_avgent.plot([0.0, 10], [1e-4, 1e-4], linewidth=.1, color='k')
    ax_avgent.plot([0.0, 10], [1e-3, 1e-3], linewidth=.1, color='k')
    ax_avgent.plot([0.0, 10], [1e-2, 1e-2], linewidth=.1, color='k')
    ax_avgent.plot([0.0, 10], [1e-1, 1e-1], linewidth=.1, color='k')
    xend = 10

    # Draw wires
    for q in range(n_qubits):
        ax_circuit.plot([QCX+QCR, xend], [q, q], linewidth=WLW, color='k', zorder=-10)

    # Add horizontal arrow indicating agent timesteps
    xticks = np.arange(0, xend) + 0.5
    xticklabels = np.arange(1, xend+1)
    ax_circuit.plot(xticks, [AEY] * len(xticks), markevery=1, marker='|',
            markersize=6, markeredgewidth=1, linewidth=TLW, color='k')
    ax_circuit.arrow(xend-0.5, AEY, 1.0, 0, linewidth=TLW, head_width=0.2, color='k')

    # Add text labels per step
    for x, lab in zip(xticks, xticklabels):
        ax_circuit.text(x, AEY - 0.45, str(lab), fontsize=LFS, ha='center')

    # Add legend
    ax_circuit.scatter([], [], s=100, color=COLOR0, label='$\mathrm{S_{ent} > 10^{-1}}$')
    ax_circuit.scatter([], [], s=100, color=COLOR1, label='$\mathrm{S_{ent} < 10^{-1}}$')
    ax_circuit.scatter([], [], s=100, color=COLOR2, label='$\mathrm{S_{ent} < 10^{-2}}$')
    ax_circuit.scatter([], [], s=100, color=COLOR3, label='$\mathrm{S_{ent} < 10^{-3}}$')
    ax_circuit.legend(loc=(0.96, .3), fontsize=11, ncols=1, frameon=False)
    return fig


def draw_trajectory(state, agent):
    actions, policy, entanglements = rollout(state, agent, 10)
    nsteps = len(actions)
    frames = []
    for i in range(0, nsteps+1):
        frame = draw_frame(policy[:i], actions[:i], entanglements[:i+1], n_qubits=4,
                           draw_percentages=False, endswith="ent")
        frames.append(frame)
        if i == nsteps:
            break
        frame = draw_frame(policy[:i+1], actions[:i+1], entanglements[:i+2], n_qubits=4,
                           draw_percentages=False, endswith="gate")
        frames.append(frame)
    return frames


if __name__ == "__main__":
    agent = torch.load("../agents/4q-agent.pt")
    state = random_quantum_state(4)
    actions, policy, entanglements = rollout(state, agent, 5)
    frame = draw_frame(policy, actions, entanglements)
    frame.savefig("../animation/test.png")

    # agent = torch.load(os.path.join(AGENTS_PATH, "agent1.pt"))
    # np.random.seed(4)
    # state = random_quantum_state(4)
    # frames = draw_trajectory(state, agent)
    # for i, frame in enumerate(frames):
    #     frame.savefig(f"../animation/frame{i}.png")
    #     plt.close(frame)