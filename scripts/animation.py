import itertools
import pickle
import numpy as np
import torch
import matplotlib as mpl
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from context import *
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

AGENTS_PATH = os.path.join(project_dir, "logs", "4q_2000iters_animation")
TRAIN_HISTORY_PATH = os.path.join(project_dir, "logs", "4q_2000iters_animation")

COLOR0 = "dodgerblue"
COLOR1 = "lightskyblue"
COLOR2 = "aqua"
COLOR3 = "lightgreen"
EPSI = 1e-3

# /// USER CONSTANTS
DPI = 240                       # dpi
FIG = (1920/DPI, 1080/DPI)      # figure size in inches
MSA = 16                        # maximum steps per ax
QCR = 0.35                      # qubits' circles radius
QCX = -1                        # qubits' circles X coordinate
QFS = 12                        # qubits' circles font size
QLW = 1                         # qubits' circles linewidth
WLW = 0.2                       # qubit wires linewidth
AEY = -0.8                      # $S_{avg}$ + per step avg. entropies Y coordinate
EFS = 9                         # single qubit entropies fontsize
ECR = 0.23                      # single qubit entropies circle radius
ECA = 0.9                       # single qubit entropies circle alpha
GOX = 0.5                       # gate's X axis offset from single qubit entanglement circle
GSS = 100                       # gate's wire circle scatter size
GLW = 2                         # gate's linewidth
PFS = 9                         # gate's probability fontsize
TLW = 0.4                       # step timeline linewidth
LFS = 9                         # other labels font size
END = 10                        # last step show on circuit


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


def draw_policy_ax(policy, ax):

    if len(policy) > 0:
        taken = np.max(policy)
        for x, p in enumerate(policy):
            color = 'tab:red' if p == taken else 'tab:blue'
            bar = patches.Rectangle((x, 0), 0.9, p, facecolor=color)
            ax.add_patch(bar)
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    ax.set_xticks([])
    ax.set_xlabel("action", fontsize=8)
    ax.set_ylabel("probability", fontsize=8)
    ax.set_title("$\pi(a_t|o_t)$", fontsize=10)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(True)
    ax.spines.bottom.set_bounds(0, 6)
    ax.tick_params("both", labelsize=8)


def draw_return_ax(ax, iteration=-1):
    with open(os.path.join(TRAIN_HISTORY_PATH, "train_history.pickle"), mode='rb') as f:
        history = pickle.load(f)

    returns = np.array([item["Return"]["avg"] for item in history])
    returns = returns[:1001]
    ax.plot(returns, linewidth=0.1, color='k', alpha=0.5, zorder=-10)
    ax.set_ylim(-10.5,0.5)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("return", fontsize=8)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.tick_params("both", labelsize=8)
    ax.set_title("$\mathcal{R}$", fontsize=10)
    # ax.set_yscale("symlog")
    ax.set_yticks([-10, -5, 0], ["-10", "-5", "0"])

    if iteration > 0:
        i = min(iteration, 1000)
        xs = np.arange(iteration)
        y1 = np.full((i,), -1000)
        y2 = returns[:i]
        ax.fill_between(xs, y2, y1, color="k", alpha=0.25, zorder=-10, linewidth=0.0)
        ax.scatter([iteration], [returns[i]], s=30, color="tab:red", zorder=0)


def draw_S_avg_ax(entanglements, ax):

    # Draw S_ent fill curve
    ys = np.maximum(np.mean(entanglements, axis=1), 1e-5)
    xs = np.arange(len(entanglements))
    ax.plot(xs, ys, color=COLOR0, linewidth=0.5, marker='.')
    ax.fill_between(xs, ys, color="lightblue", alpha=0.5)
    ax.text(-1.5-QCR, 5e-9,
        s=r"$S_\mathrm{avg} = \frac{1}{L}\sum_{j=1}^L S_\mathrm{ent}[\rho^{(j)}]$",
        transform=ax.transData, fontsize=6, ha="left")
    ax.text(-1.5-QCR, 1e-3, s=r"$\frac{S_\mathrm{avg}}{\log(2)}$", fontsize=12, ha="left")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_yticks([1e-5, 1e-3, 1e-2, 1e-1, 1],
                        ['0', "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", ''])
    ax.tick_params(axis="y", labelsize=8)
    # Draw grid
    ax.plot([0.0, 10], [1e-3, 1e-3], linewidth=.05, color='k')
    ax.plot([0.0, 10], [1e-2, 1e-2], linewidth=.05, color='k')
    ax.plot([0.0, 10], [1e-1, 1e-1], linewidth=.05, color='k')
    ax.set_xticks([], [])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines["left"].set_position(('data', 0.))
    ax.spines["bottom"].set_bounds(0.0, 10)


def draw_frame(policy, actions, entanglements, n_qubits=4, draw_percentages=True,
               draw_S_avg=True, iteration=-1, draw_policy=True,
               draw_return=True, endswith="gate"):

    steps = len(actions)

    # Initialize figure
    fig = plt.figure(figsize=FIG, dpi=DPI, layout="none")
    ax_center = fig.add_axes((0.04, 0.24, 0.65, 0.60))
    ax_bright = fig.add_axes((0.80, 0.12, 0.16, 0.14))
    ax_bottom = fig.add_axes((0.04, 0.12, 0.65, 0.12))
    ax_cright = fig.add_axes((0.8, 0.48, 0.15, 0.18))

    # Draw qubit circles & "$S_{avg}$" text
    qubits_fontdict  = dict(fontsize=QFS, ha='center', va='center', color="white")
    circle_styledict = dict(edgecolor='k', linewidth=QLW, fill=True, facecolor='k')

    # Draw qubit circles
    qubits_xs = np.full(n_qubits, QCX)
    qubits_ys = np.arange(n_qubits)
    for x, y in zip(qubits_xs, qubits_ys):
        circle = patches.Circle((x, y), QCR, **circle_styledict)
        ax_center.add_patch(circle)
        ax_center.text(x, y, f'$q_{y+1}$', **qubits_fontdict)

    # Draw wires
    for q in range(n_qubits):
        ax_center.plot([QCX+QCR, END], [q, q], linewidth=WLW, color='k', zorder=-10)

    # Draw actions & entanglements on each step
    percentages = (100 * policy).astype(np.int32)
    action_i = np.argmax(policy, axis=1)
    percentages[np.arange(steps), action_i] += 100 - percentages.sum(axis=1)

    for i in range(steps):
        # Draw single qubit entanglements
        for n in range(n_qubits):
            e = entanglements[i][n] / np.log(2)
            if e < (EPSI / np.log(2)):
                color = COLOR3
            elif EPSI < e < 1e-2:
                color = COLOR2
            elif 1e-2 < e < 1e-1:
                color = COLOR1
            else:
                color = COLOR0
            bg = patches.Rectangle((i - ECR, n - ECR),
                                   2*ECR, 2*ECR, facecolor='white', zorder=1)
            ax_center.add_patch(bg)
            # Add circles behind text
            circle = patches.Circle(
                (i, n), ECR, facecolor=color, edgecolor=None,
                alpha=ECA, zorder=2
            )
            ax_center.add_patch(circle)

        # Draw gate
        q0, q1 = actions[i]
        ax_center.plot([i + GOX, i + GOX], [q0, q1], color='k', linewidth=GLW)
        ax_center.scatter([i + GOX, i + GOX], [q0, q1], s=GSS, color='k', zorder=1)
        # Draw percentages
        if not draw_percentages:
            continue
        text_x = i + GOX - 0.3
        text_y = (q0 + q1) / 2
        if text_y - int(text_y) < 0.4:
            text_y += 0.5
        j = np.argmax(percentages[i])
        ax_center.text(text_x, text_y, f'{percentages[i][j]}\\%',
                        fontsize=PFS, rotation='vertical', va='center')

    # Draw last entanglements
    i = steps
    if endswith == "ent":
        for n in range(n_qubits):
            e = entanglements[-1][n] / np.log(2)
            if e < (EPSI / np.log(2)):
                color = COLOR3
            elif EPSI < e < 1e-2:
                color = COLOR2
            elif 1e-2 < e < 1e-1:
                color = COLOR1
            else:
                color = COLOR0
            bg = patches.Rectangle((i - ECR, n - ECR),
                                    2*ECR, 2*ECR, facecolor='white', zorder=1)
            ax_center.add_patch(bg)
            # Add circles behind text
            circle = patches.Circle(
                (i, n), ECR, facecolor=color, edgecolor=None,
                alpha=ECA, zorder=2
            )
            ax_center.add_patch(circle)

    # Draw policy
    if draw_policy:
        if len(policy) == 0:
            draw_policy_ax([], ax_bright)
        else:
            draw_policy_ax(policy[-1], ax_bright)
        ax_bright.set_xlim(0,6)
    else: 
        ax_bright.set_axis_off()

    # Draw S_avg
    if draw_S_avg:
        entanglements_ = entanglements[:-1] if endswith == "gate" else entanglements
        draw_S_avg_ax(entanglements_, ax_bottom)
        ax_bottom.set_xlim(-1.5, END+1)
    else:
        ax_bottom.set_axis_off()

    # Draw return
    if draw_return:
        draw_return_ax(ax_cright, iteration)
    else:
        ax_cright.set_axis_off()

    # Draw iteration #
    if iteration > 0:
        ax_center.text(x=END+3.4, y=n_qubits, s=f"Iteration = {iteration:>3}")

    # Draw "episode step"
    ax_center.text(x=-1.4-QCR, y=AEY+0.1, s="episode step", ma="center", fontsize=9, va="top", ha="left")

    # Set aspect & remove ticks
    ax_center.set_aspect(1.0)
    ax_center.set_xticks([], [])
    ax_center.set_yticks([], [])

    # Set limits
    ax_center.set_xlim(-1.5, 11)
    ax_center.set_ylim(-1, n_qubits)

    # Remove spines
    ax_center.spines['top'].set_visible(False)
    ax_center.spines['left'].set_visible(False)
    ax_center.spines['right'].set_visible(False)
    ax_center.spines['bottom'].set_visible(False)

    # Draw timestep arrow
    xticks = np.arange(0, END) + 0.5
    xticklabels = np.arange(1, END+1)
    ax_center.plot(xticks, [AEY] * len(xticks), markevery=1, marker='|',
            markersize=4, markeredgewidth=1, linewidth=TLW, color='k')
    ax_center.arrow(END-0.5, AEY, 1.0, 0, linewidth=TLW, head_width=0.2, color='k')
    for x, lab in zip(xticks, xticklabels):
        ax_center.text(x, AEY - 0.45, str(lab), fontsize=LFS, ha='center')

    # Draw legend
    ax_center.scatter([], [], s=40, color=COLOR0, label='$\mathrm{S_{ent} > 10^{-1}}$')
    ax_center.scatter([], [], s=40, color=COLOR1, label='$\mathrm{S_{ent} < 10^{-1}}$')
    ax_center.scatter([], [], s=40, color=COLOR2, label='$\mathrm{S_{ent} < 10^{-2}}$')
    ax_center.scatter([], [], s=40, color=COLOR3, label='$\mathrm{S_{ent} < 10^{-3}}$')
    l = ax_center.legend(loc=(0.05, 0.96), fontsize=9, ncols=4, labelspacing=0.75, frameon=True)
    l.get_frame().set_linewidth(0.5)
    return fig


def draw_trajectory(actions, policy, entanglements, iteration=-1, draw_percentages=True,
                    draw_S_avg=True, draw_policy=True, draw_return=True):
    
    nsteps = len(actions)
    for i in range(0, nsteps+1):
        frame = draw_frame(policy[:i], actions[:i], entanglements[:i+1], n_qubits=4,
                           draw_percentages=draw_percentages,
                           draw_S_avg=draw_S_avg,
                           draw_policy=draw_policy,
                           draw_return=draw_return,
                           iteration=iteration,
                           endswith="ent")
        yield frame
        if i == nsteps:
            return
        frame = draw_frame(policy[:i+1], actions[:i+1], entanglements[:i+2], n_qubits=4,
                           draw_percentages=draw_percentages,
                           draw_S_avg=draw_S_avg,
                           draw_policy=draw_policy,
                           draw_return=draw_return,
                           iteration=iteration,
                           endswith="gate")
        yield frame


def save_iteration_frames(iteration):

    os.makedirs(f"../animation/iteration{iteration}", exist_ok=True)
    agent = torch.load(os.path.join(AGENTS_PATH, f"agent{iteration+1}.pt"))

    np.random.seed(8)
    states = [random_quantum_state(4, prob=1.0) for _ in range(10)]
    states = np.array(states, dtype=np.complex64)

    # Find the longest solution path
    actions_batch = []
    policy_batch = []
    entanglements_batch = []
    lens_batch = []
    for s in states:
        a, p, e = rollout(s, agent, max_steps=10)
        actions_batch.append(a)
        policy_batch.append(p)
        entanglements_batch.append(e)
        lens_batch.append(len(a))
    sorted_lens = np.argsort(lens_batch)
    j = sorted_lens[len(sorted_lens) // 2]
    actions, policy, entanglements = actions_batch[j], policy_batch[j], entanglements_batch[j]
    # Draw the longest trajectory
    for _ in range(1):
        frames = draw_trajectory(actions, policy, entanglements, iteration=max(1,iteration))
        for i, frame in enumerate(frames):
            frame.savefig(f"../animation/iteration{iteration}/frame{i:04}.png")
            plt.close(frame)


if __name__ == "__main__":

    # agent = torch.load("../agents/4q-agent.pt")
    # state = random_quantum_state(4)
    # actions, policy, entanglements = rollout(state, agent, 6)
    # frame = draw_frame(policy, actions, entanglements, draw_policy=True,
    #                    draw_return=True, draw_S_avg=True, iteration=512, endswith="ent")
    # frame.savefig("../animation/test.png")


    os.makedirs("../animation/frames.v2", exist_ok=True)
    np.random.seed(8)
    states = [random_quantum_state(4, prob=1.0) for _ in range(10)]
    states = np.array(states, dtype=np.complex64)
    n = 0
    for i in tqdm(range(0, 1001, 20)):
        agent = torch.load(os.path.join(AGENTS_PATH, f"agent{min(i+1, 1000)}.pt"))
        # Find the longest solution path
        actions_batch = []
        policy_batch = []
        entanglements_batch = []
        lens_batch = []
        for s in states:
            a, p, e = rollout(s, agent, max_steps=10)
            actions_batch.append(a)
            policy_batch.append(p)
            entanglements_batch.append(e)
            lens_batch.append(len(a))
        sorted_lens = np.argsort(lens_batch)
        j = sorted_lens[len(sorted_lens) // 2]
        actions, policy, entanglements = actions_batch[j], policy_batch[j], entanglements_batch[j]
        # Draw the longest trajectory
        for _ in range(1):
            frames = draw_trajectory(actions, policy, entanglements, iteration=max(1,i))
            for i, frame in enumerate(frames):
                frame.savefig(f"../animation/frames.v2/frame{n:04}.png")
                plt.close(frame)
                n += 1

    save_iteration_frames(1)
    save_iteration_frames(20)
    save_iteration_frames(40)
    save_iteration_frames(50)
    save_iteration_frames(100)
    save_iteration_frames(200)
    save_iteration_frames(300)
    save_iteration_frames(1000)