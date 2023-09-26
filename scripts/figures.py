import itertools
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import sys

file_path = os.path.split(os.path.abspath(__file__))[0]
project_dir = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.append(project_dir)
from src.agent import RandomAgent
from src.environment_loop import test_agent
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'


PATH_4Q_AGENT = os.path.join(project_dir, "logs/4q_10000_iters_haar_unif2_1024envs/agent.pt")
PATH_5Q_AGENT = os.path.join(project_dir, "logs/5q_20000_iters_haar_unif2_128envs/agent.pt")
PATH_6Q_AGENT = os.path.join(project_dir, "logs/6q_4000iters_haar_unif3_512/agent.pt")


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
        name.append('|R_{' + f'{numbers[i:i+len(r)]}' + '}\\rangle')
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
    entanglements.append(env.simulator.entanglements.copy().ravel())

    return np.array(actions), np.array(entanglements), np.array(probabilities)


def peek_policy(state):
    """Returns the agent probabilites for this state."""
    _, _, probabilities = rollout(state, max_steps=1)
    return probabilities[0]


def figure3(initial_state, selected_actions=None):
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
    MSA = 10        # maximum steps per ax
    QCR = 0.4       # qubits' circles radius
    QCX = -1        # qubits' circles X coordinate
    QFS = 24        # qubits' circles font size
    QLW = 3         # qubits' circles linewidth
    WLW = 1         # qubit wires linewidth
    AFS = 18        # $S_{avg}$ text fontsize
    AEY = -0.8      # $S_{avg}$ + per step avg. entropies Y coordinate
    EFS = 18        # single qubit entropies fontsize
    ERS = 0.6       # single qubit entropies background rectangle size
    ECR = 0.3       # single qubit entropies circle radius
    ECA = 0.1       # single qubit entropies circle alpha
    GOX = 0.5       # gate's X axis offset from single qubit entanglement circle
    GSS = 150       # gate's wire circle scatter size
    GLW = 4         # gate's wire linewidth
    PFS = 16        # gate's probability fontsize
    TLX = -1.5      # step timeline starting X coordinate
    TLW = 1.2       # step timeline linewidth

    # /// DERIVED LAYOUT CONSTANTS
    NAX = divmod(steps + 1, MSA)[0] + int(((steps+1) % MSA) > 0)    # number of axs
    FIGSIZE = (16, 9 * NAX)                                         # figsize

    # Initialize figure
    fig, axs = plt.subplots(NAX, 1, figsize=FIGSIZE, squeeze=False)

    # Draw qubit circles & "$S_{avg}$" text
    qubits_fontdict = dict(fontsize=QFS, horizontalalignment='center',
                           verticalalignment='center')
    avg_ent_fontdict = dict(fontsize=AFS, horizontalalignment='center',
                            verticalalignment='center', color='k')
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
    entanglement_fontdict = dict(fontsize=EFS, horizontalalignment='center',
                                 verticalalignment='center',
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
                fontdict=dict(fontsize=AFS, horizontalalignment='center',
                              verticalalignment='center', color='k'))

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
                rotation='vertical', verticalalignment='center'))

    for i, ax in enumerate(axs.flat, 1):
        # Set aspect & remove ticks
        ax.set_aspect('equal')
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        # Set limits
        ax.set_xlim(-2, MSA + 1)
        ax.set_ylim(-2, num_qubits)
        ax.set_xlabel('episode step', fontsize=18)
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
            ax.text(x, AEY - 1, str(lab), fontsize=18, horizontalalignment='center')

    return fig


def figure2(num_qubits, num_tests=10_000):

    TEST_STATES = {
        4: ["RR-R-R", "RR-RR", "RRR-R", "RRRR"],
        5: ["RR-R-R-R", "RR-RR-R", "RRR-R-R", "RRR-RR", "RRRR-R", "RRRRR"],
        6: ["RR-R-R-R-R", "RR-RR-R-R", "RR-RR-RR", "RRR-R-R-R", "RRR-RR-R",
            "RRRR-R-R", "RRRR-RR", "RRRRR-R", "RRRRRR"]
    }
    test_state_names = TEST_STATES[num_qubits]
    NUM_ENVS = 256

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

    fig, ax = plt.subplots(figsize=(2 + 2.2*len(test_state_names), 4))
    ax.set_axis_off()
    row_labels = ["num steps",
                  "final $S_{ent}$",
                  "\% solved",
                  "num steps",
                  "final $S_{ent}$",
                  "\% solved"]
    row_colors = ["#a2dce8", "#a2dce8", "#a2dce8",
                  "#e8bca2", "#e8bca2", "#e8bca2"]

    np.random.seed(4)
    num_actions = num_qubits * (num_qubits - 1) // 2
    col_labels = []
    cell_text = []

    # Do the tests
    for state_str in test_state_names:
        # Generate test states
        initial_states = np.array(
            [str2state(state_str) for _ in range(num_tests)])

        # Test RL agent
        RL_res = test_agent(agent, initial_states, num_envs=NUM_ENVS,
                            obs_fn="rdm_2q_mean_real", max_episode_steps=250)
        RL_avg_len = np.mean(RL_res["lengths"][RL_res["done"]])
        RL_std_len = np.std(RL_res["lengths"][RL_res["done"]])
        RL_avg_ent = np.mean(RL_res["entanglements"][RL_res["done"]])
        RL_std_ent = np.std(RL_res["entanglements"][RL_res["done"]])
        RL_solves  = np.mean(RL_res["done"])

        # Test random agent
        rand_res = test_agent(RandomAgent(num_actions), initial_states,
                                num_envs=NUM_ENVS, obs_fn="rdm_2q_mean_real",
                                max_episode_steps=250)
        rand_avg_len = np.mean(rand_res["lengths"][rand_res["done"]])
        rand_std_len = np.std(rand_res["lengths"][rand_res["done"]])
        rand_avg_ent = np.mean(rand_res["entanglements"][rand_res["done"]])
        rand_std_ent = np.std(rand_res["entanglements"][rand_res["done"]])
        rand_solves  = np.mean(rand_res["done"])

        # Create a table column for this kind of initial states
        col_labels.append(str2latex(state_str))
        col_cell_text = [
            f"{RL_avg_len:.2f} ± {RL_std_len:.2f}",
            f"{RL_avg_ent:.2E}",
            f"{100 * RL_solves:.2f}%",
            f"{rand_avg_len:.2f} ± {rand_std_len:.2f}",
            f"{rand_avg_ent:.2E}",
            f"{100 * rand_solves:.2f}%"
        ]
        cell_text.append(col_cell_text)

    cell_text = np.array(cell_text).T
    table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        loc="upper center", #colWidths=[0.9/n_cols] * n_cols,
        rowColours=row_colors, bbox=[0.05, 0.1, 1.0, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    ax.set_title(f"{num_qubits} Qubits")
    return fig


def figure1(initial_state, state_name=''):
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
    QFS = 18        # fontsize of qubits' text                      (L subfig)
    QLW = 2.0       # linewidth of qubits' circles                  (L subfig)
    WLW = 1.5       # linewidth of qubit wires                      (L subfig)
    PBW = 0.9       # width of single bar in "policy" subfigure     (R subfig)
    GLW = 4         # gate wire linewidth                           (L subfig)
    GSS = 180       # gate wire connection scatter size             (L subfig)

    # /// DERIVED LAYOUT CONSTANTS
    R_SUBFIG_XMIN = num_qubits + 1
    R_SUBFIG_XMAX = R_SUBFIG_XMIN + probs_main.shape[1] + 1
    WIRES_BOTTOM = QCY + QCR
    WIRES_TOP = QCY + nsteps + 4
    FIGSIZE = (16, 5 + nsteps)

    # Initialize figure
    fig, ax = plt.subplots(1, figsize=FIGSIZE)

    # Draw qubit circles with Y=`QCY`, X=[`QCX`, `QCX` + `num_qubits`]
    qubits_xs = np.arange(QCX, QCX + num_qubits)
    qubits_ys = np.full(num_qubits, QCY)
    for x, y in zip(qubits_xs, qubits_ys):
        ax.add_patch(patches.Circle((x, y), QCR, edgecolor='k', linewidth=QLW,
                                    fill=True, facecolor='white', zorder=10))
        ax.text(x, y, f'$q_{x+1}$',
                fontdict=dict(fontsize=QFS, horizontalalignment='center',
                              verticalalignment='center', zorder=11))

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
                linestyle='--', linewidth=0.8, color='k')

        # Draw main policy actions
        pmax = np.max(probs_main[n])
        ax.plot([R_SUBFIG_XMIN - PBW/2, R_SUBFIG_XMAX], [2*n - 1, 2*n - 1],
                color='k', linewidth=0.8)
        ax.text(R_SUBFIG_XMAX, 2 * n, f'$\pi^{(n)}(a|s)$',
                fontdict=dict(fontsize=14))
        for x, p in enumerate(probs_main[n]):
            color = 'tab:red' if p == pmax else 'tab:blue'
            x_coord = R_SUBFIG_XMIN + x
            y_coord = 2*n - 1
            bar = patches.Rectangle((x_coord - PBW/2, y_coord), PBW, p * 0.9,
                                    facecolor=color)
            ax.add_patch(bar)
            q0, q1 = action_labels[x]
            ax.text(x_coord, y_coord - 0.3, f'({q0+1}, {q1+1})',
                     fontdict=dict(horizontalalignment='center',
                                   verticalalignment='center', fontsize=14))
            ax.text(x_coord, y_coord + p + 0.1, f'${int(p * 100)}\%$',
                    fontdict=dict(horizontalalignment='center', fontsize=14))

        # Draw summarized "rest" actions
        p = probs_rest[n] * 0.9
        x_coord = R_SUBFIG_XMIN + len(probs_main[n])
        y_coord = 2*n - 1
        bar = patches.Rectangle((x_coord - PBW/2, y_coord), PBW, p, facecolor='tab:cyan')
        ax.add_patch(bar)
        ax.text(x_coord, y_coord - 0.3, 'rest', fontdict=dict(
            horizontalalignment='center', verticalalignment='center', fontsize=14))
        ax.text(x_coord, y_coord + p + 0.1, f'${int(p * 100)}\%$',
                fontdict=dict(horizontalalignment='center', fontsize=14))

    # Add "actions" text
    ax.text((R_SUBFIG_XMIN + R_SUBFIG_XMAX) / 2, QCY, 'actions',
            fontdict=dict(horizontalalignment='center', fontsize=14))
    # Add state name text
    ax.text(2, -3.5, state_name,
            fontdict=dict(fontsize=18, horizontalalignment='center'))

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
    print(peek_policy(initial_state))
    fig = figure3(initial_state, [0, 1, 2, 3, 4, 5, 6])
    fig = figure3(initial_state)
    fig.savefig('test_figure3.pdf')
    fig = figure1(initial_5q_states["|RRR-RR>"],
                  state_name=r'$|R_{123}\rangle|R_{45}\rangle$')
    
    # Figure 2
    fig4 = figure2(num_qubits=4, num_tests=1000)
    fig4.savefig('test_figure2_4.pdf')
    fig5 = figure2(num_qubits=5, num_tests=1000)
    fig5.savefig('test_figure2_5.pdf')
    fig6 = figure2(num_qubits=6, num_tests=1000)
    fig6.savefig('test_figure2_6.pdf')
