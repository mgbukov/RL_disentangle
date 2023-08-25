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
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'


PATH_4Q_AGENT = ''
PATH_5Q_AGENT = os.path.join(project_dir, 'logs/5q_pGen_0.9_attnHeads_4_tLayers_4_ppoBatch_512_entReg_0.1_embed_256_mlp_512/agent.pt')
PATH_6Q_AGENT = ''


def peek_policy(initial_state):
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
        raise ValueError(f'Cannot find agent for {num_qubits}-qubit system.')
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()

    # Return policy
    observation = torch.from_numpy(env.obs_fn(env.simulator.states))
    policy = agent.policy(observation).probs[0].cpu().numpy()
    return policy


def figure3(initial_state, selected_actions=None):

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
        raise ValueError(f'Cannot find agent for {num_qubits}-qubit system.')
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
    steps = len(actions)

    # Append final entangments
    entanglements.append(env.simulator.entanglements.copy().ravel())

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
    #

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
    steps = len(actions)

    # Select actions that are to be shown in right subfigure
    policies = np.array(policies)
    masked_actions = np.max(policies, axis=0) > 0.05
    action_labels = list(itertools.combinations(range(num_qubits), 2))
    action_labels = np.array(action_labels)[masked_actions]
    # Plotted actions
    policies_main = policies[:, masked_actions]
    # Summarized actions (plotted in "rest" column)
    policies_rest = policies[:, ~masked_actions].sum(axis=1)

    # /// USER CONSTANTS
    QCY = -2        # Y coordinate of qubits' circles               (L subfig)
    QCX = 0         # Min X coordinate of qubits' circles           (L subfig)
    QCR = 0.45      # radius of qubits' circles                     (L subfig)
    QFS = 18        # fontsize of qubits' text                      (L subfig)
    QLW = 1.8       # linewidth of qubits' circles                  (L subfig)
    WLW = 1.2       # linewidth of gate wires                       (L subfig)
    PBW = 0.9       # width of single bar in "policy" subfigure     (R subfig)
    GRH = 0.6       # gate rectangle's height                       (L subfig)
    GRW = 0.2       # gate rectangle's extra width                  (L subfig)

    # /// DERIVED LAYOUT CONSTANTS
    R_SUBFIG_XMIN = num_qubits + 1
    R_SUBFIG_XMAX = R_SUBFIG_XMIN + policies_main.shape[1] + 1
    WIRES_BOTTOM = QCY + QCR
    WIRES_TOP = QCY + steps + 4
    FIGSIZE = (16, 5 + steps)


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
    for n in range(steps):
        # Draw gate
        q0, q1 = sorted(actions[n])
        rect = patches.Rectangle((q0 - GRW, 2*n), q1 - q0 + 2*GRW, GRH,
                                 facecolor='#85caff', edgecolor='k', linewidth=2,
                                 zorder=5)
        ax.add_patch(rect)
        for q in range(q0 + 1, q1):
            ax.plot([q, q], [2*n, 2*n + GRH], color='k', linewidth=WLW, zorder=6)
        # Draw horizontal line connecting policy and gate
        ax.plot([-1, R_SUBFIG_XMIN - PBW], [2*n - 1, 2*n - 1],
                linestyle='--', linewidth=0.8, color='k')

        # Draw main policy actions
        pmax = np.max(policies_main[n])
        ax.plot([R_SUBFIG_XMIN - PBW/2, R_SUBFIG_XMAX], [2*n - 1, 2*n - 1],
                color='k', linewidth=0.8)
        ax.text(R_SUBFIG_XMAX, 2 * n, f'$\pi^{(n)}(a|s)$',
                fontdict=dict(fontsize=14))
        for x, p in enumerate(policies_main[n]):
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
        p = policies_rest[n] * 0.9
        x_coord = R_SUBFIG_XMIN + len(policies_main[n])
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
    ax.set_ylim(-4, 2 * steps + 1)
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
    fig = figure3(initial_state)
    fig.savefig('test_figure3.pdf')
    fig = figure1(initial_5q_states["|RRR-RR>"], r'$|R_{123}\rangle|R_{45}\rangle$')
    fig.savefig('test_figure1.pdf')
