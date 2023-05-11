"""Inference
This script tests model inference.
The model is tested on a set of specific quantum states. For every state, on
every step of the episode, we plot the output from the attention heads, as well
as the output probability distribution.

The path to the folder containing the trained model must be provided as a
command line argument. The number of qubits must also be provided.

Example usage:

python3 inference.py \
    --seed 0 \
    --num_qubits 4 \
    --model_fld logs/4q_pGen_0.9_attnHeads_2_tLayers_2_ppoBatch_512_entReg_0.1_embed_128_mlp_256
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int, help="Seed for rng.")
parser.add_argument("--num_qubits", default=5, type=int, help="Number of qubits.")
parser.add_argument("--model_fld", default="logs",
    type=str, help="Filepath model folder.")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.set_printoptions(5, suppress=True)
torch.set_printoptions(5, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Definition of some interesting quantum states.
bell = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).astype(np.complex64)
              #000          #001          #010 #011          #100 #101 #110 #111
w = np.array([   0, 1/np.sqrt(3), 1/np.sqrt(3),   0, 1/np.sqrt(3),   0,   0,   0]).astype(np.complex64)
zero = np.array([1, 0]).astype(np.complex64)
one = np.array([0, 1]).astype(np.complex64)
def rnd_state(q): np.random.seed(0); return random_quantum_state(q=q, prob=1.)

special_states = {
    # 4-qubit special states
    4: {
        "|BB-0-0>": np.kron(bell, np.kron(zero, zero)),
        "|0-0-BB>": np.kron(zero, np.kron(zero, bell)),
        "|BB-R-R>": np.kron(bell, np.kron(rnd_state(q=1), rnd_state(q=1))),
        "|WWW-0>" : np.kron(w, zero),
        "|WWW-R>" : np.kron(w, rnd_state(q=1)),
        "|RR-R-R>": np.kron(rnd_state(q=2), np.kron(rnd_state(q=1), rnd_state(q=1))),
        "|RR-RR>" : np.kron(rnd_state(q=2), rnd_state(q=2)),
        "|RRR-0>" : np.kron(rnd_state(q=3), zero),
        "|RRR-R>" : np.kron(rnd_state(q=3), rnd_state(q=1)),
        "|RRRR>"  : rnd_state(q=4),
    },

    # 5-qubit special states
    5: {
        "|BB-0-0-0>": np.kron(bell, np.kron(zero, np.kron(zero, zero))),
        "|0-BB-0-0>": np.kron(zero, np.kron(bell, np.kron(zero, zero))),
        "|0-0-0-BB>": np.kron(zero, np.kron(zero, np.kron(zero, bell))),
        "|BB-R-R-R>": np.kron(bell, np.kron(rnd_state(q=1), np.kron(rnd_state(q=1), rnd_state(q=1)))),
        "|BB-BB-0>" : np.kron(bell, np.kron(bell, zero)),
        "|WWW-0-0>" : np.kron(w, np.kron(zero, zero)),
        "|WWW-BB>"  : np.kron(w, bell),
        "|RR-0-0-0>": np.kron(rnd_state(q=2), np.kron(zero, np.kron(zero, zero))),
        "|RR-R-R-R>": np.kron(rnd_state(q=2), np.kron(rnd_state(q=1), np.kron(rnd_state(q=1), rnd_state(q=1)))),
        "|RR-RR-0>" : np.kron(rnd_state(q=2), np.kron(rnd_state(q=2), zero)),
        "|RR-RR-R>" : np.kron(rnd_state(q=2), np.kron(rnd_state(q=2), rnd_state(q=1))),
        "|RRR-0-0>" : np.kron(rnd_state(q=3), np.kron(zero, zero)),
        "|RRR-R-R>" : np.kron(rnd_state(q=3), np.kron(rnd_state(q=1), rnd_state(q=1))),
        "|RRR-RR>"  : np.kron(rnd_state(q=3), rnd_state(q=2)),
        "|RRRR-0>"  : np.kron(rnd_state(q=4), zero),
        "|RRRR-R>"  : np.kron(rnd_state(q=4), rnd_state(q=1)),
        "|RRRRR>"   : rnd_state(q=5),
}}


def plot_model_output(figname, seq, attn, probs, acts):
    assert attn.shape[-2] == attn.shape[-1], "attention matrix must be square"
    assert len(seq) == attn.shape[-1], "attention matrix must match sequence length"

    n_layers, n_heads = attn.shape[0], attn.shape[1]
    fig, axs = plt.subplots(nrows=n_layers+1, ncols=n_heads, facecolor="lightgrey",
            figsize=(5*n_heads, 5*(n_layers+1)), tight_layout={"pad":5})

    # Plot attention matrices.
    for i in range(n_layers):
        for j in range(n_heads):
            axs[i, j].xaxis.grid()
            axs[i, j].yaxis.grid()
            axs[i, j].imshow(attn[i][j].T, vmin=0., vmax=1.)
            axs[i, j].set_title(f"L{i+1} H{j+1}")
            axs[i, j].set_xticks(np.arange(len(list(seq))), seq)
            axs[i, j].set_yticks(np.arange(len(list(seq))), seq)


    # Plot output probability distribution.
    ax = plt.subplot2grid(shape=(n_layers+1, n_heads), loc=(n_layers, 0), colspan=n_heads, fig=fig)
    xs = np.arange(len(probs))

    pps = ax.bar(xs, probs)
    for p in pps:
        height = p.get_height()
        ax.annotate("{:.3f}".format(height),
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3), # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')

    ax.set_xticks(xs, [acts[i] for i in xs])
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=14)

    fig.savefig(figname)
    plt.close(fig)


# Load the model for file.
model_fld = args.model_fld
agent = torch.load(os.path.join(model_fld, "agent.pt"))
agent.policy_network.to(device)
agent.value_network.to(device)
for enc in agent.policy_network.net:
    enc.activation_relu_or_gelu = 1

# Define the quantum environment.
num_qubits = args.num_qubits
steps_limit = 40 if num_qubits == 5 else 8
env = QuantumEnv(num_qubits=num_qubits, num_envs=1,
    max_episode_steps=steps_limit, obs_fn="rdm_2q_real")

plt.style.use("ggplot")
for sname, state in tqdm(special_states[num_qubits].items()):
    # Set the quantum state
    psi = state.reshape((1,) + (2,) * num_qubits)
    o, _ = env.reset()
    env.simulator.states = psi
    o = env.obs_fn(env.simulator.states)

    log_dir = os.path.join(model_fld, sname)
    os.makedirs(log_dir, exist_ok=True)

    # Simulate...
    path = []
    done = False
    s = 0
    while not done:
        o = torch.from_numpy(o).to(device)

        # Get the attention scores.
        with torch.no_grad():
            # Embed.
            emb = agent.policy_network.net[0](o)

            attn_weights = []
            # First layer is the embedding layer, last two layers are output layers.
            for i in range(1, len(agent.policy_network.net)-2):
                z_norm = agent.policy_network.net[i].norm1(emb)
                _, attn = agent.policy_network.net[i].self_attn(
                    z_norm, z_norm, z_norm, need_weights=True, average_attn_weights=False)
                emb = agent.policy_network.net[i](emb)
                attn_weights.append(attn.cpu().numpy())
            attn_weights = np.vstack(attn_weights)

        # Get the probabilities.
        pi = agent.policy(o)
        probs = pi.probs[0].cpu().numpy()

        # Make the plot.
        seq = [f"q{i}q{j}" for i, j in env.simulator.actions.values()]
        plot_model_output(os.path.join(log_dir, f"step_{s}.png"), seq, attn_weights, probs, env.simulator.actions)

        # Step the environment.
        acts = torch.argmax(pi.logits, dim=1)
        acts = acts.cpu().numpy()
        o, r, t, tr, i = env.step(acts)

        path.append(acts[0])
        done = (t | tr)[0]
        s += 1

#