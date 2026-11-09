import sys
from pathlib import Path

import numpy as np
import torch

PATH_SOURCE_CODE = Path(__file__).parent.parent.parent
sys.path.append(str(PATH_SOURCE_CODE))
from src.quantum_state import random_quantum_state
from src.quantum_env import QuantumEnv



ROOT            = Path(__file__).parent.parent.parent.parent
PATH_4Q_AGENT   = ROOT / "models" / "4q-agent.pt"
OUTPUT_DIR      = ROOT / "data" / "processed" / "fig25"


def get_attention_scores(agent, state):

    env = QuantumEnv(4, 1, obs_fn='rdm_2q_mean_real')
    env.reset()
    env.simulator.states = np.expand_dims(state, 0)
    observation = torch.from_numpy(env.obs_fn(env.simulator.states)).float()

    # Get attention scores
    with torch.no_grad():
        emb = agent.policy_network.net[0](observation)
        attn_weights = []
        # First layer is the embedding layer, last two layers are output layers
        for i in range(1, len(agent.policy_network.net) - 2):
            z_norm = agent.policy_network.net[i].norm1(emb)
            _, attn = agent.policy_network.net[i].self_attn(
                z_norm, z_norm, z_norm,
                need_weights=True, average_attn_weights=False
            )
            emb = agent.policy_network.net[i](emb)
            attn_weights.append(attn.numpy())
        attn_weights = np.array(attn_weights)
        # Swap (layer, batch, head, X, Y) dimensions to (batch, layer, head, X, Y)
        attn_weights = attn_weights.swapaxes(0, 1)
        # Squeeze dim 0 from `attn_weights` in any case and catch the ValueError.
        try:
            attn_weights = attn_weights.squeeze(0)
        except ValueError:
            pass
    return attn_weights


if __name__ == "__main__":

    # Load agent
    agent = torch.load(PATH_4Q_AGENT, map_location=torch.device("cpu"),
                       weights_only=False)
    for layer in agent.policy_network.net:
        layer.activation_relu_or_gelu = 1

    OUTPUT_DIR.mkdir(exist_ok=True)

    #   |0>|Bell>|0>
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex64) / np.sqrt(2)
    bell = bell.reshape(2,2)
    zero = np.array([1, 0], dtype=np.complex64)
    bell_zero = np.einsum("ij,k->ijk", bell, zero)
    zero_bell_zero = np.einsum("i,jkl->ijkl", zero, bell_zero)
    attn_distribution = get_attention_scores(agent, zero_bell_zero)
    np.save(OUTPUT_DIR / "0_Bell_0.npy", attn_distribution)

    #   |RR>|RR>
    np.random.seed(1)
    subsysA = random_quantum_state(2, 1.0)
    subsysB = random_quantum_state(2, 1.0)
    RR_RR = np.einsum("ij,kl->ijkl", subsysA, subsysB)
    attn_distribution = get_attention_scores(agent, RR_RR)
    np.save(OUTPUT_DIR / "RR_RR-seed=1.npy", attn_distribution)
