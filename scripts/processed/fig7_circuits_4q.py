import json
import sys
from pathlib import Path

import numpy as np
import torch

PATH_SOURCE_CODE = Path(__file__).parent.parent.parent
sys.path.append(str(PATH_SOURCE_CODE))
from src.util import srollout, random_quantum_state



ROOT = Path(__file__).parent.parent.parent.parent
PATH_RAW = ROOT / "data" / "raw" / "fig7"
PATH_PROCESSED = ROOT / "data" / "processed" / "fig7"
PATH_AGENT = ROOT / "models" / "fig7-8" / "4q-agent.pt"


if __name__ == "__main__":

    PATH_RAW.mkdir(exist_ok=True)
    PATH_PROCESSED.mkdir(exist_ok=True)

    # Load agent
    agent = torch.load(PATH_AGENT)
    for layer in agent.policy_network.net:
        if isinstance(layer, torch.nn.TransformerEncoderLayer):
            layer.activation_relu_or_gelu = 1
    agent.policy_network.eval()

    # Action set
    actset = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


    # Figure 4a (|Bell>|Bell> state)
    # --------------------------------------------------------------------------
    bell =  np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex64) / np.sqrt(2)
    bb = np.einsum("ij,kl -> ijkl", bell.reshape(2,2), bell.reshape(2,2))

    # Save state
    np.save(str(PATH_RAW / "Bell-Bell.npy"), bb)

    # Save actions & probabilities
    acts, _, probs = srollout(bb, agent)
    # probs = probs.round(3)
    data = [
        {"qubits": a.tolist(), "probabilities": p.tolist(), "actions": actset}
            for a, p in zip(acts, probs)
    ]
    with open(PATH_PROCESSED / "Bell-Bell-circuit.json", mode='wt') as f:
        json.dump(data, f, indent=2)


    # Figure 4b (|0>|GHZ> state)
    # --------------------------------------------------------------------------
    w = np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=np.complex64) / np.sqrt(3)
    zero = np.array([1, 0], dtype=np.complex64)
    zero_ghz = np.einsum("i,jkl -> ijkl", zero, w.reshape(2,2,2))

    # Save state
    np.save(str(PATH_RAW / "0-GHZ.npy"), zero_ghz)

    # Save actions & probabilities
    acts, _, probs = srollout(zero_ghz, agent)
    # probs = probs.round(3)
    data = [
        {"qubits": a.tolist(), "probabilities": p.tolist(), "actions": actset}
            for a, p in zip(acts, probs)
    ]
    with open(PATH_PROCESSED / "0-GHZ-circuit.json", mode='wt') as f:
        json.dump(data, f, indent=2)


    # Figure 4c (|RRR>|R> state)
    # --------------------------------------------------------------------------
    np.random.seed(23)
    rrr_r = np.einsum("ijk,l -> ijkl", random_quantum_state(3, 1.0), random_quantum_state(1, 1.0))

    # Save state
    np.save(str(PATH_RAW / "RRR-R.npy"), rrr_r)

    # Save actions & probabilities
    acts, _, probs = srollout(rrr_r, agent)
    # probs = probs.round(3)
    data = [
        {"qubits": a.tolist(), "probabilities": p.tolist(), "actions": actset}
            for a, p in zip(acts, probs)
    ]
    with open(PATH_PROCESSED / "RRR-R-circuit.json", mode='wt') as f:
        json.dump(data, f, indent=2)
