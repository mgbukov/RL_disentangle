import json
import sys
from pathlib import Path

import numpy as np
import torch

PATH_SOURCE_CODE = Path(__file__).parent.parent.parent
sys.path.append(str(PATH_SOURCE_CODE))
from src.util import srollout, random_quantum_state




ROOT = Path(__file__).parent.parent.parent.parent
PATH_RAW = ROOT / "data" / "raw" / "fig8"
PATH_PROCESSED = ROOT / "data" / "processed" / "fig8"
PATH_AGENT = ROOT / "models" / "fig7-8" / "4q-agent.pt"


if __name__ == "__main__":

    PATH_RAW.mkdir(exist_ok=True)
    PATH_PROCESSED.mkdir(exist_ok=True)

    # Load agent
    agent = torch.load(PATH_AGENT, weights_only=False)
    for layer in agent.policy_network.net:
        if isinstance(layer, torch.nn.TransformerEncoderLayer):
            layer.activation_relu_or_gelu = 1

    # Action set
    actset = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Generate & save state
    np.random.seed(45)
    state = random_quantum_state(4, 1.0)
    np.save(str(PATH_RAW / "RRRR.npy"), state)

    # Save actions & probabilities
    acts, _, probs = srollout(state, agent)
    data = [
        {"qubits": a.tolist(), "probabilities": p.tolist(), "actions": actset}
            for a, p in zip(acts, probs)
    ]
    with open(PATH_PROCESSED / "RRRR-circuit.json", mode='wt') as f:
        json.dump(data, f, indent=2)
