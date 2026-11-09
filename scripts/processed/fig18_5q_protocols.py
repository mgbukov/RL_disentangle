import json
import sys
from pathlib import Path

import numpy as np
import torch

PATH_SOURCE_CODE = Path(__file__).parent.parent.parent.parent / Path("src_code")
sys.path.append(str(PATH_SOURCE_CODE))
from src.util import srollout



ROOT            = Path(__file__).parent.parent.parent.parent
PATH_5Q_AGENT   = ROOT / "models" / "5q-agent.pt"
PATH_5Q_STATES  = ROOT / "data" / "raw" / "fig18"
PATH_OUTPUT     = ROOT / "data" / "processed" / "fig18/"


if __name__ == "__main__":

    # Load agent
    agent = torch.load(PATH_5Q_AGENT, weights_only=False)
    PATH_OUTPUT.mkdir(exist_ok=True)

    # Load state
    for seed in (4, 240, 281):
        state = np.load(PATH_5Q_STATES / f"5q-state-seed={seed}.npy")

        # Do rollout
        actions, entanglements, _probabilities = srollout(state, agent)
        probabilities = np.array([np.max(prob) for prob in _probabilities])

        with open(PATH_OUTPUT / f"5q-state-seed={seed}-protocol.json", mode='wt') as f:
            data = {
                "qubits": actions.tolist(),
                "entanglements": (entanglements / np.log(2)).tolist(),
                "probability": probabilities.tolist()
            }
            json.dump(data, f, indent=2)
