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
PATH_5Q_STATE   = ROOT / "data" / "raw" / "fig9" / "5q-state-seed=0.npy"
PATH_PROCESSED  = ROOT / "data" / "processed" / "fig9"


if __name__ == "__main__":
    # Load state & agent
    state = np.load(PATH_5Q_STATE)
    agent = torch.load(PATH_5Q_AGENT, weights_only=False)
    # Do rollout
    actions, entanglements, _probabilities = srollout(state, agent)
    probabilities = np.array([np.max(prob) for prob in _probabilities])

    PATH_PROCESSED.mkdir(exist_ok=True)

    with open(PATH_PROCESSED / "5q-state-seed=0-protocol.json", mode='wt') as f:
        data = {
            "qubits": actions.tolist(),
            "entanglements": (entanglements / np.log(2)).tolist(),
            "probability": probabilities.tolist()
        }
        json.dump(data, f, indent=2)
