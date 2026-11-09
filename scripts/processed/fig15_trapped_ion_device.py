import json
from pathlib import Path

import numpy as np


ROOT            = Path(__file__).parent.parent.parent.parent
PATH_MATT       = ROOT / "data" / "raw" / "fig15"
PATH_PROCESSED  = ROOT / "data" / "processed" / "fig15"
# Action set — same order as dist_*.npy probability arrays
ACTSET = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def load_circuit_data(folder: Path):
    """
    Load precomputed agent policy distributions and chosen actions for all steps.
    Returns list of dicts in the fig7 JSON format.
    """
    data = []
    step = 0
    while True:
        dist_path = folder / f"dist_{step}.npy"
        idxs_path = folder / f"idxs_agent_{step}.npy"
        if not dist_path.exists():
            break
        probs = np.load(dist_path).tolist()
        qubits = np.load(idxs_path).tolist()
        data.append({"qubits": qubits, "probabilities": probs, "actions": ACTSET})
        step += 1
    return data


if __name__ == "__main__":
    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Bell-Bell state
    data = load_circuit_data(PATH_MATT / "bell_bell")
    with open(PATH_PROCESSED / "Bell-Bell-circuit.json", mode="wt") as f:
        json.dump(data, f, indent=2)

    # GHZ state
    data = load_circuit_data(PATH_MATT / "ghz")
    with open(PATH_PROCESSED / "GHZ-circuit.json", mode="wt") as f:
        json.dump(data, f, indent=2)

    # W state
    data = load_circuit_data(PATH_MATT / "w")
    with open(PATH_PROCESSED / "W-circuit.json", mode="wt") as f:
        json.dump(data, f, indent=2)

    # Shallow random circuit state
    data = load_circuit_data(PATH_MATT / "shallow")
    with open(PATH_PROCESSED / "Shallow-circuit.json", mode="wt") as f:
        json.dump(data, f, indent=2)
