"""
Processing script for FIG. 13.

Reads per-seed .npy files from data/raw/fig13/, computes seed-averaged
entanglement entropies, and writes a single JSON to data/processed/fig13/
containing only the arrays that are plotted.

Run from the repo root:
    python src_code/scripts/processed/fig13_nisq_noise.py
"""
import json
from pathlib import Path

import numpy as np



ROOT = Path(__file__).parent.parent.parent.parent
PATH_RAW       = ROOT / "data" / "raw" / "fig13"
PATH_PROCESSED = ROOT / "data" / "processed" / "fig13"
NOISE_STATES = ["haar_random2", "my_custom_circuit", "bell_bell"]
SHOTS_STATES = ["haar_random",  "my_random",         "bell_bell2"]
PLOT_STEPS   = [2, 3, 5]
SEEDS        = np.arange(100)
LENGTH       = 15


def _load_noise(path_raw):
    n = len(NOISE_STATES)
    ents            = np.zeros((n, len(SEEDS), 6, LENGTH + 1))
    ents_noisy      = np.zeros((n, len(SEEDS), 6, LENGTH + 1))
    full_ents_noisy = np.zeros((n, len(SEEDS), 6, LENGTH + 1))

    for s, state in enumerate(NOISE_STATES):
        path = path_raw / "artificial_noise" / state
        for j, seed in enumerate(SEEDS):
            ents[s, j]            = np.load(path / f"__2ents_agent_4_{seed}.npy")[0]
            ents_noisy[s, j]      = np.load(path / f"__2ents_agent_noisy_4_{seed}.npy")[0]
            full_ents_noisy[s, j] = np.load(path / f"__2full_ents_agent_noisy_4_{seed}.npy")[0]

    return np.mean(ents, axis=1), np.mean(ents_noisy, axis=1), np.mean(full_ents_noisy, axis=1)


def _load_shots(path_raw):
    n = len(SHOTS_STATES)
    ents       = np.zeros((n, len(SEEDS), 5, LENGTH + 1))
    ents_noisy = np.zeros((n, len(SEEDS), 5, LENGTH + 1))

    for s, state in enumerate(SHOTS_STATES):
        path = path_raw / "shots" / state
        for j, seed in enumerate(SEEDS):
            ents[s, j]       = np.load(path / f"__2ents_agent_{seed}.npy")[0]
            ents_noisy[s, j] = np.load(path / f"__2ents_agent_noisy_{seed}.npy")[0]

    return np.mean(ents, axis=1), np.mean(ents_noisy, axis=1)


if __name__ == "__main__":
    print("Loading noise data...")
    ents_noise_mean, ents_noise_noisy_mean, full_ents_noise_noisy_mean = _load_noise(PATH_RAW)

    print("Loading shots data...")
    ents_shots_mean, ents_shots_noisy_mean = _load_shots(PATH_RAW)

    data = {
        "err_rates": [0.1 ** i for i in range(1, 7)],
        "shotss":    [10  ** i for i in range(2, 7)],
        "noise": {
            state: {
                "ents_mean":       {str(k): ents_noise_mean[s, :, k].tolist()       for k in PLOT_STEPS},
                "ents_noisy_mean": {str(k): ents_noise_noisy_mean[s, :, k].tolist() for k in PLOT_STEPS},
                **({"full_ents_noisy_mean_step3": full_ents_noise_noisy_mean[s, :, 3].tolist()}
                   if state == "bell_bell" else {}),
            }
            for s, state in enumerate(NOISE_STATES)
        },
        "shots": {
            state: {
                "ents_mean":       {str(k): ents_shots_mean[s, :, k].tolist()       for k in PLOT_STEPS},
                "ents_noisy_mean": {str(k): ents_shots_noisy_mean[s, :, k].tolist() for k in PLOT_STEPS},
            }
            for s, state in enumerate(SHOTS_STATES)
        },
    }

    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PATH_PROCESSED / "fig13_nisq_noise.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {out}")
