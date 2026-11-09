"""Aggregates raw results for Greedy agents on 12q and 16q systems"""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent.parent.parent.parent
PATH_DATA = ROOT / "data" / "raw" / "fig11"
PATH_OUTPUT = ROOT / "data" / "processed" / "fig11"


if __name__ == "__main__":
    greedy_stats = {}

    # /// 12q results
    # Read data
    #   - Greedy agent
    with open(PATH_DATA / "greedy-12q-3x4-weakly_entangled.json", mode="rt") as f:
        data = json.load(f)
        results = np.array([x["steps"] for x in data["results"]])
        greedy_stats["3x4"] = (results.mean(), results.std())
    with open(PATH_DATA / "greedy-12q-4x3-weakly_entangled.json", mode="rt") as f:
        data = json.load(f)
        results = np.array([x["steps"] for x in data["results"]])
        greedy_stats["4x3"] = (results.mean(), results.std())
    with open(PATH_DATA / "greedy-12q-6x2-weakly_entangled.json", mode="rt") as f:
        data = json.load(f)
        results = np.array([x["steps"] for x in data["results"]])
        greedy_stats["6x2"] = (results.mean(), results.std())

    # /// 16q results
    # Read data
    #   - Greedy agent
    with open(PATH_DATA / "greedy-16q-4x4-weakly_entangled.json", mode="rt") as f:
        data = json.load(f)
        results = np.array([x["steps"] for x in data["results"]])
        greedy_stats["4x4"] = (results.mean(), results.std())
    with open(PATH_DATA / "greedy-16q-5x3-weakly_entangled.json", mode="rt") as f:
        data = json.load(f)
        results = np.array([x["steps"] for x in data["results"]])
        greedy_stats["5x3"] = (results.mean(), results.std())
    with open(PATH_DATA / "greedy-16q-8x2-weakly_entangled.json", mode="rt") as f:
        data = json.load(f)
        results = np.array([x["steps"] for x in data["results"]])
        greedy_stats["8x2"] = (results.mean(), results.std())

    # Save
    PATH_OUTPUT.mkdir(exist_ok=True)
    with open(PATH_OUTPUT / "greedy-eta4.1.json", mode='wt') as f:
        json.dump(greedy_stats, f, indent=2)
