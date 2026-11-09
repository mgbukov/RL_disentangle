"""
Processes the raw data for Fig. 2 and saves it to data/processed/fig2/
"""
import json
import pickle
from pathlib import Path

import numpy as np


ROOT =  Path(__file__).parent.parent.parent.parent
PATH_DATA = ROOT / "data" / "raw"/ "fig2"
PATH_PROCESSED = ROOT / "data" / "processed" / "fig2"


if __name__ == "__main__":

    # Create output directory
    PATH_PROCESSED.mkdir(exist_ok=True)

    with open(PATH_DATA / "random-greedy-stats.pickle", mode="rb") as f:
        stats = pickle.load(f)

    system_sizes = [4,5,6,7,8]

    # Process data for the random agent
    # --------------------------------------------------------------------------
    step_index_disentangled = []
    for k in system_sizes:
        # Dimensions are (sample, state, single qubit entanglement)
        random_ent = stats[k]["random"]["entanglements"]
        random_maxsteps = max(len(x) for x in random_ent)
        avg_entanglements_per_step = []
        std_entanglements_per_step = []
        for i in range(random_maxsteps):
            ent_step_i = []
            for arr in random_ent:
                if len(arr) <= i:
                    ent_step_i.append(1e-4)
                else:
                    ent_step_i.append(np.mean(arr[i]))
            avg_entanglements_per_step.append(np.mean(ent_step_i))
            std_entanglements_per_step.append(np.std(ent_step_i))
        # Process
        avg_entanglements_per_step = np.array(avg_entanglements_per_step)
        std_entanglements_per_step = np.array(std_entanglements_per_step)
        step_index_disentangled.append(
            int(np.argmax(avg_entanglements_per_step < 1e-3))
        )
        # Save
        with open(PATH_PROCESSED / f"entanglements-random-{k}q.json", mode='wt') as f:
            data = {"avg": avg_entanglements_per_step.tolist(),
                    "std": std_entanglements_per_step.tolist(),
                    "x": list(range(len(avg_entanglements_per_step)))}
            json.dump(data, f, indent=2)

    with open(PATH_PROCESSED / f"steps-to-disentangle-random.json", mode='wt') as f:
        data = {"L": system_sizes, "steps": step_index_disentangled}
        json.dump(data, f, indent=2)


    step_index_disentangled.clear()

    # Process data for the greedy agent
    # --------------------------------------------------------------------------
    for k in system_sizes:
        # Dimensions are (sample, state, single qubit entanglement)
        greedy_ent = stats[k]["greedy"]["entanglements"]
        greedy_maxsteps = max(len(x) for x in greedy_ent)
        avg_entanglements_per_step = []
        std_entanglements_per_step = []
        for i in range(greedy_maxsteps):
            ent_step_i = []
            for arr in greedy_ent:
                if len(arr) <= i:
                    ent_step_i.append(1e-4)
                else:
                    ent_step_i.append(np.mean(arr[i]))
            avg_entanglements_per_step.append(np.mean(ent_step_i))
            std_entanglements_per_step.append(np.std(ent_step_i))
        # Process
        avg_entanglements_per_step = np.array(avg_entanglements_per_step)
        std_entanglements_per_step = np.array(std_entanglements_per_step)
        step_index_disentangled.append(
            int(np.argmax(avg_entanglements_per_step < 1e-3))
        )
        # Save
        with open(PATH_PROCESSED / f"entanglements-greedy-{k}q.json", mode='wt') as f:
            data = {"avg": avg_entanglements_per_step.tolist(),
                    "std": std_entanglements_per_step.tolist(),
                    "x": list(range(len(avg_entanglements_per_step)))}
            json.dump(data, f, indent=2)

    with open(PATH_PROCESSED / f"steps-to-disentangle-greedy.json", mode='wt') as f:
        data = {"L": system_sizes, "steps": step_index_disentangled}
        json.dump(data, f, indent=2)


