import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PATH_SOURCE_CODE = Path(__file__).parent.parent.parent
sys.path.append(str(PATH_SOURCE_CODE))
from src.stategen import sample_haar_generalized
from src.qenv import VectorizedQState
from src.util import srollout



ROOT = Path(__file__).parent.parent.parent.parent
PATH_PROCESSED = ROOT / "data" / "processed" / "fig22"


def pool_entanglements(pool: dict, num_qubits: int, trajectory=None, agent=None):
    qsim = VectorizedQState(num_qubits, 1, fast_ents=True, swaps=False, device="cpu")
    entanglements = defaultdict(list)

    for eta in pool.keys():
        for state in pool[eta]:
            if agent is not None:
                _, ents, _ = srollout(state, agent, max_steps=30)
                entanglements[eta].append(ents[-1])
            else:
                # Add `num_envs` dimension and set state
                qsim.states = np.expand_dims(state, 0)
                # Apply trajectory if any
                if trajectory is not None:
                    for pair in trajectory:
                        qsim.apply([pair])
                # Measure entanglements
                ents = qsim.entanglements[0].numpy()
                entanglements[eta].append(ents)
    return {k: np.array(v) for k,v in entanglements.items()}


def data_eta_choice():
    N = 25

    optimal_trajectory_12q = [
        (0, 1), (2, 3), (0, 2), (1, 3), (2, 3),
        (4, 5), (6, 7), (4, 6), (5, 7), (6, 7),
        (8, 9), (10, 11), (8, 10), (9, 11), (10,11)
    ]

    etas = [round(float(x), 2) for x in np.arange(3.0, 5.6, 0.2)]
    statepool_12 = defaultdict(list)
    for eta in etas:
        for _ in range(N):
            s = sample_haar_generalized(12, 4, 4, eta, eta, permute=False)
            statepool_12[eta].append(s)
    entanglements_12 = pool_entanglements(statepool_12, 12, trajectory=optimal_trajectory_12q)
    xs = sorted(entanglements_12.keys())
    scatter_x = []
    scatter_y = []
    line_x = []
    line_y = []
    for x in xs:
        ys = entanglements_12[x].max(axis=1)
        scatter_x.extend([x] * N)
        scatter_y.extend(ys.tolist())
        line_x.append(x)
        line_y.append(float(np.mean(ys)))
    return {
        "scatter_x": scatter_x,
        "scatter_y": scatter_y,
        "line_x": line_x,
        "line_y": line_y
    }


if __name__ == "__main__":

    PATH_PROCESSED.mkdir(exist_ok=True)

    np.random.seed(16)
    data = data_eta_choice()
    with open(PATH_PROCESSED / "leftover-entanglements.json", mode='wt') as f:
        json.dump(data, f, indent=2)