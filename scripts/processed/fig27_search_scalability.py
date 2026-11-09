"""
Generates the search-stats.json for Fig.27.
The script will produce different data than the one already in
data/processed/fig27, because the random seed was not fixed the first time.
The output JSON will be prefixed with underscore in order to not overwritte
the original JSON.

The 'average_visited' field is left as empty string - the code for its
computation is lost.
"""
import json
import time
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from src.quantum_state import VectorQuantumState

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT / "src_code"))
from search import BeamSearch, SearchExpert



PATH_PROCESSED = ROOT / "data" / "processed" / "fig27"


if __name__ == "__main__":

    num_test = 1000
    num_qubits = [5, 6, 8, 10]
    results = {q : {} for q in num_qubits}

    for q in num_qubits:
        print(f"Testing {q} qubits..")

        env = VectorQuantumState(num_qubits=q, num_envs=1)

        epsi = 1e-3 if q <= 8 else 1e-2
        beam = BeamSearch(beam_size=100, epsi=epsi)
        beam_qbyq = SearchExpert(beam_size=100, epsi=epsi)

        agents = {"beam": beam, "beam_qbyq": beam_qbyq}

        for name, a in agents.items():
            print(f"  running {name} agent")

            path_len, solution_time = [], []
            entanglements = []
            for i in tqdm(range(num_test)):
                env.set_random_states_()
                psi = env.states[0]

                tic = time.time()
                path, ents = a.start(psi, env, verbose=False)
                toc = time.time()

                if path is None: continue # skip non-solved

                path_len.append(len(path))
                entanglements.append(ents)
                solution_time.append(toc-tic)

            if len(path_len) == 0: continue # skip systems for which no solution was found

            results[q].update({
                name: {
                    "nsteps": np.array(path_len),
                    "elapsed": np.array(solution_time),
                    "entanglements": entanglements,
                    "average_length": float(f"{np.array(path_len).mean():.1f}"),
                    "average_time": float(f"{np.array(solution_time).mean():.3f}"),
                    "average_visited": ""
                },
            })

    with open(PATH_PROCESSED / "_search_stats.json", "w") as f:
        json.dump(results, f, indent="  ")
