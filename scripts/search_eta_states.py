import argparse
import math
import json

import numpy as np

import context
from src.quantum_state import VectorQuantumState
from src.stategen import sample_haar_product, eta_perturb
from search import RandomAgent, BeamSearch, UBSearch


EPSI = 1e-3
BEAM_SIZE = 50
UB_TRIALS = 20


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", choices=["random", "beamsearch", "ub"], type=str, help="Agent", default="random")
    parser.add_argument("--max_iters", "-n", type=int, default=10_000)
    parser.add_argument("-q", "--qubits", action="store", type=int, help="Number of qubits", default=12)
    parser.add_argument("-p", "--subsystem_size", type=int, help="Subsystem / partition size", default=4)
    # parser.add_argument("-b", "--bonds", action="extend", nargs="+", type=int, help="Bond locations")
    parser.add_argument("--min_eta", type=float, default=3.0)
    parser.add_argument("--max_eta", type=float, default=4.2)
    parser.add_argument("--eta_increment", type=float, default=0.05)
    parser.add_argument("-s", "--samples", type=int, default=10)
    parser.add_argument("--output", "-o", type=str, default='ttt.json')

    args = parser.parse_args()
    print(args)

    # Initialize agent
    if args.agent == "random":
        agent = RandomAgent(EPSI)
        print("Initialized Random agent")
    elif args.agent == "beamsearch":
        agent = BeamSearch(BEAM_SIZE, EPSI)
        print("Initialzied Beam Search agent")
    elif args.agent == "ub":
        agent = UBSearch(epsi=EPSI, trials=UB_TRIALS)
        print("Initialized UBSearch agent")
    else:
        raise ValueError("Unknown agent")

    # Initialize eta values
    etas = np.arange(args.min_eta, args.max_eta + args.eta_increment, args.eta_increment)
    etas = np.round(etas, 3)[::-1]
    print("Initialized etas:", etas)

    # Initialize Quantum State simmulator
    qstatesim = VectorQuantumState(args.qubits, 1)

    # Calculate bond locations
    temp = np.full(args.qubits, False, dtype=bool)
    temp[::args.subsystem_size] = True
    bonds = list(np.nonzero(temp)[0]) + [args.qubits]
    print("Bond locations:", bonds)

    search_stats = {}

    for eta in etas:
        temp = []
        print(f"\n\nTesting agent on states with eta = {eta}:\n\t", end='', flush=True)
        for _ in range(args.samples):
            prod = sample_haar_product(args.qubits, args.subsystem_size, args.subsystem_size, False)
            state = eta_perturb(prod, eta, bonds)
            traj = agent.start(state, qstatesim, num_iters=args.max_iters, verbose=False)
            if traj is None:
                temp.append(math.nan)
                print("NaN", end=' ', flush=True)
            else:
                temp.append(len(traj))
                print(len(traj), end=' ', flush=True)

        search_stats[eta] = temp

        with open(args.output, mode='wt') as f:
            json.dump(search_stats, f, indent=2)

    with open(args.output, mode='wt') as f:
        json.dump(search_stats, f, indent=2)

    print('\n')
