import argparse
import os

import numpy as np
from tqdm import tqdm

from context import *
from src.stategen import sample_haar_generalized


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits", "-q", type=int)
    parser.add_argument("--memmap", "-m", action="store_true")
    parser.add_argument("--min_subsystem_size", "-a", type=int)
    parser.add_argument("--max_subsystem_size", "-b", type=int)
    parser.add_argument("--min_eta", type=float)
    parser.add_argument("--max_eta", type=float)
    parser.add_argument("--num_states", "-n", type=int)
    parser.add_argument("--output", "-o", type=str)

    args = parser.parse_args()

    savepath = os.path.dirname(args.output)
    basename = os.path.basename(args.output)
    if args.memmap:
        filename_data = basename if basename.endswith(".npmm") else basename + ".npmm"
    else:
        filename_data = basename if basename.endswith(".npy") else basename + ".npy"
    filename_sidecard = os.path.splitext(filename_data)[0] + ".sidecard"
    path_data = os.path.join(savepath, filename_data)
    path_sidecard = os.path.join(savepath, filename_sidecard)


    if os.path.exists(path_data):
        print(f"\nError: File \"{path_data}\" already exists!\n")
        exit()

    if os.path.exists(path_sidecard):
        print(f"\nError: File \"{path_sidecard}\" already exists!\n")
        exit()

    if args.memmap:
        # Calculate output shape and create memory mapped file
        out_shape = (args.num_states,) + (2,) * args.num_qubits
        fp = np.memmap(path_data, dtype='complex64', mode='w+', shape=out_shape)
    else:
        temp = []

    for i in tqdm(range(args.num_states)):
        state = sample_haar_generalized(
            args.num_qubits,
            args.min_subsystem_size,
            args.max_subsystem_size,
            args.min_eta,
            args.max_eta,
            True
        )
        if args.memmap:
            fp[i] = state
        else:
            temp.append(state)

    if args.memmap:
        fp.flush()
    else:
        arr = np.array(temp)
        with open(path_data, mode="wb") as f:
            np.save(f, arr)

    # Save sidecard
    with open(os.path.join(savepath, filename_sidecard), mode="wt") as f:
        f.write(f"filename={path_data}\n")
        f.write(f"num_states={args.num_states}\n")
        f.write(f"num_qubits={args.num_qubits}\n")
        f.write(f"min_subsystem_size={args.min_subsystem_size}\n")
        f.write(f"max_subsystem_size={args.max_subsystem_size}\n")
        f.write(f"min_eta={args.min_eta}\n")
        f.write(f"max_eta={args.max_eta}")