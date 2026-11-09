"""
Processes the raw data for Fig. 4 and saves it to data/processed/fig4/
"""
import json
import pickle
from pathlib import Path

import numpy as np



ROOT = Path(__file__).parent.parent.parent.parent
PATH_RAW = ROOT / "data" / "raw" / "fig4"
PATH_PROCESSED = ROOT / "data" / "processed" / "fig4"


if __name__ == "__main__":
    # Input paths
    train_hist_5q = PATH_RAW / "5q_final_train_history.pickle"
    acc_hist_5q   = PATH_RAW / "5q_testacc_train_history.pickle"

    tracker_6q    = PATH_RAW / "6q_8000iters_haar_unif3_2nd_tracker.pickle"
    acc_hist_6q   = PATH_RAW / "6q_testacc_train_history.pickle"

    tracker_12q   = PATH_RAW / "checkpointed_12q_10000iters_XL_eta4.1_v2.1continue_tracker.pickle"

    tracker_16q   = PATH_RAW / "checkpointed_16q_20000iters_XL_eta4.1_v2.1continue_tracker.pickle"

    # Create output directory
    PATH_PROCESSED.mkdir(exist_ok=True)

    ### Process data for the 5q agent
    # --------------------------------------------------------------------------
    with open(train_hist_5q, mode='rb') as f:
        stats = pickle.load(f)
        eplen_5q_x, eplen_5q_y = [], []
        for i, x in enumerate(stats):
            eplen_5q_y.append(x["Episode Length"]["avg"])
            eplen_5q_x.append(i)

    # Save epside lengths (scatter points)
    with open(PATH_PROCESSED / Path("episode-length-5q.json"), mode='wt') as f:
        data = {"x": eplen_5q_x[:10000], "y": eplen_5q_y[:10000]}
        json.dump(data, f, indent=2)

    # Save accuracy (line)
    acc_5q = []
    with open(acc_hist_5q, mode='rb') as f:
        stats = pickle.load(f)
        for i, x in enumerate(stats):
            acc_5q.append(x["Ratio Terminated"]["test_avg"])
    with open(PATH_PROCESSED / Path("accuracy-5q.json"), mode='wt') as f:
        data = {"x": list(range(len(acc_5q))), "y": acc_5q}
        json.dump(data, f, indent=2)


    ### Process data for the 6q agent
    # --------------------------------------------------------------------------
    with open(tracker_6q, mode='rb') as f:
        data = pickle.load(f)
        eplen_6q = np.array(data["Episode Length"])

    # Save epside lengths (scatter points)
    with open(PATH_PROCESSED / Path("episode-length-6q.json"), mode='wt') as f:
        data = {"x": eplen_6q[:12000,0].tolist(),
                "y": eplen_6q[:12000,1].tolist()}
        json.dump(data, f, indent=2)

    # Save accuracy (line)
    acc_6q = []
    with open(acc_hist_6q, mode='rb') as f:
        stats = pickle.load(f)
        for i, x in enumerate(stats):
            acc_6q.append(x["Ratio Terminated"]["test_avg"])
    with open(PATH_PROCESSED / Path("accuracy-6q.json"), mode='wt') as f:
        data = {"x": list(range(len(acc_6q))), "y": acc_6q}
        json.dump(data, f, indent=2)


    ### Process data for the 12q agent
    # --------------------------------------------------------------------------
    with open(tracker_12q, mode='rb') as f:
        data = pickle.load(f)
        eplen_12q = np.array(data["Episode Length"])
        ratio_12q = np.array(data["Ratio Terminated"])

    # Save episode lengths (scatter points)
    with open(PATH_PROCESSED / Path("episode-length-12q.json"), mode='wt') as f:
        data = {"x": eplen_12q[:18000,0].tolist(),
                "y": eplen_12q[:18000,1].tolist()}
        json.dump(data, f, indent=2)

    # Save accuracy (line)
    with open(PATH_PROCESSED / Path("accuracy-12q.json"), mode='wt') as f:
        data = {"x": ratio_12q[:18000,0].tolist(),
                "y": ratio_12q[:18000,1].tolist()}
        json.dump(data, f, indent=2)


    ### Process data for the 16q agent
    # --------------------------------------------------------------------------
    with open(tracker_16q, mode='rb') as f:
        data = pickle.load(f)
        eplen_16q = np.array(data["Episode Length"])
        ratio_16q = np.array(data["Ratio Terminated"])

    # Save episode lengths (scatter points)
    with open(PATH_PROCESSED / Path("episode-length-16q.json"), mode='wt') as f:
        data = {"x": eplen_16q[:18000,0].tolist(),
                "y": eplen_16q[:18000,1].tolist()}
        json.dump(data, f, indent=2)

    # Save accuracy (line)
    with open(PATH_PROCESSED / Path("accuracy-16q.json"), mode='wt') as f:
        data = {"x": ratio_16q[:18000,0].tolist(),
                "y": ratio_16q[:18000,1].tolist()}
        json.dump(data, f, indent=2)
