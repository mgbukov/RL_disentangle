"""
python3 il_plot_figs.py --log_dir imitation_100k
"""

import argparse
import os
import pickle
import sys
sys.path.append("..")

from src.infrastructure.logging import plot_entropy_curves, plot_loss_curve, logPlot


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", dest="log_dir", type=str)
args = parser.parse_args()

# Create file to log output during training.
log_dir = os.path.join("..", "logs", "5qubits", args.log_dir)


# Plot the results.
with open(os.path.join(log_dir, "train_history.pickle"), "rb") as f:
    train_history = pickle.load(f)
with open(os.path.join(log_dir, "test_history.pickle"), "rb") as f:
    test_history = pickle.load(f)

test_returns = [test_history[i]["returns"].mean() for i in sorted(test_history.keys())]
test_nsolved = [test_history[i]["nsolved"].mean() for i in sorted(test_history.keys())]
test_nsteps = [test_history[i]["nsteps"].mean() for i in sorted(test_history.keys())]


# Plot curves.
plot_loss_curve(train_history, os.path.join(log_dir, "training_loss.png"), lw=1.4)
plot_entropy_curves(test_history, os.path.join(log_dir, "entropy.png"),
    lw=[2., 2., 2., 2.])
logPlot(figname= os.path.join(log_dir, "returns.png"),
        xs=[sorted(test_history.keys())], funcs=[test_returns],
        legends=["test_returns"], labels={"x":"Episode", "y":"Return"},
        fmt=["-r"], lw=[1.4],
        figtitle="Agent achieved average return")
logPlot(figname= os.path.join(log_dir, "nsolved.png"),
        xs=[sorted(test_history.keys())], funcs=[test_nsolved],
        legends=["test_nsolved"], labels={"x":"Episode", "y":"nsolved"},
        fmt=["-r"], lw=[1.4],
        figtitle="Agent accuracy of solved states")
logPlot(figname= os.path.join(log_dir, "nsteps.png"),
        xs=[sorted(test_history.keys())], funcs=[test_nsteps],
        legends=["test_nsteps"], labels={"x":"Episode", "y":"nsteps"},
        fmt=["-r"], lw=[1.4],
        figtitle="Avg. number of steeps to disentangle")

#