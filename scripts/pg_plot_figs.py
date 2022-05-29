"""
python3 pg_plot_figs.py --log_dir traj_4096_iters_10001_entreg_0.01
"""

import argparse
import os
import pickle
import sys
sys.path.append("..")

from src.infrastructure.logging import (
    plot_distribution, plot_entropy_curves, plot_loss_curve,
    plot_nsolved_curves, plot_return_curves, plot_nsteps)

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", dest="log_dir", type=str)
args = parser.parse_args()


# Create file to log output during training.
log_dir = os.path.join("..", "logs", "5qubits", args.log_dir)
log_probs_dir = os.path.join(log_dir, "probs")
os.makedirs(log_probs_dir, exist_ok=True)


# Plot the results.
with open(os.path.join(log_dir, "train_history.pickle"), "rb") as f:
    train_history = pickle.load(f)
with open(os.path.join(log_dir, "test_history.pickle"), "rb") as f:
    test_history = pickle.load(f)

plot_entropy_curves(train_history, os.path.join(log_dir, "final_entropy.png"))
plot_loss_curve(train_history, os.path.join(log_dir, "loss.png"))
plot_return_curves(train_history, test_history, os.path.join(log_dir, "returns.png"))
plot_nsolved_curves(train_history, test_history, os.path.join(log_dir, "nsolved.png"))
plot_nsteps(train_history, os.path.join(log_dir, "nsteps.png"))
# plot_distribution(train_history, args.log_every, log_probs_dir)

#