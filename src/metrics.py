from collections import defaultdict
import json
import os
import pickle
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


_METRIC_TRACKERS_ = {}


def getTracker():
    if "train" not in _METRIC_TRACKERS_:
        _METRIC_TRACKERS_["train"] = Tracker("train")
    return _METRIC_TRACKERS_["train"]



class Tracker:

    def __init__(self, name):
        self.name = name
        self.timestep = 0
        self.stats = defaultdict(list)

    def step(self):
        self.timestep += 1

    def add_scalar(self, name: str, val: float, std=0.0, timestep=None):
        if timestep is None:
            self.stats[name].append((self.timestep, val, std))
        else:
            self.stats[name].append((timestep, val, std))

    def get_last_scalar(self, name: str):
        if name in self.stats:
            return self.stats[name][-1]
        else:
            warnings.warn(f"No stats found for key=\"{name}\".")
            return (np.nan, np.nan, np.nan)

    def get_names(self):
        return list(self.stats.keys())

    def get_last_n_scalars(self, name: str, n: int):
        if name not in self.stats:
            warnings.warn(f"No stats found for key=\"{name}\".")
            return n * [(np.nan, np.nan, np.nan)]
        else:
            items = self.stats[name]
            result = []
            for i in range(len(items) - n, len(items)):
                if i >= 0:
                    result.append(items[i])
                else:
                    result.append((np.nan, np.nan, np.nan))
            return result

    def reset(self):
        self.stats.clear()
        self.timestep = 0

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if filepath.endswith('.json'):
            with open(filepath, mode='wt') as f:
                json.dump(self.stats, f, indent=2)
        else:
            with open(filepath, mode='wb') as f:
                pickle.dump(self.stats, f)

    def state_dict(self):
        return dict(name=self.name, timestep=self.timestep, stats=self.stats)

    def load_state_dict(self, state_dict: dict):
        self.name = state_dict["name"]
        self.timestep = state_dict["timestep"]
        self.stats = state_dict["stats"]

    def plot_scalar(self, name, savepath, xlabel="iteration", **plotargs):
        if name not in self.stats:
            warnings.warn(f"No stats found for key=\"{name}\". "
                            "No plot will be saved.")
            return

        fig, ax = plt.subplots(figsize=(16,9))
        ax.set_title(name)

        # Prepare plot data
        xs, ys, errs = [], [], []
        for item in self.stats[name]:
            xs.append(item[0])
            ys.append(item[1])
            errs.append(item[2])

        plotargs.update({"label": name})
        self._plot(ax, xs, ys, errs, **plotargs)

        # Save figure
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.tight_layout()
        fig.savefig(savepath, dpi=120)
        plt.close(fig)
        return

    def plot_scalars(self, names, title, savepath, xlabel="iteration", **plotargs):

        fig, ax = plt.subplots(figsize=(16,9))
        ax.set_title(title)

        for name in names:
            if name not in self.stats:
                warnings.warn(f"No tracked stats found for key=\"{name}\". "
                                "No plot will be saved.")
                continue

            # Prepare plot data
            xs, ys, errs = [], [], []
            for item in self.stats[name]:
                xs.append(item[0])
                ys.append(item[1])
                errs.append(item[2])

            plotargs.update({"label": name})
            self._plot(ax, xs, ys, errs, **plotargs)

        ax.set_xlabel(xlabel)
        ax.legend()

        # Save figure
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.tight_layout()
        fig.savefig(savepath, dpi=120)
        plt.close(fig)
        return

    def _plot(self, ax, xs, ys, errs, **plotargs):
        mpl.rcParams['font.size'] = 22
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        errs = np.asarray(errs)

        # If more than 1000 points are about to be plot,
        # do a scatter plot with rolling mean line
        if len(ys) <= 10:
            if np.any(errs != 0.0):
                below = ys - 0.5 * errs
                above = ys + 0.5 * errs
                ax.fill_between(xs, below, above, color="tab:blue", alpha=0.2)
            ax.plot(xs, ys, **plotargs)
        else:
            ax.scatter(xs, ys, s=40, marker='o', c="tab:blue", ec=None, alpha=0.2)
            m = np.convolve(np.pad(ys, 5, mode="edge"), np.ones(11), 'valid') / 11
            plotargs.pop("color", None)
            ax.plot(xs, m, color="tab:red", **plotargs)
        ax.grid(True, which="both", axis="both", color="gray", linewidth=0.25)

