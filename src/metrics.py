from collections import defaultdict
import json
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt


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
        with open(filepath, mode='wt') as f:
            json.dump(self.stats, f, indent=2)



    def plot_scalar(self, name, savepath, xlabel="iteration", **plotargs):
        if name not in self.stats:
            warnings.warn(f"No stats found for key=\"{name}\". "
                            "No plot will be saved.")
            return

        fig, ax = plt.subplots(figize=(16,9))
        ax.set_title(name)

        # Prepare plot data
        xs, ys, errs = [], [], []
        for item in self.stats[name]:
            xs.append(item[0])
            ys.append(item[1])
            errs.append(item[2])

        plotargs.update({"label": name})
        self._plot(fig, ax, xs, ys, errs, plotargs)
        ax.legend()

        # Save figure
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=240)
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
            self._plot(fig, ax, xs, ys, errs, plotargs)

        ax.set_xlabel(xlabel)
        ax.legend()

        # Save figure
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=240)
        plt.close(fig)
        return

    def _plot(self, fig, ax, xs, ys, errs, **plotargs):

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        errs = np.asarray(errs)

        if np.any(errs != 0.0):
            below = ys - 0.5 * errs
            above = ys + 0.5 * errs
            ax.fill_between(xs, below, above, color="tab:gray", alpha=0.5)
        ax.plot(xs, ys, **plotargs)

