import os

import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(self, log_dir):
        """ Initialize logger object.

        @param log_dir (str): Directory for logging outputs.
        """
        self._log_dir = log_dir
        self._txt_filename = None
        self._verbose = False
        self.reset()


    @property
    def log_dir(self):
        return self._log_dir


    def logTxt(self, s):
        """ Log a text string @s to a file. """
        if self._txt_filename is None:
            raise ValueError("Set filename to write to.")
        with open(os.path.join(self._log_dir, self._txt_filename), "a") as f:
            f.write(s)
            f.write("\n")
            if self._verbose:
                print(s)


    def logPlot(self, xs=None, funcs=[], legends=[None], labels={}, fmt=["--k"], lw=[0.8],
                fills=[], figtitle="", figname=None, logscaleX=False, logscaleY=False):
        """ Plot @funcs as curves on a figure and save the figure as @figname.

        @param xs (List[np.Array]): List of arrays of x-axis data points.
        @param funcs (List[np.Array]): List of arrays of data points. Every array
                of data points from the list is plotted as a curve on the figure.
        @param legends (List[str]): A list of labels for every curve that will be displayed
                in the legend.
        @param labels (Dict): A map specifying the labels of the coordinate axes.
                `labels["x"]` specifies the label of the x-axis
                `labels["y"]` specifies the label of the y-axis
        @param fmt (List[str]): A list of formating strings for every curve.
        @param lw (List[float]): A list of line widths for every curve.
        @param fills (List[Tuple(np.Array)]): A list of tuples of curves [(f11,f21), (f12,f22), ...].
                Fill the area between the curves f1 and f2.
        @param figtitle (str): Figure title.
        @param figname (str): Save the figure to a file.
        @param logscaleX (bool): If True, plot the x-axis on a logarithmic scale.
        @param logscaleY (bool): If True, plot the y-axis on a logarithmic scale.
        """
        if xs is None:
            xs = [np.arange(len(f)) for f in funcs]
        if len(legends) == 1:
            legends = legends * len(funcs)
        if len(fmt) == 1:
            fmt = fmt * len(funcs)
        if len(lw) == 1:
            lw = lw * len(funcs)

        # Set figure sizes.
        fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
        ax.set_title(figtitle, fontsize=30, pad=30)
        ax.set_xlabel(labels.get("x"), fontsize=24, labelpad=30)
        ax.set_ylabel(labels.get("y"), fontsize=24, labelpad=30)
        ax.tick_params(axis="both", which="both", labelsize=20)
        ax.grid()
        if logscaleX:
            ax.set_xscale("log")
        if logscaleY:
            ax.set_yscale("log")

        # Plot curves.
        for x, f, l, c, w in zip(xs, funcs, legends, fmt, lw):
            ax.plot(x, f, c, label=l, linewidth=w)
        for f1, f2 in fills:
            x = np.arange(len(f1))
            ax.fill_between(x, f1, f2, color='k', alpha=0.25)

        ax.legend(loc="upper left", fontsize=20)
        fig.savefig(os.path.join(self._log_dir, figname))


    def logHistogram(self, func, bins, figname, figtitle="", labels={}, align="mid", rwidth=1.0):
        """ Plot @func as a histogram on a figure and save the figure as @figname. """
        fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
        ax.set_title(figtitle, fontsize=30, pad=30)
        ax.set_xlabel(labels.get("x"), fontsize=24, labelpad=30)
        ax.set_ylabel(labels.get("y"), fontsize=24, labelpad=30)
        ax.tick_params(axis="both", which="both", labelsize=20)
        ax.grid()
        ax.hist(func, bins, align=align, rwidth=rwidth)
        fig.savefig(os.path.join(self._log_dir, figname))


    def logBarchart(self, x, func, figname, figtitle ="", labels={}):
        """ Plot @func as a bar chart on a figure and save the figure as @figname. """
        fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
        ax.set_title(figtitle, fontsize=30, pad=30)
        ax.set_xlabel(labels.get("x"), fontsize=24, labelpad=30)
        ax.set_ylabel(labels.get("y"), fontsize=24, labelpad=30)
        ax.tick_params(axis="both", which="both", labelsize=20)
        ax.grid()
        ax.bar(x, func)
        fig.savefig(os.path.join(self._log_dir, figname))


    def logPcolor(self, func, figname, figtitle="", labels={}):
        fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
        ax.set_title(figtitle, fontsize=30, pad=30)
        ax.set_xlabel(labels.get("x"), fontsize=24, labelpad=30)
        ax.set_ylabel(labels.get("y"), fontsize=24, labelpad=30)
        ax.tick_params(axis="both", which="both", labelsize=20)
        c = ax.pcolor(func, cmap="Greys_r")
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.tick_params(labelsize=20)
        fig.savefig(os.path.join(self._log_dir, figname))


    def reset(self):
        """ Create a log folder if it does not exist. """
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)


    def verboseTxtLogging(self, verbose):
        self._verbose = verbose


    def setLogTxtFilename(self, filename, append=False):
        if filename is not None:
            self._txt_filename = filename
            if append:
                return
            f = open(os.path.join(self._log_dir, self._txt_filename), "w")
            f.close()

#