import os
import sys

import matplotlib.pyplot as plt
import numpy as np


#------------------------------ General logging functions -------------------------------#
def logPlot(figname, xs=None, funcs=[], legends=[None], labels={}, fmt=["--k"], lw=[0.8],
            fills=[], figtitle="", logscaleX=False, logscaleY=False):
    """Plot @funcs as curves on a figure and save the figure as `figname`.

    Args:
        figname (str): Full path to save the figure to a file.
        xs (list[np.Array], optional): List of arrays of x-axis data points.
            Default value is None.
        funcs (list[np.Array], optional): List of arrays of data points. Every array of
            data points from the list is plotted as a curve on the figure.
            Default value is [].
        legends (list[str], optional): A list of labels for every curve that will be
            displayed in the legend. Default value is [None].
        labels (dict, optional): A map specifying the labels of the coordinate axes.
            `labels["x"]` specifies the label of the x-axis.
            `labels["y"]` specifies the label of the y-axis.
            Default value is {}.
        fmt (list[str], optional): A list of formating strings for every curve.
            Default value is ["--k"].
        lw (list[float], optional): A list of line widths for every curve.
            Default value is [0.8].
        fills (list[tuple(np.Array)], optional): A list of tuples of curves [(f11,f21), (f12,f22), ...].
            Fill the area between the curves f1 and f2. Default value is [].
        figtitle (str, optional): Figure title. Default value is "".
        logscaleX (bool, optional): If True, plot the x-axis on a logarithmic scale.
            Default value is False.
        logscaleY (bool, optional): If True, plot the y-axis on a logarithmic scale.
            Default value is False.
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
    fig.savefig(figname)
    plt.close(fig)

def logHistogram(figname, func, bins, figtitle="", labels={}, align="mid", rwidth=1.0):
    """Plot `func` as a histogram on a figure and save the figure as `figname`.

    Args:
        figname (str): Full path to save the figure to a file.
        func (np.Array): Array of data points.
        bins (int or np.Array): Number of equal-width bins or bin edges.
        figtitle (str, optional): Figure title. Default value is "".
        labels (dict, optional): A map specifying the labels of the coordinate axes.
            `labels["x"]` specifies the label of the x-axis.
            `labels["y"]` specifies the label of the y-axis.
            Default value is {}.
        align (["left", "mid", "right"], optional): Horizontal alignment of the histogram
            bars. Default value is "mid".
        rwidth (float, optional): Relative width of the bars. Default value is 1.0.
    """
    fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
    ax.set_title(figtitle, fontsize=30, pad=30)
    ax.set_xlabel(labels.get("x"), fontsize=24, labelpad=30)
    ax.set_ylabel(labels.get("y"), fontsize=24, labelpad=30)
    ax.tick_params(axis="both", which="both", labelsize=20)
    ax.grid()
    ax.hist(func, bins, align=align, rwidth=rwidth)
    fig.savefig(figname)
    plt.close(fig)

def logBarchart(figname, x, func, figtitle ="", labels={}, xlim=None, ylim=None):
    """Plot @func as a bar chart on a figure and save the figure as @figname.

    Args:
        figname (str): Full path to save the figure to a file.
        x (np.Array): Array of x-axis data points.
        func (np.Array): Array of data points.
        figtitle (str, optional): Figure title. Default value is "".
        labels (dict, optional): A map specifying the labels of the coordinate axes.
            `labels["x"]` specifies the label of the x-axis.
            `labels["y"]` specifies the label of the y-axis.
            Default value is {}.
        xlim (tuple, optional): X-axis view limits. Default value is None.
        ylim (tuple, optional): Y-axis view limits. Default value is None.
    """
    fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
    ax.set_title(figtitle, fontsize=30, pad=30)
    ax.set_xlabel(labels.get("x"), fontsize=24, labelpad=30)
    ax.set_ylabel(labels.get("y"), fontsize=24, labelpad=30)
    ax.tick_params(axis="both", which="both", labelsize=20)
    ax.grid()
    ax.bar(x, func)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig(figname)
    plt.close(fig)

def logPcolor(figname, func, figtitle="", labels={}):
    fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
    ax.set_title(figtitle, fontsize=30, pad=30)
    ax.set_xlabel(labels.get("x"), fontsize=24, labelpad=30)
    ax.set_ylabel(labels.get("y"), fontsize=24, labelpad=30)
    ax.tick_params(axis="both", which="both", labelsize=20)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
    c = ax.pcolor(func, cmap="PuRd", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    fig.savefig(figname)
    plt.close(fig)

#------------------------------ Specific logging functions ------------------------------#
def log_train_stats(stats, stdout=sys.stdout):
    """Append training statistics to the log file.

    Args:
        stdout (file, optional): File object (stream) used for standard output of logging
            information. Default value is `sys.stdout`.
    """
    batch_size = len(stats["rewards"])
    probs = stats["policy_output"]
    print(f"""    Mean final reward:        {np.mean(np.array(stats["rewards"])[:,-1]):.4f}
    Mean return:              {np.mean(np.sum(stats["rewards"], axis=1)):.4f}
    Mean exploration return:  {np.mean(np.sum(stats["exploration"], axis=1)):.4f}
    Mean final entropy:       {np.mean(stats["entropy"]):.4f}
    Max final entropy:        {np.max(stats["entropy"]):.4f}
    95 percentile entropy:    {np.percentile(stats["entropy"], 95.0):.5f}
    Policy entropy:           {-np.mean(np.sum(probs*np.log(probs),axis=-1)):.4f}
    Pseudo loss:              {stats["loss"]:.5f}
    Total gradient norm:      {stats["total_norm"]:.5f}
    Solved trajectories:      {stats["nsolved"]} / {batch_size}
    Avg steps to disentangle: {np.mean(stats["nsteps"][stats["nsteps"].nonzero()]):.3f}
    """, file=stdout)

def log_test_stats(stats, stdout):
    entropies, returns, nsolved = stats
    num_trajects = len(returns)
    solved = sum(nsolved)
    print(f"""    Solved states:         {solved:.0f} / {num_trajects} = {solved/(num_trajects)*100:.3f}%
    Min entropy:           {entropies.min():.5f}
    Mean final entropy:    {np.mean(entropies):.4f}
    95 percentile entropy: {np.quantile(entropies.mean(axis=-1).reshape(-1), 0.95):.5f}
    Max entropy:           {entropies.max():.5f}
    Mean return:           {np.mean(returns):.4f}
    """, file=stdout)

def plot_entropy_curves(train_history, file_path):
    num_iter = len(train_history)
    _, steps = train_history[0]["rewards"].shape

    # Define entropies curve.
    ent_min = np.array([np.min(train_history[i]["entropy"]) for i in range(num_iter)])
    ent_max = np.array([np.max(train_history[i]["entropy"]) for i in range(num_iter)])
    ent_mean = np.array([np.mean(train_history[i]["entropy"]) for i in range(num_iter)])
    ent_std = np.array([np.std(train_history[i]["entropy"]) for i in range(num_iter)])
    ent_mean_minus_std = ent_mean - 0.5 * ent_std
    ent_mean_plus_std = ent_mean + 0.5 * ent_std
    ent_quantile = np.array([np.quantile(train_history[i]["entropy"], 0.95) for i in range(num_iter)])

    # Plot curves.
    logPlot(figname=file_path,
            funcs=[ent_min, ent_max, ent_mean, ent_quantile],
            legends=["min", "max", "mean", "95%quantile"],
            labels={"x":"Iteration", "y":"Entropy"},
            fmt=["--r", "--b", "-k", ":m"],
            fills=[(ent_mean_minus_std, ent_mean_plus_std)],
            figtitle="System entropy after {} steps".format(steps))

def plot_loss_curve(train_history, file_path):
    num_iter = len(train_history)
    loss = [train_history[i]["loss"] for i in range(num_iter)]
    logPlot(figname=file_path, funcs=[loss], legends=["loss"],
            labels={"x":"Iteration", "y":"Loss"}, fmt=["-b"], figtitle="Training Loss")

def plot_return_curves(train_history, test_history, file_path):
    num_iter = len(train_history)
    num_test = len(test_history)
    test_every = num_iter // (num_test - 1)
    avg_every = test_every // 10

    # Define return curves.
    returns = [np.sum(train_history[i]["rewards"], axis=1).mean() for i in range(num_iter)]
    avg_returns = np.insert(np.mean(np.array(returns[1:]).reshape(-1, avg_every), axis=1), 0, returns[0])
    test_returns = [test_history[i]["returns"].mean() for i in range(0, num_iter, test_every)]

    # Plot curves.
    logPlot(figname=file_path,
            xs=[np.arange(num_iter),
                np.arange(0, num_iter, avg_every),
                np.arange(0, num_iter, test_every)],
            funcs=[returns, avg_returns, test_returns],
            legends=["mean_batch_returns", "avg_returns", "test_returns"],
            labels={"x":"Iteration", "y":"Return"},
            fmt=["--r", "-k", "-b"],
            lw=[1.0, 4.0, 4.0],
            figtitle="Agent Obtained Return")

    # Plot training return curve on a log-scale.
    # logPlot(figname="returns_logX.png", funcs=[returns], legends=["mean_batch_returns"],
    #         fmt=["--r"], figtitle="Return", logscaleX=True)
    # MUST HAVE POSITIVE VALUES ALONG Y-AXIS!!!
    # self.logger.logPlot(funcs=[returns], legends=["batch_returns"], fmt=["--r"],
    #                     figtitle="Training Loss", figname="returns_logY.png", logscaleY=True)

def plot_nsolved_curves(train_history, test_history, file_path):
    batch_size, steps = train_history[0]["rewards"].shape
    num_iter = len(train_history)
    num_test = len(test_history)
    test_every = num_iter // (num_test - 1)
    avg_every = test_every // 10

    # Define trajectories curves.
    nsolved = [train_history[i]["nsolved"] / batch_size for i in range(num_iter)]
    avg_nsolved = np.insert(np.mean(np.array(nsolved[1:]).reshape(-1, avg_every), axis=1), 0, nsolved[0])
    test_nsolved = [test_history[i]["nsolved"].mean() for i in range(0, num_iter, test_every)]

    # Plot curves.
    logPlot(figname=file_path,
            xs=[np.arange(num_iter),
                np.arange(0, num_iter, avg_every),
                np.arange(0, num_iter, test_every)],
            funcs=[nsolved, avg_nsolved, test_nsolved],
            legends=["batch_nsolved", "avg_nsolved", "test_nsolved"],
            labels={"x":"Episode", "y":"nsolved"},
            fmt=["--r", "-k", "-b"],
            lw=[1.0, 4.0, 4.0],
            figtitle="Agent accuracy of solved states")

def plot_distribution(train_history, log_every, log_dir):
    num_iter = len(train_history)
    for i in range(0, num_iter, log_every):
        logPcolor(figname=os.path.join(log_dir, f"policy_output_step_{i}.png"),
                func=train_history[i]["policy_output"].T, 
                figtitle=f"Probabilities of actions given by the policy at step {i}",
                labels={"x":"Step", "y":"Actions"})

#