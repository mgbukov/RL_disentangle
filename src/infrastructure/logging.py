import logging
import os

import matplotlib.pyplot as plt
import numpy as np


#------------------------------ General logging functions -------------------------------#
def logText(msg, logfile):
    """Append a log message to the given logfile.

    Args:
        msg (str): String message to be logged.
        logfile (str, optional): File path to the file where logging information should be
            written. If empty the logging information is printed to the console.
            Default value is empty string.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]: # remove all old handlers
        logger.removeHandler(hdlr)
    if logfile != "":
        fileh = logging.FileHandler(logfile, "a")
        logger.addHandler(fileh)    # set the new handler
    logging.info(msg)

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
    cm = 1/2.54 # cm to inch
    fontsize = 10
    fig, ax = plt.subplots(figsize=(16*cm, 12*cm), dpi=330, tight_layout={"pad":1.4})
    ax.set_title(figtitle, fontsize=fontsize, pad=2)
    ax.set_xlabel(labels.get("x"), fontsize=fontsize, labelpad=2)#, font=fpath)
    ax.set_ylabel(labels.get("y"), fontsize=fontsize, labelpad=2)#, font=fpath)
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    ax.grid(which="major", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", linestyle="--", linewidth=0.3)

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
    ax.legend(loc="upper left", fontsize=fontsize)
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
def log_train_stats(stats, logfile):
    """Append training statistics to the log file.

    Args:
        stats (dict): A dictionary of numpy arrays containing:
            entropy (np.Array): A numpy array of shape (b, L), giving single-qubit entropies
                for the final states in every trajectory of the batch.
            rewards (np.Array): A numpy array of shape (b, t), giving the rewards obtained
                during trajectory rollout.
            exploration (np.Array): A numpy array of shape (b, t), giving the exploration
                rewards at every time-step.
            policy_entropy (float): The average entropy of the policy for this batch.
            loss (float): The value of the loss for this batch of episodes.
            total_norm (float): The value of the total gradient norm for this batch.
            nsolved (float): The number of solved trajectories for this batch.
            nsteps (np.Array): A numpy array of shape (b,), giving the length of every
                trajectory in the batch.
        logfile (str, optional): File path to the file where logging information should be
            written. If empty the logging information is printed to the console.
            Default value is empty string.
    """
    batch_size = len(stats["rewards"])
    logText(f"""\
    Mean return:              {np.mean(np.sum(stats["rewards"], axis=1)):.4f}
    Mean episode entropy:     {np.mean(stats["exploration"]):.4f}
    Mean final entropy:       {np.mean(stats["entropy"]):.4f}
    Median final entropy:     {np.median(stats["entropy"]):.4f}
    Max final entropy:        {np.max(stats["entropy"]):.4f}
    95 percentile entropy:    {np.percentile(stats["entropy"], 95.0):.5f}
    Value loss:               {stats["value_loss"]:.4f}
    Value Total grad norm     {stats["value_total_norm"]:.5f}
    Policy entropy:           {stats["policy_entropy"]:.4f}
    Policy pseudo loss:       {stats["policy_loss"]:.5f}
    Policy Total grad norm:   {stats["policy_total_norm"]:.5f}
    Solved trajectories:      {stats["nsolved"]} / {batch_size}
    Avg steps to disentangle: {np.mean(stats["nsteps"][stats["nsteps"].nonzero()]):.3f}
    Median steps to disent.: {np.median(stats["nsteps"][stats["nsteps"].nonzero()]):.1f}
    """, logfile)

def log_test_stats(stats, logfile=""):
    """Append test statistics to the log file.

    Args:
        stats (dict): A dictionary of numpy arrays containing:
            entropies (np.Array): A numpy array of shape (num_episodes, L), giving the
                final entropies for each trajectory during testing,
            returns (np.Array): A numpy array of shape (num_episodes,), giving the
                obtained return during each trajectory.
            nsolved (np.Array): A numpy array of shape (num_episodes,), of boolean values,
                indicating which trajectories are disentangled.
            nsteps (np.Array): A numpy array of shape (num_episodes,), giving the number
                of steps for each episode.
        logfile (str, optional): File path to the file where logging information should be
            written. If empty the logging information is printed to the console.
            Default value is empty string.
    """
    entropies = stats["entropies"]
    returns = stats["returns"]
    nsolved = stats["nsolved"]
    nsteps = stats["nsteps"]
    num_episodes = len(returns)
    solved = sum(nsolved)
    logText(f"""\
    Solved states:         {solved:.0f} / {num_episodes} = {solved/(num_episodes)*100:.3f}%
    Min entropy:           {entropies.min():.5f}
    Mean final entropy:    {np.mean(entropies):.4f}
    95 percentile entropy: {np.quantile(entropies.mean(axis=-1).reshape(-1), 0.95):.5f}
    Max entropy:           {entropies.max():.5f}
    Mean return:           {np.mean(returns):.4f}
    Avg steps to disentangle: {np.mean(nsteps[nsteps.nonzero()]):.3f}
    Median steps to disent.: {np.median(nsteps[nsteps.nonzero()]):.1f}
    """, logfile)

def plot_entropy_curves(train_history, filepath, lw=[0.4, 0.4, 0.6, 0.6]):
    keys = sorted(train_history.keys())

    # Define entropies curve.
    ent_min = np.array([np.min(train_history[i]["entropy"]) for i in keys])
    ent_max = np.array([np.max(train_history[i]["entropy"]) for i in keys])
    ent_mean = np.array([np.mean(train_history[i]["entropy"]) for i in keys])
    ent_quantile = np.array([np.quantile(train_history[i]["entropy"], 0.95) for i in keys])

    # Plot curves.
    logPlot(figname=filepath,
            xs=[keys, keys, keys, keys],
            funcs=[ent_min, ent_max, ent_mean, ent_quantile],
            legends=["min", "max", "mean", "95%quantile"],
            labels={"x":"Iteration", "y":"Entropy"},
            fmt=["--r", "--b", "-k", ":m"], lw=lw,
            figtitle="System entropy at episode end")

def plot_policy_loss(train_history, filepath, lw=0.4):
    num_iter = len(train_history)
    policy_loss = [train_history[i]["policy_loss"] for i in range(num_iter)]
    logPlot(figname=filepath, funcs=[policy_loss], legends=["loss"],
        labels={"x":"Iteration", "y":"Loss"}, fmt=["-b"], lw=[lw], figtitle="Policy Training Loss")

def plot_value_loss(train_history, filepath, lw=0.4):
    num_iter = len(train_history)
    value_loss = [train_history[i]["value_loss"] for i in range(num_iter)]
    logPlot(figname=filepath, funcs=[value_loss], legends=["loss"],
        labels={"x":"Iteration", "y":"Loss"}, fmt=["-b"], lw=[lw], figtitle="Value Training Loss")

def plot_policy_entropy(train_history, filepath, lw=0.4):
    num_iters = len(train_history)
    policy_entropy = [train_history[i]["policy_entropy"] for i in range(num_iters)]
    logPlot(figname=filepath, funcs=[policy_entropy], legends=["policy_entropy"], fmt=["-b"], lw=[lw],
        labels={"x":"Iteration", "y":"Policy Entropy"}, figtitle="Average policy entropy")

def plot_return_curves(train_history, test_history, filepath):
    num_iter = len(train_history)
    num_test = len(test_history)
    test_every = num_iter // (num_test - 1) if num_test > 1 else 0
    avg_every = max(1, test_every // 10)

    # Define return curves.
    returns = [np.sum(train_history[i]["rewards"], axis=1).mean() for i in range(num_iter)]
    avg_returns = np.convolve(returns, np.ones(avg_every), mode='valid') / avg_every
    avg_returns = avg_returns[::-1][::avg_every][::-1]
    avg_returns = np.concatenate((returns[:1], avg_returns))
    test_returns = [test_history[i]["returns"].mean() for i in sorted(test_history.keys())]

    # Plot curves.
    logPlot(figname=filepath,
            xs=[np.arange(num_iter),
                np.arange(0, num_iter, avg_every),
                sorted(test_history.keys())],
            funcs=[returns, avg_returns, test_returns],
            legends=["mean_batch_returns", "avg_returns", "test_returns"],
            labels={"x":"Iteration", "y":"Return"},
            fmt=["--r", "-k", "-b"],
            lw=[0.4, 1.2, 1.2],
            figtitle="Agent Obtained Return")

def plot_nsolved_curves(train_history, test_history, filepath):
    batch_size, steps = train_history[0]["rewards"].shape
    num_iter = len(train_history)
    num_test = len(test_history)
    test_every = num_iter // (num_test - 1) if num_test > 1 else 0
    avg_every = max(1, test_every // 10)

    # Define trajectories curves.
    nsolved = [train_history[i]["nsolved"] / batch_size for i in range(num_iter)]
    avg_nsolved = np.convolve(nsolved, np.ones(avg_every), mode='valid') / avg_every
    avg_nsolved = avg_nsolved[::-1][::avg_every][::-1]
    avg_nsolved = np.concatenate((nsolved[:1], avg_nsolved))
    test_nsolved = [test_history[i]["nsolved"].mean() for i in sorted(test_history.keys())]

    # Plot curves.
    logPlot(figname=filepath,
            xs=[np.arange(num_iter),
                np.arange(0, num_iter, avg_every),
                sorted(test_history.keys())],
            funcs=[nsolved, avg_nsolved, test_nsolved],
            legends=["batch_nsolved", "avg_nsolved", "test_nsolved"],
            labels={"x":"Episode", "y":"nsolved"},
            fmt=["--r", "-k", "-b"],
            lw=[0.4, 1.2, 1.2],
            figtitle="Agent accuracy of solved states")

def plot_distribution(train_history, log_every, log_dir):
    num_iter = len(train_history)
    for i in range(0, num_iter, log_every):
        logPcolor(figname=os.path.join(log_dir, f"policy_output_step_{i}.png"),
                func=train_history[i]["policy_output"].T,
                figtitle=f"Probabilities of actions given by the policy at step {i}",
                labels={"x":"Step", "y":"Actions"})

def plot_nsteps(train_history, filepath):
    num_iter = len(train_history)
    xs = np.arange(num_iter)
    ys = [np.mean(train_history[i]["nsteps"][train_history[i]["nsteps"].nonzero()])
        for i in range(num_iter)]
    logPlot(figname=filepath,
            xs=[xs],
            funcs=[ys],
            legends=["nsteps"],
            labels={"x": "Iteration", "y": "Steps"},
            fmt=["-r"],
            lw=[0.4],
            figtitle="Steps to Disentangle")

def plot_reward_function(env, filepath):
    N = 10
    L = env.L
    B = env.batch_size
    x = np.linspace(0.0, 0.7, N)[None, :]                       # (1, N)
    dummy_entropies = np.tile(x, reps=(L, 1))[None, :, :]       # (1, L, N)
    dummy_entropies = np.tile(dummy_entropies, reps=(B, 1, 1))  # (B, L, N)
    rewards = np.zeros((N, L), dtype=np.float32)
    for i in range(N):
        ent = dummy_entropies[:, :, i]
        r = env.Reward(env.states, entropies=ent)
        rewards[i] = r[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    c = ax.pcolormesh(rewards.T, cmap='PuRd', edgecolors='k')
    ax.set_title('Reward Function')
    ax.set_ylabel('qubit index')
    ax.set_xlabel('single qubit entropy')
    ax.set_xticks(ticks=list(range(N)), minor=False)
    ax.set_xticklabels(np.round(x.ravel(), 2), minor=False)
    # Text annotation
    for i, j in np.indices(rewards.shape).reshape(2, -1).T:
        val = rewards[i, j]
        ax.text(i + 0.5, j + 0.5, f'{val:.2f}',
                horizontalalignment='center', verticalalignment='center',
                color='k', fontsize=8.0)
    fig.colorbar(c, ax=ax)
    fig.savefig(filepath, dpi=160)
    plt.close(fig)

#