import copy
import itertools
import logging
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from .ppo import PPOAgent
from .config import get_default_config, get_logdir
from .qenv import QEnv
from .evaluation import test_lengths
from . import metrics
from . import util
from .stategen import (
    StateGenerator,
    sample_haar_product,
    sample_haar_generalized,
    sample_from_mmap
)


def envloop(agent: PPOAgent, env: QEnv, num_iters: int, steps: int,
        start_iter: int=1, triggers=tuple(), config=None):
    """
    Runs `num_iters` agent-environment interaction loops

    Args:
        agent: Any
            RL agent object
        env: Any
            Vectorized environment conforming to the gym API.
        num_iters: int
            Number of agent-environment interaction loops.
        steps: int
            Number of time-steps to rollout each of the interaction loops.
        start_iter: int, default=1
            Start iteration. When training continues from checkpoint use this
            argument to ensure proper logging and metric tracking
        triggers: Iterable
            Iterable of triggers, that are called every `trigger_every`
            iterations. Triggers are callables, that can alter the parameters
            of the RL environment and the RL agent.
        config: yacs.CfgNode, optional
            Instance of YACS CfgNode. If `None`, default config from .config
            is loaded. The config object specifies many hyperparameters, so
            unless this function is called in an interactive debugging session,
            DON'T NEGLECT this argument
    """
    if config is None:
        config = get_default_config()

    device = torch.device(config.model_device)
    num_envs = env.num_envs
    env.reset()

    # Allocate PyTorch tensors
    obs_shape = (steps, num_envs) + env.single_observation_space.shape
    obs = torch.zeros(obs_shape, dtype=env.obs_dtype, device=device)
    logprobs = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
    actions = torch.zeros((steps, num_envs), dtype=torch.int64, device=device)
    rewards = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
    done = torch.zeros((steps, num_envs), dtype=torch.bool, device=device)

    run_ret, run_len = np.nan, np.nan

    for i in tqdm(range(start_iter, start_iter + num_iters)):
        # Measure elapsed time
        tic = time.time()

        # Perform parallel step-by-step rollout along multiple trajectories.
        episode_returns, episode_lengths = [], []
        terminated = 0
        o = env.obs_fn(env.simulator.states, device=device)

        for s in range(steps):
            # Store the current observation
            obs[s] = o
            # Sample an action from the agent and step the environment /
            # uses torch.no_grad().
            p = agent.policy(o)
            acts = p.sample()
            # Store the log probabilities for the selected actions
            logprobs[s] = p.log_prob(acts)
            o, r, t, tr, infos = env.step(acts.cpu().numpy())

            actions[s] = acts
            rewards[s] = r
            done[s] = (t | tr)

            # If any of the environments is done, then save the statistics.
            if done[s].any():
                episode_returns.extend([
                    infos["episode"]["r"][k] for k in range(num_envs) if (t | tr)[k]
                ])
                episode_lengths.extend([
                    infos["episode"]["l"][k] for k in range(num_envs) if (t | tr)[k]
                ])
                terminated += sum((1 for i in range(num_envs) if t[i]))

        # Transpose `step` and `num_envs` dimensions and pass the experiences
        # to the agent to update the policy
        agent.update(
            obs.transpose(1, 0),
            actions.transpose(1, 0),
            rewards.transpose(1, 0),
            done.transpose(1, 0),
            logprobs.transpose(1, 0)
        )

        # Measure elapsed time
        elapsed = time.time() - tic

        # Bookkeeping
        assert len(episode_returns) == len(episode_lengths), "lengths must match"
        total_ep = len(episode_returns)
        ratio_terminated = terminated / total_ep if total_ep > 0 else np.nan
        for r, l in zip(episode_returns, episode_lengths):
            run_ret = r if run_ret is np.nan else 0.99 * run_ret + 0.01 * r
            run_len = l if run_len is np.nan else 0.99 * run_len + 0.01 * l
        with warnings.catch_warnings():
            # We might finish the rollout without completing any episodes. Then
            # taking the mean or std of an empty slice throws a runtime warning.
            # As a result we would get a NaN, which is exactly what we want.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_r, avg_l = np.mean(episode_returns), np.mean(episode_lengths)
            std_r, std_l = np.std(episode_returns), np.std(episode_lengths)

        tracker = metrics.getTracker()
        tracker.add_scalar("Return", avg_r, std_r)
        tracker.add_scalar("Return (run)", run_ret)
        tracker.add_scalar("Episode Length", avg_l, std_l)
        tracker.add_scalar("Episode Length (run)", run_len)
        tracker.add_scalar("Ratio Terminated", ratio_terminated)

        # Test the agent
        if i > 0 and i % config.test_every == 0:
            demo_single_subsystem_size(agent, env, config)

        # Log results
        if i % config.log_every == 0:
            logging.info(f"\nIteration ({i} / {start_iter + num_iters}) [{elapsed:.1f} sec]:")
            for name in tracker.get_names():
                j, val, std = tracker.get_last_scalar(name)
                if j == i:
                    logging.info(f"\t{name:<24}: {val:.6f} ± {std:.6f}")

        # Save checkpoint
        if i > 0 and i % config.checkpoint_every == 0:
            util.save_checkpoint(config, agent, triggers, iteration=i)
            # Save plots
            for name in tracker.get_names():
                logdir = get_logdir(config)
                figpath = os.path.join(logdir, f"{name}.png")
                tracker.plot_scalar(name, figpath, linewidth=2.5)

        # Run triggers
        if i > 0 and i % config.trigger_every == 0:
            logging.info("\n\n" + 75 * '=' + "\nRunning triggers...\n")
            for trigger in triggers:
                trigger(i)
            logging.info("\n" + 75 * '=' + "\n")

        # Increment tracker timestep
        tracker.step()


@torch.no_grad()
def demo_single_subsystem_size(agent: PPOAgent, env: QEnv, config):

    def log_results(lengths):
        ratio_terminated = np.sum(~np.isnan(lengths)) / lengths.size
        avg_length = np.nanmean(lengths)
        std_length = np.nanstd(lengths)
        min_length = np.nanmin(lengths)
        max_length = np.nanmax(lengths)
        p90_length = np.nanpercentile(lengths, 90)
        if np.isnan(avg_length):
            avg_length = np.inf
            std_length = np.inf
            min_length = np.inf
            max_length = np.inf
            p90_length = np.inf
        logging.info(f"\t\tRatio terminated: {ratio_terminated:.3f}")
        logging.info(f"\t\tAverage length:   {avg_length:.1f} ± {std_length:.1f}")
        logging.info(f"\t\tMinimum length:   {min_length:.1f}")
        logging.info(f"\t\tMaximum length:   {max_length:.1f}")
        logging.info(f"\t\t90-th % length:   {p90_length:.1f}")


    logging.info('\n\n' + 75 * '-')
    logging.info("Testing agent on various subsystem sizes...")

    for size in range(config.test_min_subsystem_size, config.test_max_subsystem_size + 1):
        logging.info(f"\n\tSubsystem size: {size}")
        params = dict(min_subsystem_size=size, max_subsystem_size=size)
        if config.stategen_fn == "sample_haar_product":
            sampler = StateGenerator(sample_haar_product, env.num_qubits, params)
        elif config.stategen_fn == "sample_haar_generalized":
            # Get `min_eta` and `max_eta` from the environment object,
            # because they may be changed during training with triggers
            params["min_eta"] = env.state_generator.sample_params["min_eta"]
            params["max_eta"] = env.state_generator.sample_params["max_eta"]
            sampler = StateGenerator(sample_haar_generalized, env.num_qubits, params)
        elif config.stategen_fn == "sample_from_mmap":
            # TODO: Fix somehow
            params["min_eta"] = 4.1
            params["max_eta"] = 4.1
            sampler = StateGenerator(sample_haar_generalized, env.num_qubits, params)
        else:
            raise ValueError("Unsupported `stategen_fn` in demo_single_subsystem_size()")
        lengths = test_lengths(agent, sampler,
                               max_steps=env.max_episode_steps,
                               obs_fn=config.obs_fn, num_tests=config.num_tests,
                               epsi=env.epsi, greedy=True)
        log_results(lengths)
    logging.info('\n' + 75 * '-' + '\n')


# @torch.no_grad()
# def demo_agent(agent: PPOAgent, env: QEnv, config):

#     def log_results(lengths):
#         ratio_terminated = np.sum(~np.isnan(lengths)) / lengths.size
#         avg_length = np.nanmean(lengths)
#         std_length = np.nanstd(lengths)
#         min_length = np.nanmin(lengths)
#         max_length = np.nanmax(lengths)
#         p90_length = np.nanpercentile(lengths, 90)
#         if np.isnan(avg_length):
#             avg_length = np.inf
#             std_length = np.inf
#             min_length = np.inf
#             max_length = np.inf
#             p90_length = np.inf
#         logging.info(f"\t\tRatio terminated: {ratio_terminated:.3f}")
#         logging.info(f"\t\tAverage length:   {avg_length:.1f} ± {std_length:.1f}")
#         logging.info(f"\t\tMinimum length:   {min_length:.1f}")
#         logging.info(f"\t\tMaximum length:   {max_length:.1f}")
#         logging.info(f"\t\t90-th % length:   {p90_length:.1f}\n")


#     logging.info('\n\n' + 75 * '-')
#     logging.info("Testing agent on various subsystem sizes...")

#     # Iterate trough list of states descriptions.
#     # Example: ["RR-RR-RR", "RRR-RRR", "RRRRRR"]
#     for sdescr in config.test_states:
#         states = itertools.cycle(util.str2state(sdescr) for _ in range(config.num_tests))
#         sample_fn = lambda x: next(states)
#         sampler = StateGenerator(sample_fn, env.num_qubits)

#         lengths = test_lengths(agent, sampler,
#                                max_steps=env.max_episode_steps,
#                                obs_fn=config.obs_fn, num_tests=config.num_tests,
#                                epsi=env.epsi, greedy=True)
#         logging.info(f"\n\tState description: {sdescr}")
#         log_results(lengths)


#     max_subsystem_size = env.state_generator.sample_params.get("max_subsystem_size")
#     if max_subsystem_size is not None:

#         logging.info("\nTesting agent on various subsystem sizes using "
#                      "environment's state sampler...")

#         for subsystem_size in range(2, max_subsystem_size+1):
#             # Clone state generator and change `min` and `max_subsystem_size`
#             sampler = copy.deepcopy(env.state_generator)
#             sampler.update(**dict(min_subsystem_size=subsystem_size,
#                                   max_subsystem_size=subsystem_size))

#             lengths = test_lengths(agent, sampler,
#                                 max_steps=env.max_episode_steps,
#                                 obs_fn=config.obs_fn, num_tests=config.num_tests,
#                                 epsi=env.epsi, greedy=True)
#             logging.info(f"\n\tSubsystem size: {subsystem_size}")
#             log_results(lengths)
#     else:
#         pass
#     logging.info('\n' + 75 * '-' + '\n')



# @torch.no_grad()
# def test_agent(agent, initial_states, **environment_kwargs):
#     """
#     Test the agent on batch of initial states.

#     Parameters:
#         agent (BaseAgent) :
#             The agent to test
#         initial_states (numpy.ndarray) :
#             Batch of initial states.
#         environment_kwargs (dict) :
#             Arguments passed to environment at initialization

#     Returns: dict
#         Dictionary with test results
#     """

#     # Initialize the environment
#     num_qubits = initial_states.ndim - 1
#     environment_kwargs["num_qubits"] = num_qubits
#     n_states = len(initial_states)
#     num_envs = environment_kwargs.get('num_envs', -1)
#     if num_envs < 0:
#         num_envs = 32
#     if num_envs > n_states:
#         warnings.warn("`num_envs` is greater than the number of initial states."
#                       " `num_envs` will be changed to `len(initial_states)`.")
#         num_envs = n_states
#     environment_kwargs.pop("num_qubits", None)
#     environment_kwargs.pop("num_envs", None)

#     results = defaultdict(list)
#     for n in range(0, n_states, num_envs):
#         batch = initial_states[n:n+num_envs]
#         env = QEnv(num_qubits=num_qubits, num_envs=len(batch),
#                          **environment_kwargs)
#         env.reset()
#         env.set_states(batch)
#         o = env.obs_fn(env.simulator.states)

#         lengths = [None] * env.num_envs
#         done = [False for _ in range(env.num_envs)]

#         for _ in range(env.max_episode_steps - 1):
#             o = torch.from_numpy(o)
#             pi = agent.policy(o)    # uses torch.no_grad
#             acts = torch.argmax(pi.probs, dim=1).cpu().numpy() # greedy selection

#             o, r, t, tr, infos = env.step(acts, reset=False)
#             if (t | tr).any():
#                 for k in range(env.num_envs):
#                     if (t[k] | tr[k]) and (lengths[k] is None):
#                         lengths[k] = infos["episode"]["l"][k]
#                         done[k] = True
#             if np.all(done):
#                 break
#         results["lengths"].extend(lengths)
#         results["done"].extend(done)
#         results["entanglements"].extend(env.simulator.entanglements)

#     results["lengths"] = np.array(results["lengths"])
#     results["done"] = np.array(results["done"])
#     results["entanglements"] = np.array(results["entanglements"])
#     return results