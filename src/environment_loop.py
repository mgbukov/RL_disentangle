import logging
import os
import time
import warnings
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import sys

# file_path = os.path.split(os.path.abspath(__file__))[0]
# project_dir = os.path.abspath(os.path.join(file_path, os.pardir))
# sys.path.append(project_dir)

from .config import get_default_config, get_logdir
from .quantum_env import QuantumEnv


def environment_loop_old(seed, agent, env, num_iters, steps, log_dir,
                     log_every=1, checkpoint_every=None, demo=None):
    """Runs a number of agent-environment interaction loops.

    Args:
        seed: int
            Seed for random number generation.
        agent: RL agent object
        env: gym.VectorEnv
            Vectorized environment conforming to the gym API.
        num_iters: int
            Number of agent-environment interaction loops.
        steps: int
            Number of time-steps to rollout each of the interaction loops.
        log_dir: str
            Path to a logging folder where useful information will be stored.
        log_every: int, optional
            Log the results every `log_every` iterations. Default: 1.
        checkpoint_every: int
            If not None, checkpoint the agent every `checkpoint_every` iterations
        demo: func(int, agent), optional
            Function that accepts an integer (the current iteration number) and
            an agent and produces a demo of the performance of the agent.
            Default: None.
    """
    logging.basicConfig(format="%(message)s", filemode="w", level=logging.INFO,
        filename=os.path.join(log_dir, "train.log"))
    tic = time.time()

    # Reset the environment and store the initial observations.
    # Note that during the interaction loops we will not be resetting the
    # environment. The vector environment will autoreset sub-environments after
    # they terminate or truncate.
    num_envs = env.num_envs
    o, _ = env.reset(seed=seed)

    run_ret, run_len = np.nan, np.nan
    for i in tqdm(range(num_iters)):
        # Allocate tensors for the rollout observations.
        obs = np.zeros(
            shape=(steps, num_envs, *env.single_observation_space.shape),
            dtype=env.obs_dtype
        )
        actions = np.zeros(shape=(steps, num_envs), dtype=int)
        rewards = np.zeros(shape=(steps, num_envs), dtype=np.float32)
        done = np.zeros(shape=(steps, num_envs), dtype=bool)

        # Perform parallel step-by-step rollout along multiple trajectories.
        episode_returns, episode_lengths = [], []
        terminated = 0
        for s in range(steps):
            # Sample an action from the agent and step the environment.
            obs[s] = o
            acts = agent.policy(torch.from_numpy(o)).sample() # uses torch.no_grad()
            acts = acts.cpu().numpy()
            o, r, t, tr, infos = env.step(acts)

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

        # Transpose `step` and `num_envs` dimensions and cast to torch tensors.
        obs = torch.from_numpy(obs).transpose(1, 0)
        actions = torch.from_numpy(actions).transpose(1, 0)
        rewards = torch.from_numpy(rewards).transpose(1, 0)
        done = torch.from_numpy(done).transpose(1, 0)

        # Pass the experiences to the agent to update the policy.
        agent.update(obs, actions, rewards, done)

        # Bookkeeping.
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
        # When indexing in `train_history` use -1 instead of `i`, because,
        # training may have continued from a checkpoint, and `i` starts from 0,
        # instead of last train iteration.
        agent.train_history[-1].update({
            "Return"           : {"avg" : avg_r, "std" : std_r, "run" : run_ret},
            "Episode Length"   : {"avg" : avg_l, "std" : std_l, "run" : run_len},
            "Ratio Terminated" : {"avg" : ratio_terminated},
        })

        # Demo.
        if demo is not None:
            demo(i, agent)

        # Log results.
        if i % log_every == 0:
            logging.info(f"\nIteration ({i+1} / {num_iters}):")
            for k, v in agent.train_history[-1].items():
                # If `v` is dictionary, round values to 6 digits of precision
                if isinstance(v, dict):
                    v_rounded = {}
                    for kk, vv in v.items():
                        try:
                            v_rounded[kk] = round(float(vv), 6)
                        except:
                            v_rounded[kk] = vv
                    v = v_rounded
                elif isinstance(v, float):
                    v = round(v, 6)
                elif isinstance(v, np.inexact):
                    v = round(float(v), 6)
                # Log
                logging.info(f"\t{k:<24}: {v}")

        # Checkpoint
        if checkpoint_every is not None and i % checkpoint_every == 0 and i > 0:
            agent.save(log_dir, increment=i)

    # Time the entire agent-environment loop.
    toc = time.time()
    logging.info(f"\nTraining took {toc-tic:.3f} seconds in total.")

    # Close the environment and save the agent.
    env.close()
    agent.save(log_dir)


def environment_loop(agent, env, tracker, triggers_list, num_iters, steps, config=None):

    if config is None:
        config = get_default_config()

    num_envs = env.num_envs
    env.reset()


    run_ret, run_len = np.nan, np.nan
    for i in tqdm(range(1, num_iters + 1)):
        # Allocate tensors for the rollout observations.
        obs = np.zeros(
            shape=(steps, num_envs, *env.single_observation_space.shape),
            dtype=env.obs_dtype
        )
        actions = np.zeros(shape=(steps, num_envs), dtype=int)
        rewards = np.zeros(shape=(steps, num_envs), dtype=np.float32)
        done = np.zeros(shape=(steps, num_envs), dtype=bool)

        # Perform parallel step-by-step rollout along multiple trajectories.
        episode_returns, episode_lengths = [], []
        terminated = 0
        o = env.obs_fn(env.simulator.states)
        for s in range(steps):
            # Sample an action from the agent and step the environment.
            obs[s] = o
            acts = agent.policy(torch.from_numpy(o)).sample() # uses torch.no_grad()
            acts = acts.cpu().numpy()
            o, r, t, tr, infos = env.step(acts)

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

        # Transpose `step` and `num_envs` dimensions and cast to torch tensors.
        obs = torch.from_numpy(obs).transpose(1, 0)
        actions = torch.from_numpy(actions).transpose(1, 0)
        rewards = torch.from_numpy(rewards).transpose(1, 0)
        done = torch.from_numpy(done).transpose(1, 0)

        # Pass the experiences to the agent to update the policy.
        agent.update(obs, actions, rewards, done)

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

        tracker.add_scalar("Return", avg_r, std_r)
        tracker.add_scalar("Return (run)", run_ret)
        tracker.add_scalar("Episode Length", avg_l, std_l)
        tracker.add_scalar("Episode Length (run)", run_len)
        tracker.add_scalar("Ratio Terminated", ratio_terminated)

        # Test the agent
        if i > 0 and i % config.test_every == 0:
            demo_agent(agent, config)

        # Log results
        if i % config.log_every == 0:
            logging.info(f"\nIteration ({i} / {num_iters}):")
            for name in tracker.get_names():
                j, val, std = tracker.get_last_scalar(name)
                logging.info(f"\t{name:<24}: {val:.6f} ± {std:.6f}")

        # Save checkpoint
        if i > 0 and i % config.checkpoint_every == 0:
            agent.save(get_logdir(config), increment=i)

        # Run triggers
        if i > 0 and i % config.trigger_every == 0:
            logging.info("\n\n" + 75 * '=' + "\nRunning triggers...\n")
            for trigger in triggers_list:
                trigger(i)
            logging.info("\n" + 75 * '=' + "\n")

        # Increment tracker timestep
        tracker.step()



def demo_agent(agent, config):
    pass


def test_agent(agent, initial_states, **environment_kwargs):
    """
    Test the agent on batch of initial states.

    Parameters:
        agent: BaseAgent
            The agent to test
        initial_states: numpy.ndarray
            Batch of initial states.
        environment_kwargs: dict
            Arguments passed to environment at initialization

    Returns: dict
        Dictionary with test results
    """

    # Initialize the environment.
    num_qubits = initial_states.ndim - 1
    environment_kwargs["num_qubits"] = num_qubits
    n_states = len(initial_states)
    num_envs = environment_kwargs.get('num_envs', -1)
    if num_envs < 0:
        num_envs = 128
    if num_envs > n_states:
        warnings.warn("`num_envs` is greater than the number of initial states."
                      " `num_envs` will be changed to `len(initial_states)`.")
        num_envs = n_states
    environment_kwargs.pop("num_qubits", None)
    environment_kwargs.pop("num_envs", None)

    results = defaultdict(list)
    for n in range(0, n_states, num_envs):
        batch = initial_states[n:n+num_envs]
        env = QuantumEnv(num_qubits=num_qubits, num_envs=len(batch),
                         **environment_kwargs)
        env.reset()
        env.simulator.states = batch
        o = env.obs_fn(env.simulator.states)

        lengths = [None] * env.num_envs
        done = [False for _ in range(env.num_envs)]

        for _ in range(env.max_episode_steps - 1):
            o = torch.from_numpy(o)
            pi = agent.policy(o)    # uses torch.no_grad
            acts = torch.argmax(pi.probs, dim=1).cpu().numpy() # greedy selection

            o, r, t, tr, infos = env.step(acts, reset=False)
            if (t | tr).any():
                for k in range(env.num_envs):
                    if (t[k] | tr[k]) and (lengths[k] is None):
                        lengths[k] = infos["episode"]["l"][k]
                        done[k] = True
            if np.all(done):
                break
        results["lengths"].extend(lengths)
        results["done"].extend(done)
        results["entanglements"].extend(env.simulator.entanglements)

    results["lengths"] = np.array(results["lengths"])
    results["done"] = np.array(results["done"])
    results["entanglements"] = np.array(results["entanglements"])
    return results