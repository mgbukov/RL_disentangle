import argparse
import itertools
import json
import pickle
import os
import collections.abc as abc

import numpy as np
import torch

from src.config import get_default_config
from src.stategen import sample_mps, sample_haar_generalized
from src.util import str2state, load_checkpoint
from src.ppo import PPOAgent
from src.qenv import QEnv
from src.networks import TransformerPE_2qRDM, MLP



# Define the initial states on which we want to test the agent.
TEST_HAAR_PRODUCT_STATES = {
    4: ["RR-R-R", "RR-RR", "RRR-R", "RRRR"],
    5: ["RR-R-R-R", "RR-RR-R", "RRR-R-R", "RRR-RR", "RRRR-R", "RRRRR"],
    6: ["RR-R-R-R-R", "RR-RR-R-R", "RR-RR-RR", "RRR-R-R-R", "RRR-RR-R",
        "RRRR-R-R", "RRRR-RR", "RRRRR-R", "RRRRRR"],
    7: ["RR-RR-RR-R", "RRR-RRR-R", "RRRR-RRR", "RRRRR-RR", "RRRRRR-R"],
    8: ["RRR-RRR-RR", "RRRR-RRRR", "RRRRR-RRR", "RRRRRR-RR", "RRRRRR-R", "RRRRRRRR"],
    10: ["RRR-RRR-RRR-R", "RRR-RRRR-RRR", "RRRR-RRRR-RR", "RRRRR-RRRRR",
         "RRRRRR-RRRR", "RRRRRRRR-RR", "RRRRRRRRR-R", "RRRRRRRRRR"],
    12: ["RR-RR-RR-RR-RR-RR", "RRR-RRR-RRR-RRR", "RRRR-RRRR-RRRR", "RRRR-RRRRR-R",
         "RRRRRR-RRRRRR", "RRRR-RRRRRRRR", "RR-RRRRRRRRRR"],
    15: ["RR-RR-RR-RR-RR-RR-RR-R", "RRR-RRR-RRR-RRR-RRR", "RRRR-RRRR-RRRR-RRR",
         "RRRRR-RRRRR-RRRRR"],
    16: ["RR-RR-RR-RR-RR-RR-RR-RR", "RRR-RRR-RRR-RRR-RRR-R", "RRRR-RRRR-RRRR-RRRR"]
}


def test_agent(agent, states, greedy=False, **env_kwargs):
    # Initialize the environment
    states = np.asarray(states)
    num_envs = env_kwargs.pop("num_envs", states.shape[0])
    num_qubits = env_kwargs.pop("num_qubits", states.ndim - 1)
    shape = (num_envs,) + (2,) * num_qubits
    env = QEnv(num_qubits=num_qubits, num_envs=num_envs, **env_kwargs)
    env.reset()
    env.simulator.states = states.reshape(shape)

    o = env.obs_fn(env.simulator.states)
    lengths = np.full(num_envs, np.nan, dtype=np.float32)
    done = np.full(num_envs, False, dtype=bool)
    solved = 0
    for _ in range(env.max_episode_steps):
        pi = agent.policy(o)    # uses torch.no_grad
        if greedy:
            acts = torch.argmax(pi.probs, dim=1).cpu().numpy()
        else:
            acts = pi.sample().cpu().numpy()
        o, r, t, tr, infos = env.step(acts)
        if t.any():
            for k in range(env.num_envs):
                if t[k] and not done[k]:
                    lengths[k] = infos["episode"]["l"][k]
                    done[k] = True
                    solved += 1
        if np.all(done):
            break

    return {
        "avg_len": float(np.nanmean(lengths)),
        "std": float(np.nanstd(lengths)),
        "95_percentile": float(np.nanpercentile(lengths, 95.)),
        "max_len": float(np.nanmax(lengths)),
        "ratio_solved": float(solved / num_envs),
    }


def test_on_haar_random(agent, num_qubits, n_tests=128, greedy=True, **env_kwargs):

    results = {}
    if not isinstance(num_qubits, abc.Iterable):
        num_qubits = (num_qubits,)

    for L in num_qubits:
        try:
            names = TEST_HAAR_PRODUCT_STATES[L]
        except:
            continue
        results[L] = {}
        for name in names:
            states = np.array([str2state(name) for _ in range(n_tests)])
            results[L][name] = test_agent(agent, states, greedy, **env_kwargs)

    return results


def test_on_mps(agent, num_qubits, chi_max, n_tests=128, greedy=True, **env_kwargs):

    if not isinstance(num_qubits, abc.Iterable):
        num_qubits = (num_qubits,)

    if not isinstance(chi_max, abc.Iterable):
        chi_max = (chi_max,)

    results = {}

    for L in num_qubits:
        results[L] = {}
        for chi in chi_max:
            states = np.array([sample_mps(L, chi) for _ in range(n_tests)])
            results[L][f"chimax={chi}"] = test_agent(agent, states, greedy, **env_kwargs)

    return results


def test_on_weakly_entangled(agent, num_qubits, subsystem_size, eta,
                             n_tests=128, greedy=True, **env_kwargs):

    if not isinstance(num_qubits, abc.Iterable):
        num_qubits = (num_qubits,)

    if not isinstance(subsystem_size, abc.Iterable):
        subsystem_size = (subsystem_size,)

    if not isinstance(eta, abc.Iterable):
        eta = (eta,)

    results = {}

    for L in num_qubits:
        results[L] = {}
        # Test with variable sybsystem size
        minS = min(subsystem_size)
        maxS = max(subsystem_size)
        for E in eta:
            states = np.array([sample_haar_generalized(L, minS, maxS, E, E) for _ in range(n_tests)])
            results[L][f"eta={E}"] = test_agent(agent, states, greedy, **env_kwargs)
        # Test with fixed subsystem size
        for S in subsystem_size:
            for E in eta:
                states = np.array([sample_haar_generalized(L, minS, maxS, E, E) for _ in range(n_tests)])
                results[L][f"subsystem_size={S},eta={E}"] = test_agent(agent, states, greedy, **env_kwargs)


    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str)
    # TODO
    # parser.add_argument("--greedy", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_qubits", nargs='+', type=int)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--epsi", type=float, default=1e-3)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--obs_fn", type=str, default="rdm2m")
    parser.add_argument("--n_tests", type=int, default=128)
    parser.add_argument("--chi_max", nargs='+', type=int, default=[4])
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--haar_random", action="store_true")
    parser.add_argument("--weakly_entangled", action="store_true")
    parser.add_argument("--eta", nargs="+", type=float)
    parser.add_argument("--subsystem_size", nargs="+", type=int)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    device = "cpu" if args.cuda is False else "cuda"

    # Load agent directly or reinitialize from checkpoint
    if args.agent:
        print(f"Loading agent from \"{args.agent}\"...")
        agent = torch.load(args.agent, map_location=device)
        print(f"Agent loaded successfuly!")
        agent.policy_network.eval()
        agent.value_network.eval()
    elif args.checkpoint and args.config:
        # Load config
        config = get_default_config()
        config.merge_from_file(args.config)
        config.freeze()
        # Initialize RL environment
        env = QEnv(config.num_qubits, config.num_envs, device=device)
        # Initialize policy network
        in_shape = env.single_observation_space.shape
        policy_network = TransformerPE_2qRDM(
            in_shape[1],
            embed_dim=      config.embed_dim,
            dim_mlp=        config.dim_mlp,
            n_heads=        config.attn_heads,
            n_layers=       config.transformer_layers
        ).to(device)
        print("Initialized policy network")
        # Initialize value network
        value_network = MLP(in_shape, [128, 256], 1).to(device)
        # Initialize PPOAgent
        agent = PPOAgent(policy_network, value_network, config={
            "pi_lr":        config.pi_lr,
            "vf_lr":        config.vf_lr,
            "discount":     config.discount,
            "batch_size":   config.batch_size,
            "clip_grad":    config.clip_grad,
            "entropy_reg":  config.entropy_reg,

            # PPO-specific
            "pi_clip":              0.2,
            "vf_clip":              10.0,
            "tgt_KL":               0.01,
            "num_ppo_updates":      96,
            "lamb":                 0.95
        })
        print("Initialized RL agent")
        # Load parameters from checkpoint
        checkpointed_state = load_checkpoint(args.checkpoint)
        agent.policy_network.load_state_dict(checkpointed_state["policy_fn"])
        print("Loaded parameters from checkpoint")
    else:
        exit(1)


    if args.cuda:
        agent.policy_network.cuda()
        agent.value_network.cuda()

    results = {}
    env_kwargs = dict(
        max_episode_steps=      args.max_steps,
        epsi=                   args.epsi,
        obs_fn=                 args.obs_fn,
    )
    results['agent'] = args.agent
    results["env_kwargs"] = env_kwargs
    # results["chi_max"] = args.chi_max
    results["n_tests"] = args.n_tests

    if args.haar_random:
        print("Testing on Haar product states...")
        results["haar_random"] = test_on_haar_random(
            agent,
            args.num_qubits,
            args.n_tests,
            not args.sample,
            **env_kwargs
        )
        with open(args.output, mode='wt') as f:
            json.dump(results, f, indent=2)

    if args.mps:
        print("Testing on Matrix product states...")
        results["mps"] = test_on_mps(
            agent,
            max(args.num_qubits),
            args.chi_max,
            args.n_tests,
            not args.sample,
            **env_kwargs
        )
        with open(args.output, mode='wt') as f:
            json.dump(results, f, indent=2)

    if args.weakly_entangled:
        print("Testing on weakly entangled states...")
        results["weakly_entangled"] = test_on_weakly_entangled(
            agent,
            args.num_qubits,
            args.subsystem_size,
            args.eta,
            args.n_tests,
            not args.sample,
            **env_kwargs
        )
        with open(args.output, mode='wt') as f:
            json.dump(results, f, indent=2)

    print("Testing completed! Writing results...")
    with open(args.output, mode='wt') as f:
        json.dump(results, f, indent=2)
    print("Done!")
