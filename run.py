import json
import os
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.environment_loop import environment_loop
from src.networks import MLP, TransformerPE_2qRDM
from src.ppo import PPOAgent
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state


def demo(args):

    def thunk(i, agent):
        if i % args.demo_every != 0:
            return

        num_envs = 1024
        env = QuantumEnv(num_qubits=args.num_qubits, num_envs=num_envs,
            epsi=args.epsi, p_gen=args.p_gen, max_episode_steps=args.steps_limit,
            reward_fn=args.reward_fn, obs_fn=args.obs_fn,
        )
        _ = env.reset() # prepare the environment

        # Generate the initial states as fully-entangled states.
        env.simulator.set_random_states_()
        initial_states = env.simulator.states.copy()

        # Choose actions greedily or by sampling from the distribution.
        for choose in ["greedy"]: #, "sample"]:
            env.simulator.states = initial_states
            o = env.obs_fn(env.simulator.states)
            returns, lengths = [None] * env.num_envs, [None] * env.num_envs
            done = np.array([False] * env.num_envs)
            solved = 0
            while not done.all():
                o = torch.from_numpy(o)
                pi = agent.policy(o)    # uses torch.no_grad

                if choose == "greedy":
                    acts = torch.argmax(pi.probs, dim=1).cpu().numpy()
                if choose == "sample":
                    acts = pi.sample().cpu().numpy()

                o, r, t, tr, infos = env.step(acts)
                if (t | tr).any():
                    for k in range(env.num_envs):
                        if (t[k] | tr[k]) and (returns[k] is None):
                            returns[k] = infos["episode"]["r"][k]
                            lengths[k] = infos["episode"]["l"][k]
                            done[k] = True
                            if t[k]: solved += 1

            agent.train_history[i]["Return"].update({
                "test_avg" : np.mean(returns),
                "test_std" : np.std(returns),
            })
            agent.train_history[i]["Episode Length"].update({
                "test_avg" : np.mean(lengths),
                "test_std" : np.std(lengths),
            })
            agent.train_history[i]["Ratio Terminated"].update({
                "test_avg" : solved / num_envs,
            })

    return thunk


def test(agent):
    """Test the agent on a set of specifically generated quantum states.
    Note that this function will reset the numpy rng seed.
    """
    # Define a function wrapper for executing a function with a set rng seed.
    def fixed_rng(fn): np.random.seed(0); return fn

    # Define the initial states on which we want to test the agent. For each
    # of the special configurations provide a generating function.
    initial_states = {
        "|RR-000>" : fixed_rng(lambda: np.kron(
            random_quantum_state(q=2, prob=1.),
            np.kron(
                np.kron(
                    random_quantum_state(q=1, prob=1.),
                    random_quantum_state(q=1, prob=1.),
                ),
                random_quantum_state(q=1, prob=1.),
            ),
        ).reshape((2,) * 5).astype(np.complex64)),

        "|RR-RR-0>": fixed_rng(lambda: np.kron(
            random_quantum_state(q=2, prob=1.),
            np.kron(
                random_quantum_state(q=2, prob=1.),
                random_quantum_state(q=1, prob=1.),
            ),
        ).reshape((2,) * 5).astype(np.complex64)),

        "|RRR-00>" : fixed_rng(lambda: np.kron(
            random_quantum_state(q=3, prob=1.),
            np.kron(
                random_quantum_state(q=1, prob=1.),
                random_quantum_state(q=1, prob=1.),
            ),
        ).reshape((2,) * 5).astype(np.complex64)),

        "|RRR-RR>" : fixed_rng(lambda: np.kron(
            random_quantum_state(q=3, prob=1.),
            random_quantum_state(q=2, prob=1.),
        ).reshape((2,) * 5).astype(np.complex64)),

        "|RRRR-0>" : fixed_rng(lambda: np.kron(
            random_quantum_state(q=4, prob=1.),
            random_quantum_state(q=1, prob=1.),
        ).reshape((2,) * 5).astype(np.complex64)),

        "|RRRRR>"  : fixed_rng(lambda: random_quantum_state(q=5, prob=1.)),
    }

    # Define the environment.
    num_envs = 1024
    env = QuantumEnv(num_qubits=5, num_envs=num_envs,
        epsi=1e-3, max_episode_steps=40, obs_fn="rdm_2q_real",
    )

    # Try to solve each of the special configurations.
    results = {}
    for name, fn in initial_states.items():
        states = np.array([fn() for _ in range(num_envs)])
        _ = env.reset() # prepare the environment.
        env.simulator.states = states
        o = env.obs_fn(env.simulator.states)

        lengths = [None] * env.num_envs
        done = np.array([False] * env.num_envs)
        solved = 0
        while not done.all():
            o = torch.from_numpy(o)
            pi = agent.policy(o)    # uses torch.no_grad
            acts = torch.argmax(pi.probs, dim=1).cpu().numpy() # greedy selection

            o, r, t, tr, infos = env.step(acts)
            if (t | tr).any():
                for k in range(env.num_envs):
                    if (t[k] | tr[k]) and (lengths[k] is None):
                        lengths[k] = infos["episode"]["l"][k]
                        done[k] = True
                        if t[k]: solved += 1

        # Bookkeeping.
        results[name] = {
            "avg_len": np.mean(lengths),
            "95_percentile": np.percentile(lengths, 95.),
            "max_len": np.max(lengths),
            "ratio_solved": solved / num_envs,
        }

    return results


def pg_solves_quantum(args):
    # Use cuda.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seeds.
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create a vectorized quantum environment for simulating quantum states.
    # Pass the reward and observation functions to be used with the environment.
    env = QuantumEnv(
        num_qubits=args.num_qubits,
        num_envs=args.num_envs,
        epsi=args.epsi,
        p_gen=args.p_gen,
        max_episode_steps=args.steps_limit,
        reward_fn=args.reward_fn,
        obs_fn=args.obs_fn,
    )

    # Create the RL agent.
    in_shape = env.single_observation_space.shape
    in_dim = in_shape[1]
    out_dim = env.single_action_space.n
    policy_network = TransformerPE_2qRDM(
        in_dim,
        embed_dim=args.embed_dim,
        dim_mlp=args.dim_mlp,
        n_heads=args.attn_heads,
        n_layers=args.transformer_layers,
    ).to(device)
    value_network = MLP(in_shape, [256, 256], 1).to(device)
    # agent = VPGAgent(policy_network, value_network, config={
    agent = PPOAgent(policy_network, value_network, config={
        "pi_lr"     : args.pi_lr,
        "vf_lr"     : args.vf_lr,
        "discount"  : args.discount,
        "batch_size": args.batch_size,
        "clip_grad" : args.clip_grad,
        "entropy_reg": args.entropy_reg,

        # PPO-specific
        "pi_clip" : 0.2,
        "vf_clip" : 10.,
        "tgt_KL"  : 0.01,
        "n_epochs": 3,
        "lamb"    : 0.95,
    })

    # Run the environment loop
    log_dir = os.path.join("logs",
        f"pGen_{args.p_gen}_attnHeads_{args.attn_heads}_tLayers_{args.transformer_layers}"+
        f"_ppoBatch_{args.batch_size}_entReg_{args.entropy_reg}_embed_{args.embed_dim}_mlp_{args.dim_mlp}")
    os.makedirs(log_dir, exist_ok=True)
    environment_loop(seed, agent, env, args.num_iters, args.steps, log_dir, args.log_every, demo=demo(args))

    # Test the final agent and store the results.
    results = test(agent)
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots.
    plt.style.use("ggplot")
    for k in agent.train_history[0].keys():
        fig, ax = plt.subplots()
        if "avg" in agent.train_history[0][k].keys():
            avg = np.array([agent.train_history[i][k]["avg"] for i in range(args.num_iters)])
            ax.plot(avg, label="Average")
        if "std" in agent.train_history[0][k].keys():
            std = np.array([agent.train_history[i][k]["std"] for i in range(args.num_iters)])
            ax.fill_between(np.arange(args.num_iters), avg-0.5*std, avg+0.5*std, color="k", alpha=0.25)
        if "run" in agent.train_history[0][k].keys():
            run = np.array([agent.train_history[i][k]["run"] for i in range(args.num_iters)])
            ax.plot(run, label="Running")
        if "test_avg" in agent.train_history[0][k].keys():
            test_avg = np.array([agent.train_history[i][k]["test_avg"]
                for i in range(args.num_iters) if "test_avg" in agent.train_history[i][k].keys()])
            xs = np.linspace(0, args.num_iters, len(test_avg))
            ax.plot(xs, test_avg, label="Test Average")
        if "test_std" in agent.train_history[0][k].keys():
            test_std = np.array([agent.train_history[i][k]["test_std"]
                for i in range(args.num_iters) if "test_std" in agent.train_history[i][k].keys()])
            ax.fill_between(xs, test_avg-0.5*test_std, test_avg+0.5*test_std, color="k", alpha=0.25)
        ax.legend(loc="upper left")
        ax.set_xlabel("Number of training iterations")
        ax.set_ylabel(k)
        fig.savefig(os.path.join(log_dir, k.replace(" ", "_")+".png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--pi_lr", default=1e-4, type=float)
    parser.add_argument("--vf_lr", default=3e-4, type=float)
    parser.add_argument("--discount", default=1., type=float)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--clip_grad", default=1., type=float)
    parser.add_argument("--entropy_reg", default=0.1, type=float)
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--dim_mlp", default=256, type=int)
    parser.add_argument("--attn_heads", default=4, type=int)
    parser.add_argument("--transformer_layers", default=4, type=int)

    parser.add_argument("--num_qubits", default=5, type=int)
    parser.add_argument("--num_iters", default=1001, type=int)
    parser.add_argument("--num_envs", default=128, type=int)
    parser.add_argument("--steps", default=64, type=int)
    parser.add_argument("--steps_limit", default=40, type=int)
    parser.add_argument("--epsi", default=1e-3, type=float)
    parser.add_argument("--reward_fn", default="relative_delta", type=str)
    parser.add_argument("--obs_fn", default="rdm_2q_mean_real", type=str)
    parser.add_argument("--p_gen", default=0.95, type=float)

    parser.add_argument("--log_every", default=100, type=int)
    parser.add_argument("--demo_every", default=100, type=int)

    args = parser.parse_args()
    pg_solves_quantum(args)

#