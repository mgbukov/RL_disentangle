import os
import pickle
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.environment_loop import environment_loop
from src.networks import MLP, Transformer, TransformerPE
from src.vpg import VPGAgent
from src.ppo import PPOAgent
from src.quantum_env import QuantumEnv


def demo(args):

    def thunk(i, agent):
        if i % args.demo_every != 0:
            return

        num_envs = 1024
        env = QuantumEnv(num_qubits=args.num_qubits, num_envs=num_envs,
            epsi=args.epsi, max_episode_steps=args.steps_limit,
            reward_fn=args.reward_fn, obs_fn=args.obs_fn,
        )
        _ = env.reset()
        env.simulator.set_random_states_() # test on fully-entangled states
        o = env.obs_fn(env.simulator.states)

        returns, lengths = [None] * env.num_envs, [None] * env.num_envs
        done = np.array([False] * env.num_envs)
        solved = 0
        while not done.all():
            o = torch.from_numpy(o)
            pi = agent.policy(o)    # uses torch.no_grad
            acts = torch.argmax(pi.probs, dim=1).cpu().numpy()
            o, r, t, tr, infos = env.step(acts)

            if (t | tr).any():
                for k in range(env.num_envs):
                    if (t[k] | tr[k]) and (returns[k] is None):
                        returns[k] = infos["episode"]["r"][k]
                        lengths[k] = infos["episode"]["l"][k]
                        done[k] = True
                        if t[k]: solved += 1

        agent.train_history[i].update({
            "test_avg_r" : np.mean(returns),
            "test_std_r" : np.std(returns),
            "test_avg_l" : np.mean(lengths),
            "test_std_l" : np.std(lengths),
            "test_solved": solved / num_envs,
        })

    return thunk


def pg_solves_quantum(args):
    # Use cuda.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seeds.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create a vectorized quantum environment for simulating quantum states.
    # Pass the reward and observation functions to be used with the environment.
    env = QuantumEnv(
        num_qubits=args.num_qubits,
        num_envs=args.num_envs,
        epsi=args.epsi,
        max_episode_steps=args.steps_limit,
        reward_fn=args.reward_fn,
        obs_fn=args.obs_fn,
    )

    # Create the RL agent.
    in_shape = env.single_observation_space.shape
    in_dim = in_shape[1]
    out_dim = env.single_action_space.n
    # policy_network = MLP(in_shape, [256, 256, 256], out_dim).to(device)
    # policy_network = Transformer(in_shape, embed_dim=128, hid_dim=256, out_dim=out_dim,
    #     dim_mlp=128, n_heads=4, n_layers=2).to(device)
    policy_network = TransformerPE(in_dim, embed_dim=128, dim_mlp=128, n_heads=4, n_layers=2).to(device)
    value_network = MLP(in_shape, [256, 256], 1).to(device)
    agent = VPGAgent(policy_network, value_network, config={
        "pi_lr"     : args.pi_lr,
        "vf_lr"     : args.vf_lr,
        "discount"  : args.discount,
        "batch_size": args.batch_size,
        "clip_grad" : args.clip_grad,
        "entropy_reg": args.entropy_reg,
    })

    # agent = PPOAgent(policy_network, value_network, config={
    #     "pi_lr"     : args.pi_lr,
    #     "vf_lr"     : args.vf_lr,
    #     "discount"  : args.discount,
    #     "batch_size": args.batch_size,
    #     "clip_grad" : args.clip_grad,
    #     "entropy_reg": args.entropy_reg,

    #     # PPO-specific
    #     "pi_clip" : 0.2,
    #     "vf_clip" : 10.,
    #     "tgt_KL"  : 0.01,
    #     "n_epochs": 3,
    #     "lamb"    : 0.95,
    # })

    # Run the environment loop
    log_dir = os.path.join("logs",
        f"pg_pomdp_TPE_{args.num_qubits}q_R{args.reward_fn}_iters_{args.num_iters}_ent_{args.entropy_reg}_pilr_{args.pi_lr}")
    os.makedirs(log_dir, exist_ok=True)
    environment_loop(seed, agent, env, args.num_iters, args.steps, log_dir, args.log_every, demo=demo(args))
    plot_progress(log_dir)


def plot_progress(log_dir):
    with open(os.path.join(log_dir, "train_history.pickle"), "rb") as f:
        train_history = pickle.load(f)

    # Unpack training data.
    num_iters = len(train_history)
    run_return = np.array([train_history[i]["run_return"] for i in range(num_iters)])
    avg_return = np.array([train_history[i]["avg_return"] for i in range(num_iters)])
    std_return = np.array([train_history[i]["std_return"] for i in range(num_iters)])
    run_length = np.array([train_history[i]["run_length"] for i in range(num_iters)])
    avg_length = np.array([train_history[i]["avg_length"] for i in range(num_iters)])
    std_length = np.array([train_history[i]["std_length"] for i in range(num_iters)])
    policy_entropy = np.array([train_history[i]["policy_entropy"] for i in range(num_iters)])
    vf_loss = np.array([train_history[i]["value_avg_loss"] for i in range(num_iters)])
    pi_loss = np.array([train_history[i]["total_loss"] for i in range(num_iters)])
    pi_gnorm = np.array([train_history[i]["policy_grad_norm"] for i in range(num_iters)])
    ratio_terminated = np.array([
        train_history[i]["terminated"] / train_history[i]["total_ep"]
        if train_history[i]["total_ep"] > 0 else 0
        for i in range(num_iters)
    ])

    # Unpack test data.
    test_avg_r = np.array([train_history[i]["test_avg_r"]
        for i in range(num_iters) if "test_avg_r" in train_history[i].keys()])
    test_std_r = np.array([train_history[i]["test_std_r"]
        for i in range(num_iters) if "test_std_r" in train_history[i].keys()])
    test_avg_l = np.array([train_history[i]["test_avg_l"]
        for i in range(num_iters) if "test_avg_l" in train_history[i].keys()])
    test_std_l = np.array([train_history[i]["test_std_l"]
        for i in range(num_iters) if "test_std_l" in train_history[i].keys()])
    test_solved = np.array([train_history[i]["test_solved"]
        for i in range(num_iters) if "test_solved" in train_history[i].keys()])
    num_test = len(test_avg_r)

    plt.style.use("ggplot")

    # Plot training results.
    # Returns.
    fig, ax = plt.subplots()
    ax.plot(run_return, label="Running Return", lw=2.)
    ax.plot(avg_return, label="Average Return", lw=0.75)
    ax.fill_between(np.arange(num_iters), avg_return - 0.5*std_return, avg_return + 0.5*std_return, color="k", alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Accumulated return")
    fig.savefig(os.path.join(log_dir, "returns.png"))
    # Lengths.
    fig, ax = plt.subplots()
    ax.plot(run_length, label="Running Length", lw=2.)
    ax.plot(avg_length, label="Average Length", lw=0.75)
    ax.fill_between(np.arange(num_iters), avg_length - 0.5*std_length, avg_length + 0.5*std_length, color="k", alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Episode lengths")
    fig.savefig(os.path.join(log_dir, "lengths.png"))
    # Policy entropy.
    fig, ax = plt.subplots()
    ax.plot(policy_entropy, lw=0.8)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Average policy entropy")
    fig.savefig(os.path.join(log_dir, "policy_entropy.png"))
    # Plot % of terminated.
    fig, ax = plt.subplots()
    ax.plot(ratio_terminated, lw=0.8)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Ratio of terminated episodes")
    fig.savefig(os.path.join(log_dir, "terminated.png"))
    # Value function loss.
    fig, ax = plt.subplots()
    ax.plot(vf_loss, lw=0.8)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Value network loss")
    fig.savefig(os.path.join(log_dir, "vf_loss.png"))
    # Policy loss.
    fig, ax = plt.subplots()
    ax.plot(pi_loss, lw=0.8)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Policy network loss")
    fig.savefig(os.path.join(log_dir, "pi_loss.png"))
    # Policy grad norm.
    fig, ax = plt.subplots()
    ax.plot(pi_gnorm, lw=0.8)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Policy network grad norm")
    fig.savefig(os.path.join(log_dir, "policy_grad_norm.png"))

    # Plot test results.
    xs = np.linspace(0, num_iters, num_test)
    # Returns.
    fig, ax = plt.subplots()
    ax.plot(xs, test_avg_r)
    ax.fill_between(xs, test_avg_r - 0.5*test_std_r, test_avg_r + 0.5*test_std_r, color="k", alpha=0.25)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Accumulated return")
    fig.savefig(os.path.join(log_dir, "test_returns.png"))
    # Lengths.
    fig, ax = plt.subplots()
    ax.plot(xs, test_avg_l)
    ax.fill_between(xs, test_avg_l - 0.5*test_std_l, test_avg_l + 0.5*test_std_l, color="k", alpha=0.25)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Accumulated return")
    fig.savefig(os.path.join(log_dir, "test_lengths.png"))
    # Solved.
    fig, ax = plt.subplots()
    ax.plot(xs, test_solved, lw=0.8)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Ratio of terminated episodes")
    fig.savefig(os.path.join(log_dir, "test_solved.png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi_lr", default=3e-4, type=float)
    parser.add_argument("--vf_lr", default=3e-4, type=float)
    parser.add_argument("--discount", default=1., type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--clip_grad", default=1., type=float)
    parser.add_argument("--entropy_reg", default=1e-2, type=float)

    parser.add_argument("--num_qubits", default=5, type=int)
    parser.add_argument("--num_iters", default=1001, type=int)
    parser.add_argument("--num_envs", default=128, type=int)
    parser.add_argument("--steps", default=40, type=int)
    parser.add_argument("--steps_limit", default=40, type=int)
    parser.add_argument("--epsi", default=1e-3, type=float)
    parser.add_argument("--reward_fn", default="sparse", type=str)
    parser.add_argument("--obs_fn", default="phase_norm", type=str)

    parser.add_argument("--log_every", default=100, type=int)
    parser.add_argument("--demo_every", default=100, type=int)

    args = parser.parse_args()
    pg_solves_quantum(args)

#