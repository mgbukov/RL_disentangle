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
    out_dim = env.single_action_space.n
    policy_network = MLP(in_shape, [256, 256, 256], out_dim).to(device)                                                                   # works for 5 qubits
    # policy_network = Transformer(in_shape, embed_dims=128, hid_dim=256, out_dim=out_dim, dim_mlp=128, n_head=4, n_layers=2).to(device)    # works for 5 qubits
    # value_network = MLP(in_shape, [128, 128], 1).to(device)                                                                               # works for 5 qubits
    # embed_dims = 128
    # hid_dim = 256
    # dim_mlp = 128
    # policy_network = Transformer(
    #     in_shape,
    #     embed_dims,
    #     hid_dim,
    #     out_dim,
    #     dim_mlp,
    #     n_heads=4,
    #     n_layers=4,
    # ).to(device) # bigger transformer used for 6-qubits
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
    #     "tgt_KL"  : 0.02,
    #     "n_epochs": 3,
    #     "lamb"    : 0.95,
    # })

    # Run the environment loop
    log_dir = os.path.join("logs", f"6q_Rscale_pg_pomdp_T_BnoPE_iters_{args.num_iters}_ent_{args.entropy_reg}")
    log_every = 1
    os.makedirs(log_dir, exist_ok=True)
    environment_loop(seed, agent, env, args.num_iters, args.steps, log_dir, log_every, demo=None)
    plot_progress(log_dir)


def plot_progress(log_dir):
    with open(os.path.join(log_dir, "train_history.pickle"), "rb") as f:
        train_history = pickle.load(f)

    num_iters = len(train_history)
    run_return = np.array([train_history[i]["run_return"] for i in range(num_iters)])
    avg_return = np.array([train_history[i]["avg_return"] for i in range(num_iters)])
    std_return = np.array([train_history[i]["std_return"] for i in range(num_iters)])
    run_length = np.array([train_history[i]["run_length"] for i in range(num_iters)])
    avg_length = np.array([train_history[i]["avg_length"] for i in range(num_iters)])
    std_length = np.array([train_history[i]["std_length"] for i in range(num_iters)])
    policy_entropy = np.array([train_history[i]["policy_entropy"] for i in range(num_iters)])
    terminated = np.array([train_history[i]["terminated"] for i in range(num_iters)])
    total_ep = np.array([train_history[i]["total_ep"] for i in range(num_iters)])

    plt.style.use("ggplot")

    # Plot returns.
    fig, ax = plt.subplots()
    ax.plot(run_return, label="Running Return", lw=2.)
    ax.plot(avg_return, label="Average Return", lw=0.75)
    ax.fill_between(np.arange(num_iters), avg_return - 0.5*std_return, avg_return + 0.5*std_return, color="k", alpha=0.25)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(log_dir, "returns.png"))

    # Plot episode lengths.
    fig, ax = plt.subplots()
    ax.plot(run_length, label="Running Length", lw=2.)
    ax.plot(avg_length, label="Average Length", lw=0.75)
    ax.fill_between(np.arange(num_iters), avg_length - 0.5*std_length, avg_length + 0.5*std_length, color="k", alpha=0.25)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(log_dir, "lengths.png"))

    # Plot policy entropy.
    fig, ax = plt.subplots()
    ax.plot(policy_entropy, lw=0.8)
    fig.savefig(os.path.join(log_dir, "policy_entropy.png"))

    # Plot % of terminated.
    fig, ax = plt.subplots()
    ax.plot(terminated / total_ep, lw=0.8)
    fig.savefig(os.path.join(log_dir, "terminated.png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi_lr", default=3e-4, type=float)
    parser.add_argument("--vf_lr", default=3e-4, type=float)
    parser.add_argument("--discount", default=1., type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--clip_grad", default=10., type=float)
    parser.add_argument("--entropy_reg", default=1e-2, type=float)

    parser.add_argument("--num_qubits", default=5, type=int)
    parser.add_argument("--num_iters", default=1001, type=int)
    parser.add_argument("--num_envs", default=128, type=int)
    parser.add_argument("--steps", default=40, type=int)
    parser.add_argument("--steps_limit", default=40, type=int)
    parser.add_argument("--epsi", default=1e-3, type=float)
    parser.add_argument("--reward_fn", default="sparse", type=str)
    parser.add_argument("--obs_fn", default="phase_norm", type=str)

    args = parser.parse_args()
    pg_solves_quantum(args)

#