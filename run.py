"""Run
This script runs the training procedure.
The updates to the agent policy are performed using PPO.
The policy network is a Transformer model, the value network is a fully-connected
model. Training log is stored inside `logs/model`.
When training finishes the model is tested on a set of specially defined states.
Test results are stored in `logs/model/results.json`.

The model configuration as well as the training parameters can be set by
providing the respective command line parameters. Run `python3 run.py --help` to
see the list of parameters that can be provided.

Example usage:

python3 run.py \
    --seed 0 --num_qubits 4 --num_envs 32 --steps 16 --steps_limit 8 --num_iters 1001 \
    --attn_heads 2 --transformer_layers 2 --embed_dim 128 --dim_mlp 256 \
    --batch_size 512 --pi_lr 1e-4 --entropy_reg 0.1 --obs_fn rdm_2q_real
    --state_generator haar_geom --p_gen 0.9
    --checkpoint_every 100
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.environment_loop import environment_loop
from src.networks import MLP, TransformerPE_2qRDM
from src.ppo import PPOAgent
from src.quantum_env import QuantumEnv
from src.util import str2state


def demo(args):

    def thunk(i, agent):
        if i % args.demo_every != 0:
            return

        num_envs = 100
        env = QuantumEnv(
            num_qubits=         args.num_qubits,
            num_envs=           num_envs,
            epsi=               args.epsi,
            max_episode_steps=  args.steps_limit,
            reward_fn=          args.reward_fn,
            obs_fn=             args.obs_fn,
            state_generator=    args.state_generator,
            generator_kwargs=   dict(
                                        p_gen=          args.p_gen,
                                        min_entangled=  args.min_entangled,
                                        max_entangled=  args.max_entangled if args.max_entangled > 0 else None,
                                        chi_max=        args.chi_max
                                    )
        )
        _ = env.reset() # prepare the environment

        # Generate the initial states as fully-entangled states.
        # env.simulator.set_random_states_()
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


def test(agent, args):
    """Test the agent on a set of specifically generated quantum states.
    Note that this function will reset the numpy rng seed.
    """
    # Define the initial states on which we want to test the agent. For each
    # of the special configurations provide a generating function.
    initial_states = {
        4: ["RR-R-R", "RR-RR", "RRR-R", "RRRR"],
        5: ["RR-R-R-R", "RR-RR-R", "RRR-R-R", "RRR-RR", "RRRR-R", "RRRRR"],
        6: ["RR-R-R-R-R", "RR-RR-R-R", "RR-RR-RR", "RRR-R-R-R", "RRR-RR-R",
            "RRRR-R-R", "RRRR-RR", "RRRRR-R", "RRRRRR"]
    }

    # Define the environment.
    num_envs = 1024
    env = QuantumEnv(num_qubits=args.num_qubits, num_envs=num_envs,
        epsi=args.epsi, max_episode_steps=args.steps_limit, obs_fn=args.obs_fn,
    )

    # Try to solve each of the special configurations.
    results = {}
    for name in initial_states[args.num_qubits]:
        np.random.seed(args.seed)
        states = np.array([str2state(name) for _ in range(num_envs)])
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
        results["|" + name + ">"] = {
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
        max_episode_steps=args.steps_limit,
        reward_fn=args.reward_fn,
        obs_fn=args.obs_fn,
        state_generator=args.state_generator,
        generator_kwargs=dict(
            p_gen=args.p_gen,
            min_entangled=args.min_entangled,
            max_entangled=args.max_entangled if args.max_entangled > 0 else None,
            chi_max=args.chi_max
        )
    )

    # Initialize value network
    in_shape = env.single_observation_space.shape
    value_network = MLP(in_shape, [256, 256], 1).to(device)

    # Try loading the RL agent from checkpoint or ...
    if args.agent_checkpoint:
        try:
            agent = torch.load(args.agent_checkpoint, map_location=device)
            # Re-initialize value network
            if args.reset_value_network:
                agent.value_network = value_network
        except Exception as ex:
            print("Cannot load agent from checkpoint:")
            print('\t', ex)
            exit(1)
        if args.reset_optimizers:
            print("Resetting optimizers state...")
            agent.policy_optim = type(agent.policy_optim)(agent.policy_network.parameters(), lr=args.pi_lr)
            agent.value_optim = type(agent.value_optim)(agent.value_network.parameters(), lr=args.vf_lr)
        # Set parameters
        agent.pi_lr =       args.pi_lr
        agent.vf_lr =       args.vf_lr
        agent.discount =    args.discount
        agent.batch_size =  args.batch_size
        agent.clip_grad =   args.clip_grad
        agent.entropy_reg = args.entropy_reg
        agent.pi_clip =     0.2
        agent.vf_clop =     10.0
        agent.tgt_KL =      0.01
        agent.n_epochs =    3
        agent.lamb =        0.95
    # ... create the RL agent.
    else:
        in_dim = in_shape[1]
        out_dim = env.single_action_space.n
        policy_network = TransformerPE_2qRDM(
            in_dim,
            embed_dim=args.embed_dim,
            dim_mlp=args.dim_mlp,
            n_heads=args.attn_heads,
            n_layers=args.transformer_layers,
        ).to(device)
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

    # Create log directory
    log_dirname = f"{args.num_qubits}q_{args.num_iters}iters_" + args.suffix
    log_dir = os.path.join("logs", log_dirname)
    os.makedirs(log_dir, exist_ok=True)

    # Save the cmd arguments to text file in log directory
    with open(os.path.join(log_dir, "args.txt"), mode='wt') as f:
        f.write("{d:<20} = {dd}\n".format(d="device", dd=str(device)))
        for arg, val in args._get_kwargs():
            f.write(f"{arg:<20} = {val}\n")

    # Run the environment loop
    environment_loop(seed, agent, env, args.num_iters, args.steps, log_dir,
                     args.log_every, args.checkpoint_every, demo=demo(args))

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

    # # Test the final agent and store the results.
    # results = test(agent, args)
    # with open(os.path.join(log_dir, "results.json"), "w") as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int,
        help="Seed for random number generation.")

    parser.add_argument("--pi_lr", default=1e-4, type=float,
        help="Policy network learning rate.")
    parser.add_argument("--vf_lr", default=3e-4, type=float,
        help="Value network learning rate.")
    parser.add_argument("--discount", default=1., type=float,
        help="Discount factor for future rewards.")
    parser.add_argument("--batch_size", default=2048, type=int,
        help="Batch size for PPO iterations.")
    parser.add_argument("--clip_grad", default=1., type=float,
        help="Clip value for gradient clipping by norm.")
    parser.add_argument("--entropy_reg", default=0.1, type=float,
        help="Entropy regularization parameter.")
    parser.add_argument("--embed_dim", default=256, type=int,
        help="Embedding dimension for self-attention keys, queries and values.")
    parser.add_argument("--dim_mlp", default=256, type=int,
        help="Transformer encoder layer MLP dimension size.")
    parser.add_argument("--attn_heads", default=4, type=int,
        help="Number of attention heads per transformer encoder layer.")
    parser.add_argument("--transformer_layers", default=4, type=int,
        help="Number of transformer layers.")

    parser.add_argument("--num_qubits", default=5, type=int,
        help="Number of qubits in the quantum state.")
    parser.add_argument("--num_iters", default=1001, type=int,
        help="Number of training iterations.")
    parser.add_argument("--num_envs", default=128, type=int,
        help="Number of parallel environments.")
    parser.add_argument("--steps", default=64, type=int,
        help="Number of episode steps.")
    parser.add_argument("--steps_limit", default=40, type=int,
        help="Maximum steps before truncating an environment.")
    parser.add_argument("--epsi", default=1e-3, type=float,
        help="Threshold for disentanglement.")
    parser.add_argument("--reward_fn", default="relative_delta", type=str,
        help="The name of the reward function to be used. One of " \
             "['sparse', 'relative_delta'].")
    parser.add_argument("--obs_fn", default="rdm_2q_mean_real", type=str,
        help="The name of the observation function to be used. One of " \
             "['phase_norm', 'rdm_1q', 'rdm_2q_real', rdm_2q_mean_real']")
    parser.add_argument("--state_generator", default="haar_geom", type=str,
        help="Name of quantum state generator used in reset() function.")
    parser.add_argument("--p_gen", default=0.95, type=float,
        help="Parameter of `haar_geom` state generator.")
    parser.add_argument("--min_entangled", default=1, type=int,
        help="Parameter of `haar_unif` state generator.")
    parser.add_argument("--max_entangled", default=-1, type=int,
        help="Parameter of `haar_unif` state generator.")
    parser.add_argument("--chi_max", default=2, type=int,
        help="Maximum bond dimension for `mps` state generator.")

    parser.add_argument("--log_every", default=100, type=int,
        help="Log training data ${log_every} iterations.")
    parser.add_argument("--checkpoint_every", default=500, type=int,
        help="Checkpoint model every ${checkpoint_every} iterations.")
    parser.add_argument("--demo_every", default=100, type=int,
        help="Demo the agent every ${demo_every} iterations.")
    parser.add_argument("--agent_checkpoint", default='', type=str,
        help="Path to checkpointed agent")
    parser.add_argument("--reset_optimizers", action="store_true",
        help="Reset the state of the optimizers")
    parser.add_argument("--reset_value_network", action="store_true",
        help="Reinitialize the value network")
    parser.add_argument("--suffix", type=str, default='',
        help="Suffix appended to log directory name")

    args = parser.parse_args()
    pg_solves_quantum(args)

#