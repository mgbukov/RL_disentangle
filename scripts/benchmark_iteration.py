import argparse
import json
import os
import time

import context
from src.config import get_default_config
from src.qenv import QEnv
from src.ppo import PPOAgent
from src.envloop import envloop
from src.networks import TransformerPE_2qRDM, MLP


NUM_QUBITS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
EMBED_DIM = 256
DIM_MLP = 1024
NUM_HEADS = 4
NUM_LAYERS = 4
STEPS = 10
NUM_ITERS = 10


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--fast_obs", action="store_true", default=False)
    parser.add_argument("--fast_ents", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--env_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--agent_device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    timings = {}
    agent_config = dict(
        pi_lr=1e-4, vf_lr=1e-4, discount=0.99, batch_size=256,
        clip_grad=10.0, entropy_reg=0.01, pi_clip=0.2, vf_clip=10.0,
        tgt_KL=0.01, n_epochs=3, lamb=0.95
    )

    for num_qubits in NUM_QUBITS:

        print(f"\nStarting benchmark for {num_qubits} qubits")
        # Initialize default config
        config = get_default_config()
        config["env_device"] = args.env_device
        config["model_device"] = args.agent_device

        # Initialize RL environment
        env = QEnv(num_qubits=num_qubits, num_envs=args.num_envs, epsi=1e-3,
                   fast_ents=args.fast_ents, fast_obs=args.fast_obs,
                   device=args.env_device)

        # Initialize RL agent
        in_dim = env.single_observation_space.shape
        value_net = MLP(in_dim, [128, 256], 1).to(device=args.agent_device)
        policy_net = TransformerPE_2qRDM(
            in_dim[1], embed_dim=EMBED_DIM, dim_mlp=DIM_MLP,
            n_heads=NUM_HEADS, n_layers=NUM_LAYERS
        ).to(device=args.agent_device)
        agent = PPOAgent(policy_net, value_net, agent_config)

        # Run training loop
        tic = time.time()
        envloop(agent, env, num_iters=NUM_ITERS, steps=STEPS, config=config)
        toc = time.time()
        timings[num_qubits] = toc - tic
        print(f"10 iterations for {num_qubits}-qubit system took {toc - tic:.2f} sec.")
        del agent
        del policy_net
        del value_net
        del env

    # Create output dict
    result = {
        "fast_obs":         args.fast_obs,
        "fast_ents":        args.fast_ents,
        "num_envs":         args.num_envs,
        "env_device":       args.env_device,
        "agent_device":     args.agent_device,
        "embed_dim":        EMBED_DIM,
        "dim_mlp":          DIM_MLP,
        "n_heads":          NUM_HEADS,
        "n_layers":         NUM_LAYERS,
        "steps":            STEPS,
        "num_iters":        NUM_ITERS,
        "timings":          timings
    }

    # Save to JSON
    with open(args.output, mode="wt") as f:
        json.dump(result, f, indent=2)

    print("Done")
