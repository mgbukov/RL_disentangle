"""
Train a PPO agent to disentangle quantum states.
The updates to the agent policy are performed using PPO.
The policy network is a Transformer model, the value network is a fully-connected
model. Training log is stored inside `logs/model`.
When training finishes the model is tested on a set of specially defined states.
Test results are stored in `logs/model/results.json`.

The model configuration as well as the training parameters can be set by
providing the respective command line parameters. Run `python3 run.py --help` to
see the list of parameters that can be provided.

Example usage:

python3 train.py -c config.yaml
"""
import argparse
import json
import os
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import get_default_config, get_logdir
from src.environment_loop import environment_loop
from src.networks import TransformerPE_2qRDM, PermutationInvariantMLP
from src.ppo import PPOAgent
from src.quantum_env import QuantumEnv
from src.metrics import Tracker
import src.stategen as stategen
import src.triggers as triggers


def load_checkpoint(config):
    assert config.agent_checkpoint != ''

    device = torch.device(config.device)
    try:
        agent = torch.load(config.agent_checkpoint, map_location=device)
    except Exception as ex:
        logging.error("Loading from checkpoint failed! " + str(ex))
        return

    logging.info("Checkpoint loaded successfully!")

    if config.reset_optimizers:
        logging.info("Resetting optimizers state")
        agent.policy_optim = type(agent.policy_optim)(
            agent.policy_network.parameters(),
            lr=config.PI_LR
        )
        agent.value_optim = type(agent.value_optim)(
            agent.value_network.parameters(),
            lr=config.VF_LR
        )

    # Set parameters
    agent.discount =    config.discount
    agent.batch_size =  config.batch_size
    agent.clip_grad =   config.clip_grad
    agent.entropy_reg = config.entropy_reg

    return agent


def init_triggers(config, agent, env):
    triggers_list = []
    for name in config.triggers:
        match name:
            case "StagedStateGeneratorTrigger":
                x = triggers.StagedStateGeneratorTrigger(config, agent, env)
                triggers_list.append(x)
            case _:
                logging.error("Trigger \"{name}\" is not defined. Ignorring...")
    return triggers_list


def train(config):
    """
    Initializes the RL environment and the agent and enters the environment loop
    """

    # Initialize tracker
    tracker = Tracker("train")

    # Initialize state generator / sampler
    stategen_fn_name = config.stategen_fn
    stategen_fn = getattr(stategen, stategen_fn_name, None)
    if stategen_fn is None:
        logging.error("State generation function \"{stategen_fn_name}\" is not defined!")
        return
    stategen_params = dict(config.stategen_params)
    state_generator = stategen.StateGenerator(stategen_fn, config.num_qubits, stategen_params)
    logging.debug("Initialized state generator. Parameters:")
    for name, val in stategen_params.items():
        logging.debug(f"\t{name} = {val}")

    # Initialize RL environment
    env = QuantumEnv(
        num_qubits=             config.num_qubits,
        num_envs=               config.num_envs,
        epsi=                   config.epsi,
        max_episode_steps=      config.steps_limit,
        reward_fn=              config.reward_fn,
        obs_fn=                 config.obs_fn,
        state_generator=        state_generator
    )
    logging.debug("Initialized RL environment")

    # Initialize value function
    in_shape = env.single_observation_space.shape
    value_network = PermutationInvariantMLP(in_shape[-1], [128, 256], 1).to(device)
    logging.debug("Initialized value network")

    # Initialize policy function
    policy_network = TransformerPE_2qRDM(
        in_shape[1],
        embed_dim=      config.embed_dim,
        dim_mlp=        config.dim_mlp,
        n_heads=        config.attn_heads,
        n_layers=       config.transformer_layers
    )

    # Load checkpoint if any
    if config.agent_checkpoint:
        agent = load_checkpoint(config)
        if agent is None:
            return
        # Reset the tracker or get reference to the checkpointer tracker
        if config.reset_tracker:
            agent.tracker = tracker
        else:
            tracker = agent.tracker
    # Or initialize the agent
    else:
        agent = PPOAgent(policy_network, value_network, tracker=tracker, config={
            "pi_lr":        config.pi_lr,
            "vf_lr":        config.vf_lr,
            "discount":     config.discount,
            "batch_size":   config.batch_size,
            "clip_grad":    config.clip_grad,
            "entropy_reg":  config.entropy_reg,

            # PPO-specific
            "pi_clip":      0.2,
            "vf_clip":      10.0,
            "tgt_KL":       0.01,
            "n_epochs":     3,
            "lamb":         0.95
        })

    # Log agent parameters
    logging.info(f"\tagent.discount =     {agent.discount}")
    logging.info(f"\tagent.batch_size =   {agent.batch_size}")
    logging.info(f"\tagent.clip_grad =    {agent.clip_grad}")
    logging.info(f"\tagent.entropy_reg =  {agent.entropy_reg}")
    logging.info(f"\tagent.pi_clip =      {agent.pi_clip}")
    logging.info(f"\tagent.vf_clip =      {agent.vf_clip}")
    logging.info(f"\tagent.tgt_KL =       {agent.tgt_KL}")
    logging.info(f"\tagent.n_epochs =     {agent.n_epochs}")
    logging.info(f"\tagent.lamb =         {agent.lamb}")

    # Initialize triggers
    triggers_list = init_triggers(config, agent, env)

    # Run the environment loop
    tic = time.time()
    environment_loop(agent, env, tracker, triggers_list,
                     config.num_iters, config.steps, config)
    toc = time.time()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(int(toc - tic)))
    logging.info(f"Training took {elapsed}")

    # Close the environment and save the agent and tracker.
    env.close()
    agent.save(logdir)
    tracker.save(os.path.join(logdir, "tracker.json"))

    # Save plots
    logdir = get_logdir(config)
    for name in tracker.get_names():
        figpath = os.path.join(logdir, f"{name}.png")
        tracker.plot_scalar(name, figpath, linewidth=1)

    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    config = get_default_config()
    config.merge_from_file(args.config)
    config.freeze()

    # Create log directory
    logdir = get_logdir(config)
    if os.path.exists(logdir):
        print(f"Log directory \"{logdir}\" already exists!")
        exit(1)
    else:
        os.makedirs(logdir, exist_ok=False)

    # Save config to text file in log directory
    with open(os.path.join(logdir, "config.yaml"), mode='wt') as f:
        f.write(config.dump())
        f.flush()

    # Initizize logger
    logging.basicConfig(format="%(message)s", filemode="w", level=logging.INFO,
        filename=os.path.join(logdir, "train.log"))

    # Set device
    device = torch.device(config.device)

    # Set random seeds
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Start training
    train(config)

