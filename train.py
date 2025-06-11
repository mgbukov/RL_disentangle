"""
Train a PPO agent to disentangle quantum states.
The policy network is a Transformer model, the value network is a
permutation-invariant MLP.

The model configuration as well as the training parameters can be set by
providing a YAML config file

Example usage:

python3 train.py -c config.yaml
"""
import argparse
import json
import os
import logging
import pickle
import time

import numpy as np
import torch

import src.metrics as metrics
import src.stategen as stategen
import src.triggers as triggers
import src.util as util
from src.config import get_default_config, get_logdir
from src.environment_loop import environment_loop
from src.networks import TransformerPE_2qRDM, PermutationInvariantMLP
from src.ppo import PPOAgent
from src.quantum_env import QuantumEnv



def init_triggers(config, agent, env, checkpointed_state=None):
    triggers_list = []
    for name in config.triggers:
        match name:
            case "StagedTrainingTrigger":
                x = triggers.StagedTrainingTrigger(config, agent, env)
                triggers_list.append(x)
            case _:
                logging.error("Trigger \"{name}\" is not defined. Ignorring...")

    # Load checkpoints if any
    if checkpointed_state is not None and config.checkpoint.use_triggers:
        checkpointed_triggers = checkpointed_state["triggers"]
        for trigger in triggers_list:
            try:
                state_dict = checkpointed_triggers[str(type(trigger))]
                trigger.load_state_dict(state_dict)
            except KeyError:
                logging.error(f"No checkpointed state found for `{str(type(trigger))}`")

    return triggers_list


def train_ppo(config):
    """
    Initializes the RL environment and the PPO agent and enters the environment loop
    """

    # Initialize tracker
    tracker = metrics.getTracker()

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
        state_generator=        state_generator,
        fast_obs=               config.fast_obs
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
    ).to(device)
    logging.debug("Initialized policy network")

    # Initialize RL agent
    agent = PPOAgent(policy_network, value_network, config={
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
    logging.debug("Initialized RL agent")


    # Load checkpint if any. Triggers checkpoint loading is postponed to their
    # initialization
    checkpointed_state = None
    if config.checkpoint.filepath:
        checkpointed_state = util.load_checkpoint(config)
        if config.checkpoint.use_policy_fn:
            agent.policy_network.load_state_dict(checkpointed_state["policy_fn"])
        if config.checkpoint.use_value_fn:
            agent.value_network.load_state_dict(checkpointed_state["value_fn"])
        if config.checkpoint.use_policy_optim:
            agent.policy_optim.load_state_dict(checkpointed_state["policy_optim"])
            # Update the learning rate
            for pg in agent.policy_optim.param_groups:
                pg["lr"] = config.pi_lr
        if config.checkpoint.use_value_optim:
            agent.value_optim.load_state_dict(checkpointed_state["value_optim"])
            # Update the learning rate
            for pg in agent.value_optim.param_groups:
                pg["lr"] = config.vf_lr
        if config.checkpoint.use_tracker:
            tracker.load_state_dict(checkpointed_state["tracker"])

    # Log agent parameters
    logging.debug(f"\tagent.discount =     {agent.discount}")
    logging.debug(f"\tagent.batch_size =   {agent.batch_size}")
    logging.debug(f"\tagent.clip_grad =    {agent.clip_grad}")
    logging.debug(f"\tagent.entropy_reg =  {agent.entropy_reg}")
    logging.debug(f"\tagent.pi_clip =      {agent.pi_clip}")
    logging.debug(f"\tagent.vf_clip =      {agent.vf_clip}")
    logging.debug(f"\tagent.tgt_KL =       {agent.tgt_KL}")
    logging.debug(f"\tagent.n_epochs =     {agent.n_epochs}")
    logging.debug(f"\tagent.lamb =         {agent.lamb}")
    policy_fn_descr = "\n\t\t".join(str(agent.policy_network).split('\n'))
    logging.debug(f"\tagent.policy_fn =\n\t{policy_fn_descr}\n")
    value_fn_descr = "\n\t\t".join(str(agent.value_network).split('\n'))
    logging.debug(f"\tagent.value_fn =\n\t{value_fn_descr}\n")
    policy_optim_descr = "\n\t\t".join(str(agent.policy_optim).split('\n'))
    logging.debug(f"\tagent.policy_optim =\n\t{policy_optim_descr}\n")
    value_optim_descr = "\n\t\t".join(str(agent.value_optim).split('\n'))
    logging.debug(f"\tagent.value_optim =\n\t{value_optim_descr}\n")

    # Initialize triggers
    triggers_list = init_triggers(config, agent, env, checkpointed_state)

    # Get start iteration number
    if checkpointed_state is not None and config.checkpoint.use_iteration:
        start_iter = checkpointed_state["iteration"] + 1
    else:
        start_iter = 1
    tracker.timestep = start_iter

    # Run the environment loop
    tic = time.time()
    environment_loop(agent, env, config.num_iters, config.steps, start_iter,
                     triggers_list, config)
    toc = time.time()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(int(toc - tic)))
    logging.info(f"\n\nTraining took {elapsed}")

    # Close the environment and save the agent and tracker.
    env.close()
    logdir = get_logdir(config)
    agent.save(logdir)
    with open(os.path.join(logdir, "agent-statedict.pickle"), mode='wb') as f:
        pickle.dump(agent.state_dict(), f)

    tracker.save(os.path.join(logdir, "tracker.pickle"))

    logging.disable()

    # Save plots
    for name in tracker.get_names():
        figpath = os.path.join(logdir, f"{name}.png")
        tracker.plot_scalar(name, figpath, linewidth=2.5)

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
    train_ppo(config)

