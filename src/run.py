import time

import numpy as np
import torch

from envs.batched_env import QubitsEnvironment
from policies.fcnn_policy import FCNNPolicy
from agents.pg_agent import PGAgent


def fix_random_seeds(seed):
    """ Manually set the seed for random number generation.
    Also set CuDNN flags for reproducible results using deterministic algorithms.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    fix_random_seeds(0)

    # Create the environment.
    num_qubits = 2
    epsi = 1e-3
    num_trajectories = 256
    num_repetitions = 1
    env = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=num_trajectories, pack_size=num_repetitions)

    # Initialize the policy.
    input_size = 2 ** (num_qubits + 1)
    hidden_dims = [2048, 2048]
    output_size = env.num_actions
    dropout_rate = 0.0
    policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate)

    # Train the policy-gradient agent.
    num_episodes = 100
    steps = 3
    learning_rate = 1e-4
    lr_decay = 0.95
    reg = 0.0
    log_every = 100
    verbose = True
    policy_save_path = "bin/vanilla_pg_final.bin"
    agent = PGAgent(env, policy)
    agent.train(num_episodes, steps, learning_rate, lr_decay, reg, log_every, verbose)
    agent.policy.save(policy_save_path)

    # Test the model.
    num_test = 1000
    agent.test_accuracy(num_test, steps)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print("Training took {:.3f} seconds".format(toc-tic))

#