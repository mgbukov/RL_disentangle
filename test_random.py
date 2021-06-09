import time

import numpy as np

from src.envs.batched_env import QubitsEnvironment
from src.policies.fcnn_policy import FCNNPolicy
from src.agents.pg_agent import PGAgent
from src.utils.util_funcs import fix_random_seeds


def main():
    seed = 0
    fix_random_seeds(seed)

    # Create the environment.
    num_qubits = 2
    epsi = 1e-3
    batch_size = 128
    env = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=batch_size)

    # Initialize the policy.
    input_size = 2 ** (num_qubits + 1)
    hidden_dims = [256, 256, 256]
    output_size = env.num_actions
    dropout_rate = 0.25
    policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate)

    # # Train the policy-gradient agent.
    # num_episodes = 1000
    steps = 3
    # learning_rate = 1e-3
    # lr_decay = np.power(0.05, 1.0/num_episodes)
    # reg = 1e-4
    # log_every = 10
    # verbose = True
    log_dir = "logs/2qubits/random"
    agent = PGAgent(env, policy, log_dir=log_dir)

    # batch_mode = False
    # initial_states = None
    # agent.train(num_episodes, steps, learning_rate, lr_decay, reg, log_every,
    #             verbose, initial_states, batch_mode)
    # agent.policy.save(log_dir + "/policy.bin")

    num_test = 100
    agent.logger.verboseTxtLogging(True)
    agent.log_test_accuracy(num_test, steps)

if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print("Training took {:.3f} seconds".format(toc-tic))

#