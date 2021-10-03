import time
import os

import numpy as np

from src.envs.rdm_environment import QubitsEnvironment
from src.policies.fcnn_policy import FCNNPolicy
from src.agents.pg_agent import PGAgent
from src.utils.util_funcs import fix_random_seeds


def main(seed=0, batch_size=64, entropy_reg=0.0):
    fix_random_seeds(seed)
 
    # Create the environment.
    num_qubits = 4
    epsi = 1e-3
    batch_size = 256
    env = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=batch_size)

    # Initialize the policy.
    input_size = 2 ** (num_qubits + 1)
    hidden_dims = [4096, 2048, 512]
    output_size = env.num_actions
    dropout_rate = 0.0
    policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate)

    # Train the policy-gradient agent.
    num_episodes = 10001
    steps = 5               # we should get rid of this parameter and run trajectories until the end!
    learning_rate = 1e-4
    lr_decay = 1.0 # np.power(0.1, 1.0/num_episodes)
    clip_grad = 1.0
    reg = 0.0
    entropy_reg = 0.0       # try 0.1, or 1.0, or sth else
    log_every = 10
    test_every = 1000
    verbose = True
    # log_dir = "logs/4qubits/traj_{}_episodes_{}_hid_{}_phaseNorm".format(batch_size, num_episodes, hidden_dims)
    log_dir = "logs/4qubits/unique_batch"
    agent = PGAgent(env, policy, log_dir=log_dir)
    agent.train(num_episodes, steps, learning_rate, lr_decay, clip_grad, reg, entropy_reg,
                log_every, test_every, verbose)

    # num_test = 10
    # agent.logger.setLogTxtFilename("final_test.txt")
    # agent.log_test_accuracy(num_test=num_test, steps=steps)
    # agent.log_test_accuracy(num_test=num_test, steps=steps+1)
    # agent.log_test_accuracy(num_test=num_test, steps=steps+2)


if __name__ == "__main__":
    seed = 0
    tic = time.time()
    main(seed)
    toc = time.time()
    print("Training took {:.3f} seconds".format(toc-tic))

#