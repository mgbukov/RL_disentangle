import time
import os

import numpy as np

from src.envs.batched_env import QubitsEnvironment
from src.policies.fcnn_policy import FCNNPolicy
from src.agents.pg_agent import PGAgent
from src.utils.util_funcs import fix_random_seeds


def main(reward_func=None, seed=0, batch_size=64, entropy_reg=0.0):
    fix_random_seeds(seed)
 
    # Create the environment.
    num_qubits = 2
    epsi = 1e-3
    batch_size = 256    # try 512
    env = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=batch_size, reward_func=reward_func)
    for k, v in env.idx_to_operator.items():
        print(k, ":", v)

    # Initialize the policy.
    input_size = 2 ** (num_qubits + 1)
    hidden_dims = [4096, 4096, 512]
    output_size = env.num_actions
    dropout_rate = 0.0
    policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate)
    # policy = FCNNPolicy.load("logs/2qubits/tests/entropy_reg/policy.bin")

    # Train the policy-gradient agent.
    num_episodes = 10001
    steps = 3               # we should get rid of this parameter and run trajectories until the end!
    learning_rate = 1e-4
    lr_decay = 1.0 # np.power(0.1, 1.0/num_episodes)
    clip_grad = 10.0
    reg = 0.0
    entropy_reg = 1.0       # try 0.1, or 1.0, or sth else
    log_every = 100
    test_every = 1000
    verbose = True
    log_dir = "logs/2qubits/reward={}/traj_{}_episodes_{}_hid_{}_MaskTraject_entropy_{}".format(
            reward_func, batch_size, num_episodes, hidden_dims, entropy_reg)
    agent = PGAgent(env, policy, log_dir=log_dir)
    f = open(os.path.join(log_dir, "seed_{:d}.foo".format(seed)), "w")
    f.close()

    batch_mode = False
    initial_states = None
    agent.train(num_episodes, steps, learning_rate, lr_decay, clip_grad, reg, entropy_reg,
                log_every, test_every, verbose, initial_states, batch_mode)

    num_test = 10
    agent.logger.setLogTxtFilename("final_test.txt")
    agent.log_test_accuracy(num_test=num_test, steps=steps)
    agent.log_test_accuracy(num_test=num_test, steps=steps+1)
    agent.log_test_accuracy(num_test=num_test, steps=steps+2)


if __name__ == "__main__":
    seed = 0
    tic = time.time()
    # for reward_func in ["-1+1", "log_entropy", "log_log_1-entropy"]:
    # for reward_func in ["log_entropy"]:
    reward_func = "log_entropy"
    # main(reward_func, seed, batch_size=64, entropy_reg=0.0)
    # main(reward_func, seed, batch_size=64, entropy_reg=0.3)
    # main(reward_func, seed, batch_size=256, entropy_reg=0.0)
    # main(reward_func, seed, batch_size=256, entropy_reg=0.5)

    main(reward_func, seed)
    toc = time.time()
    print("Training took {:.3f} seconds".format(toc-tic))

#