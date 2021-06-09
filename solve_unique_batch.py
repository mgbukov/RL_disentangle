import time

import numpy as np

from src.envs.batched_env import QubitsEnvironment
from src.policies.fcnn_policy import FCNNPolicy
from src.agents.pg_agent import PGAgent
from src.utils.util_funcs import fix_random_seeds
from src.utils.logger import Logger


def main():
    log = Logger("logs/2qubits/unique_batch")
    log.setLogTxtFilename("2qubits_3stesps_adam.txt")
    log.verboseTxtLogging(True)
    num_trials = 100
    solved = 0
    max_ent = 0.0

    tic = time.time()
    for i in range(num_trials):
        seed = i ** 3 - 17 * i ** 2  + 12345
        fix_random_seeds(seed)
        log.logTxt("\nstep: {}".format(i+1))
        log.logTxt("seed: {}".format(seed))

        # Create the environment.
        num_qubits = 2
        epsi = 1e-3
        batch_size = 8
        env = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=batch_size)

        # Initialize the policy.
        input_size = 2 ** (num_qubits + 1)
        hidden_dims = [64]
        output_size = env.num_actions
        dropout_rate = 0.0
        policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate)

        # Train the policy-gradient agent.
        num_episodes = 100
        steps = 3
        learning_rate = 1e-2
        lr_decay = 1.0#np.power(0.1, 1.0/num_episodes)
        reg = 0.0
        log_every = 10
        verbose = True
        log_dir = "logs/2qubits/unique_batch/tmp"
        agent = PGAgent(env, policy, log_dir)

        batch_mode = False
        agent.env.set_random_state(batch_mode)
        initial_states = agent.env._state.copy()
        agent.train(num_episodes, steps, learning_rate, lr_decay, reg, log_every,
                    verbose, initial_states, batch_mode)

        with open(log_dir + "/train_history.txt", "r") as f:
            for line in f.readlines():
                log.logTxt(line.strip('\n'))

        agent.env.state = initial_states
        states, actions, rewards, done = agent.rollout(steps, greedy=True)
        solved += sum(done[:, -1]).item()
        entropies = agent.env.entropy()
        log.logTxt("Testing with greedy policy:")
        log.logTxt("  Start entropy: {}".format(agent.env.Entropy(initial_states, agent.env.basis)))
        log.logTxt("  Mean final entropy: {:.6f}".format(entropies.mean()))
        log.logTxt("  Max final entropy: {:.6f}".format(entropies.max()))
        log.logTxt("  Solved trajectories: {} / {}".format(sum(done[:, -1]), batch_size))
        if max_ent < entropies.max():
            max_ent = entropies.max()

        agent.env.state = initial_states
        result = agent.env.compute_best_path(steps, batch_mode)
        log.logTxt("Exaushtive search:")
        log.logTxt("  Minimum entropy: {}".format(agent.env.entropy()))
        log.logTxt("  Max Min Entropy: {}".format(max(agent.env.entropy())))
        log.logTxt("  Best actions: {}".format(result))
        
        log.logTxt("\n####################################\n")

    toc = time.time()
    log.logTxt("Training took {:.3f} seconds".format(toc-tic))
    log.logTxt("Solved trajectories: {} / {}".format(solved, num_trials * batch_size))
    log.logTxt("Max entropy at final step: {:.5f}".format(max_ent))


if __name__ == "__main__":
    main()

#