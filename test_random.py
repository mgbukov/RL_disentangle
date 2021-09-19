import time

from src.envs.rdm_environment import QubitsEnvironment
from src.policies.fcnn_policy import FCNNPolicy
from src.agents.pg_agent import PGAgent
from src.utils.util_funcs import fix_random_seeds


def main():
    seed = 0
    fix_random_seeds(seed)

    # Create the environment.
    num_qubits = 4
    epsi = 1e-3
    batch_size = 128
    env = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=batch_size)

    # Initialize the policy.
    input_size = 2 ** (num_qubits + 1)
    hidden_dims = [256, 256]
    output_size = env.num_actions
    dropout_rate = 0.25
    policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate)

    steps = 20
    log_dir = "logs/4qubits/random"
    agent = PGAgent(env, policy, log_dir=log_dir)

    num_test = 100
    agent.logger.setLogTxtFilename("random_test.txt")
    agent.logger.verboseTxtLogging(True)
    agent.log_test_accuracy(num_test, steps)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print("Training took {:.3f} seconds".format(toc-tic))

#