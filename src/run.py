import numpy as np
import time
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
    env = QubitsEnvironment(num_qubits, epsi, num_trajectories)

    # Initialize the policy.
    num_episodes = 10
    steps = 10
    learning_rate = 1e-5
    reg = 1e-5
    dropout_rate = 0.3
    policy = FCNNPolicy(2 ** (num_qubits + 1), [256, 256], env.num_actions, dropout_rate)

    # Train the policy-gradient agent.
    policy_save_path = "fcnn_policy.bin"
    agent = PGAgent(env, policy)
    agent.train(num_episodes, steps, learning_rate, reg)

    # Test the model.
    num_test = 10
    solved = 0
    for i in range(num_test):
        agent.env.set_random_state()
        states, actions, rewards, done = agent.rollout(10, greedy=True)
        solved += sum(done[:,-1])
    solved = solved.item()
    print("Solved states: %d / %d = %.3f %%" % (solved, num_test*num_trajectories,
        solved/(num_test*num_trajectories)*100))


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print("Training took %.3f seconds" % (toc-tic))