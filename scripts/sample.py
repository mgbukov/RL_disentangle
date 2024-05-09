"""Script that shows a sample usage of the RL agents."""
import os
import sys

import numpy as np
import torch

file_path = os.path.split(os.path.abspath(__file__))[0]
project_dir = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.append(project_dir)
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state


PATH_4Q_AGENT = os.path.join(project_dir, "agents/4q-agent.pt")
PATH_5Q_AGENT = os.path.join(project_dir, "agents/5q-agent.pt")
PATH_6Q_AGENT = os.path.join(project_dir, "agents/6q-agent.pt")

# Load agents
AGENT_4Q = torch.load(PATH_4Q_AGENT, map_location='cpu')
AGENT_5Q = torch.load(PATH_5Q_AGENT, map_location='cpu')
AGENT_6Q = torch.load(PATH_6Q_AGENT, map_location='cpu')

for agent in (AGENT_4Q, AGENT_5Q, AGENT_6Q):
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()


def get_action(state):
    """Returns action using RL agents for `state`."""

    # Initialize environment
    num_qubits = int(np.log2(state.size))
    shape = (2,) * num_qubits
    env = QuantumEnv(num_qubits, 1, obs_fn='rdm_2q_mean_real')
    env.reset()
    env.simulator.states = np.expand_dims(state.reshape(shape), 0)

    # Choose agent
    agent = None
    if num_qubits == 4:
        agent = AGENT_4Q
    elif num_qubits == 5:
        agent = AGENT_5Q
    elif num_qubits == 6:
        agent = AGENT_6Q
    else:
        raise ValueError(f"No agent available for {num_qubits}-qubits system.")

    # Do inference
    observation = torch.from_numpy(env.obs_fn(env.simulator.states))
    probs = agent.policy(observation).probs[0].cpu().numpy()
    a = np.argmax(probs)

    return env.simulator.actions[a]


def get_disentangling_trajectory(state, max_steps=100):
    """
    Returns a list of [a_0, a_1, ..., a_n] actions that disentangle the given state
    and a boolean flag that indicates if solution is found - `True` if the state
    is disentangled after (n + 1) steps, `False` if the RL agent failed to
    disentangle the state.
    """

    # Initialize environment
    num_qubits = int(np.log2(state.size))
    shape = (2,) * num_qubits
    env = QuantumEnv(num_qubits, 1, obs_fn='rdm_2q_mean_real')
    env.reset()
    env.simulator.states = np.expand_dims(state.reshape(shape), 0)

    # Choose agent
    agent = None
    if num_qubits == 4:
        agent = AGENT_4Q
    elif num_qubits == 5:
        agent = AGENT_5Q
    elif num_qubits == 6:
        agent = AGENT_6Q
    else:
        raise ValueError(f"No agent available for {num_qubits}-qubits system.")
    
    trajectory = []
    success = False
    for _ in range(max_steps):
        observation = torch.from_numpy(env.obs_fn(env.simulator.states))
        probs = agent.policy(observation).probs[0].cpu().numpy()
        a = np.argmax(probs)
        trajectory.append(env.simulator.actions[a])
        o, r, t, tr, i = env.step([a], reset=False)
        if np.all(t):
            success = True
            break
    
    return trajectory, success


if __name__ == "__main__":

    # Generate 5-qubit quantum state
    state = random_quantum_state(5)

    # Print the first action
    print("\nQuantum state:\n", state.ravel())
    print("\nRL Action [get_action()]:\n", get_action(state), '\n')

    # Print the whole trajectory that disentangles the state
    traj, success = get_disentangling_trajectory(state)
    print("\nRL Trajectory [get_disentangling_trajectory()]:\n", traj[0], '\n')
