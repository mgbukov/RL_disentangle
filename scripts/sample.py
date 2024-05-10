"""Script that shows a sample usage of the RL agents."""
import os
import numpy as np
import torch

from context import *
from src.quantum_env import QuantumEnv
from src.quantum_state import random_quantum_state


# Load agents
AGENT_4Q = torch.load(os.path.join(project_dir, "agents/4q-agent.pt"))
AGENT_5Q = torch.load(os.path.join(project_dir, "agents/5q-agent.pt"))
AGENT_6Q = torch.load(os.path.join(project_dir, "agents/6q-agent.pt"))


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
    print("\nRL Trajectory [get_disentangling_trajectory()]:\n", traj, '\n')
