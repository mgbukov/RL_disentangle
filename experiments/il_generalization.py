"""
Tests the performance of Imitation Learning agent over pertubations of
solved states in the training set (generalization strength).
"""
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import sys

sys.path.append(os.path.abspath('..'))
from src.envs.rdm_environment import QubitsEnvironment
from src.agents.il_agent import ILAgent
from src.policies.fcnn_policy import FCNNPolicy
from src.envs.util import _random_pure_state
from src.agents.expert import SearchExpert


L = 5
B = 32
# SHAPE = (B,) + (2,) * L
DUMMY_SHAPE = (1,) + (2,) * L
np.random.seed(3)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True



# env = QubitsEnvironment(num_qubits=L, epsi=1e-3, batch_size=B)
dummyenv = QubitsEnvironment(num_qubits=L, epsi=1e-3, batch_size=1)

# # Load IL policy
input_size = 2 ** (L + 1)
hidden_dims = [4096, 4096, 512]
output_size = dummyenv.num_actions
policy = FCNNPolicy(input_size, hidden_dims, output_size, dropout_rate=0.0)
POLICY = policy.load('../data/il-policy.bin')

# Load train dataset
with open('../data/100000_episodes.pickle', mode='rb') as f:
    dataset = pickle.load(f)

# agent = ILAgent(env, policy)
# dummyagent = ILAgent(dummyenv, policy)
# expert = SearchExpert(dummyenv, 100)
# delta = np.linspace(0, 0.1, B)

# solvable = []
# solveline = []
# matches = []
# count = 0


def real_to_complex(state):
    s = state.reshape(2, -1)
    return s[0] + 1j * s[1]

def fidelity(psi, phi):
    return np.abs(psi.conj().T @ phi) ** 2

def is_disentangled(psi):
    return dummyenv.Disentangled(psi.reshape(DUMMY_SHAPE), epsi=1e-3)[0]


def pertube(state):
    print('called pertube()')
    P = 20  # how many random pertubations
    R = 40  # resolution
    env = QubitsEnvironment(num_qubits=L, epsi=1e-3, batch_size=B)
    dummyenv = QubitsEnvironment(num_qubits=L, epsi=1e-3, batch_size=1)
    agent = ILAgent(env, POLICY)
    dummyagent = ILAgent(dummyenv, POLICY)
    delta = np.linspace(0, 0.1, R)

    dummyagent.env.states = state.reshape((1,) + (2,) * L)
    states0, actions0, _, _ = dummyagent.rollout(30, greedy=True)
    states0 = states0.numpy()[0]
    states0 = np.array([real_to_complex(psi) for psi in states0])
    states0[0] = state
    actions0 = actions0.numpy()
    done = [is_disentangled(psi) for psi in states0]

    matches  = []  # (P, R)
    nsolves  = []  # (P, R)
    overlaps = []  # (P, R)
    if (not done[0] and done[-1]):
        # Generate `P` random states
        for _ in range(P):
            psi = _random_pure_state(L)
            batch = state + np.outer(delta, psi)
            batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            overlap = fidelity(batch.T, state)
            overlaps.append(overlap)
            env.states = batch.reshape((R,) + (2,) * L).copy()
            states, actions, rewards, masks = agent.rollout(30, greedy=True)
            states = states.numpy()
            actions = actions.numpy()
            match = np.cumprod(actions == actions0, axis=1)
            last = states[:, -1, :]
            ns = [is_disentangled(real_to_complex(psi)) for psi in last]
            nsolves.append(ns)
            matches.append(match.sum(axis=1))
        # (P, R)
        return np.array(matches), np.array(nsolves), np.array(overlaps)
    return np.array([]), np.array([]), np.array([])


for i in range(1000):
    state = dataset['states'][i]
    state = real_to_complex(state)
    matches, nsolves, overlaps = pertube(state)
    if len(matches) > 1:
        break


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('IL Generalization (Single State, Many Pertubations)')
ax.set_xlabel('delta')
ax.set_ylabel('steps in trajectories that match')
xs = np.linspace(0, 0.1, matches.shape[1])
ys = np.mean(matches, axis=0)
err = np.std(matches, axis=0)
ax.errorbar(xs, ys, yerr=err, marker='|', linestyle='--', markerfacecolor='k')
ax.plot(xs, np.max(matches, axis=0), color='r', ls='--')
ax.plot(xs, np.min(matches, axis=0), color='r', ls='--')
fig.savefig('matches_single_state.png', dpi=160)


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('IL Generalization (Single State, Many Pertubations)')
ax.set_xlabel('delta')
ax.set_ylabel('solved states')
xs = np.linspace(0, 0.1, nsolves.shape[1])
ys = np.mean(nsolves, axis=0)
err = np.std(nsolves, axis=0)
ax.errorbar(xs, ys, yerr=err, marker='|', linestyle='--', markerfacecolor='k')
ax.plot(xs, np.max(nsolves, axis=0), color='r', ls='--')
ax.plot(xs, np.min(nsolves, axis=0), color='r', ls='--')
fig.savefig('nsolves_single_state.png', dpi=160)


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('IL Generalization (Single State, Many Pertubations)')
ax.set_xlabel(r'$|\langle \psi_0 | \psi_0 + \delta \rangle | ^2$')
ax.set_ylabel('solved states')
xs = overlaps.ravel()
ys = matches.ravel()
ax.scatter(xs, ys, s=10)
fig.savefig('matches_overlap_single_state.png', dpi=160)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('IL Generalization (Single State, Many Pertubations)')
ax.set_xlabel(r'$|\langle \psi_0 | \psi_0 + \delta \rangle | ^2$')
ax.set_ylabel('solved states')
xs = overlaps.ravel()
ys = nsolves.ravel()
ax.scatter(xs, ys, s=10)
fig.savefig('matches_overlap_single_state.png', dpi=160)


# for i in range(100000):
#     state = dataset['states'][i]
#     state = real_to_complex(state)

#     # Test if the state is solved by the agent
#     dummyagent.env.states = state.reshape((1,) + (2,) * L)

#     states0, actions0, _, _ = dummyagent.rollout(30, greedy=True)
#     states0 = states0.numpy()[0]
#     states0 = np.array([real_to_complex(psi) for psi in states0])
#     states0[0] = state
#     actions0 = actions0.numpy()
#     done = [is_disentangled(psi) for psi in states0]
#     # print(done)
#     if (not done[0] and done[-1]):
#         solvable.append(states0[0].copy())
#         psi = _random_pure_state(L)
#         batch = state + np.outer(delta, psi)
#         batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)
#         # batch[0] = state
#         # trajectories = []
#         # for psi in batch:
#         #     dummyenv.states = psi.reshape(DUMMY_SHAPE)
#         #     _, traj = expert.rollout(psi.reshape((2,) * L))
#         #     trajectories.append(traj)
#         env.states = batch.reshape((B,) + (2,) * L).copy()
#         env._states[0] = state.reshape((2,) * L).copy()
#         assert np.all(env._states[0].ravel() == state.ravel())
#         states, actions, rewards, masks = agent.rollout(30, greedy=True)
#         states = states.numpy()
#         actions = actions.numpy()
#         match = np.cumprod(actions == actions0, axis=1)
#         last = states[:, -1, :]
#         nsolved = [is_disentangled(real_to_complex(psi)) for psi in last]
#         assert np.all(real_to_complex(states[0, 0]) == state)
#         if not nsolved[0]:
#             print('Assertion error')
#         solveline.append(nsolved)
#         match = match.sum(axis=1)
#         matches.append(match)
#         count += 1
#         print(count)
#         if count >= 100:
#             break





# fig, ax = plt.subplots(figsize=(12, 8))
# ax.set_title('IL generalization')
# ax.set_xlabel('delta')
# ax.set_ylabel('steps in trajectories that match')
# ax.plot(delta, np.mean(matches, axis=0))
# fig.savefig('matches.png', dpi=160)


# fig, ax = plt.subplots(figsize=(12, 8))
# ax.set_title('IL generalization')
# ax.set_xlabel('delta')
# ax.set_ylabel('solved states')
# ax.plot(delta, np.mean(solveline, axis=0))
# fig.savefig('solveline.png', dpi=160)