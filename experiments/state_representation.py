""" Exploration of state representations as inputs to neural networks. """
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('..')
from src.envs.rdm_environment import QubitsEnvironment
from src.agents.expert import SearchExpert

OUTPUT_DIR = '../data/state_representations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(16)

L = 4

def pack_complex_state(psi):
    m = len(psi) // 2
    return psi[:m] + 1j * psi[m:]

def make_random_state(num_qubits):
    a = np.random.uniform(size=2**num_qubits)
    b = 1j * np.random.uniform(size=2**num_qubits)
    return ((a + b) / np.linalg.norm(a + b)).astype(np.complex64)


def make_semientagled_state(num_qubits, i):
    """
    Returns quantum system of `num_qubits` qubits in which qubits [0, ..., i-1]
    are in product state while qubits [i, ..., `num_qubits`) are entangled.
    """
    assert i >= 0
    product = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
    for _ in range(i):
        q = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
        product = np.kron(product, q)
    rsz = 2 ** (num_qubits - i - 1)
    if rsz == 1:
        return (product / np.linalg.norm(product)).astype(np.complex64)
    assert rsz >= 2 
    entangled = np.random.uniform(size=rsz) + 1j * np.random.uniform(size=rsz)
    state = np.kron(product, entangled)
    state /= np.linalg.norm(state)
    return state.astype(np.complex64)


def make_product_state(num_qubits):
    psi = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
    for _ in range(num_qubits-1):
        q = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
        psi = np.kron(psi, q)
    return (psi / np.linalg.norm(psi)).astype(np.complex64)


def plot_qubits_2d(psi, filename='sample'):
    qmap = psi.reshape(-1, 1) / psi.reshape(1, -1)
    qmap = np.tril(qmap)
    x = np.abs(qmap.real)
    y = np.abs(qmap.imag)
    x = x / np.max(x)
    y = y / np.max(y)
    z = np.zeros(qmap.shape, dtype=np.float32)
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(np.dstack([x,y,z]))
    axs.set_aspect(1.0)
    fig.savefig(filename + '.png', dpi=120)
    plt.close(fig)


for L in range(3, 11):
    for i in range(L):
        psi = make_semientagled_state(L, i)
        plot_qubits_2d(psi, filename=os.path.join(OUTPUT_DIR, f'{L}q-disent-{i}'))
    psi = make_random_state(L) 
    plot_qubits_2d(psi, filename=os.path.join(OUTPUT_DIR, f'{L}q-entang'))
    #


env = QubitsEnvironment(5, epsi=1e-3, batch_size=1)
search_expert = SearchExpert(env)
env.set_random_states()
states, actions = search_expert.rollout(env.states[0])
for i, s in enumerate(states):
    s = pack_complex_state(s)
    plot_qubits_2d(s, filename=os.path.join(OUTPUT_DIR, f'5q-rollout-{i}'))

env = QubitsEnvironment(4, epsi=1e-3, batch_size=1)
search_expert = SearchExpert(env)
env.set_random_states()
states, actions = search_expert.rollout(env.states[0])
for i, s in enumerate(states):
    s = pack_complex_state(s)
    plot_qubits_2d(s, filename=os.path.join(OUTPUT_DIR, f'4q-rollout-{i}'))