import itertools
import numpy as np
import sys

sys.path.append("..")
from src.envs.rdm_environment import QubitsEnvironment
from src.envs.util import phase_norm
from src.infrastructure.util_funcs import fix_random_seeds


TEST_BATCHES = (1, 2, 16, 256, 1024)
TEST_QUBITS = (3, 4, 5, 6, 7, 8)


def pack_complex_state(psi):
    m = len(psi) // 2
    return psi[:m] + 1j * psi[m:]

def unpack_complex_state(psi):
    return np.hstack([psi.real, psi.imag])

def make_semientagled_state(num_qubits, i):
    """ Returns quantum system of `num_qubits` qubits in which qubits [0, ...,i-1]
    are in product state while qubits [i, ..., `num_qubits`) are entangled. """
    assert i >= 0
    product = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
    for _ in range(i):
        q = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
        product = np.kron(product, q)
    rsz = 2 ** (num_qubits - i - 1)
    assert rsz >= 2
    entangled = np.random.uniform(size=rsz) + 1j * np.random.uniform(size=rsz)
    state = np.kron(product, entangled)
    state /= np.linalg.norm(state)
    return state

def make_product_state(num_qubits):
    psi = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
    psi /= np.linalg.norm(psi)
    for _ in range(num_qubits-1):
        q = np.random.uniform(size=2) + 1j * np.random.uniform(size=2)
        q /= np.linalg.norm(q)
        psi = np.kron(psi, q)
    return psi / np.linalg.norm(psi)

def environment_generator(nqubits=None, bsizes=None, seed=14, epsi=1e-3):
    """ Python generator for environments parameterized by batch sizes and
    number of qubits. """
    if nqubits is None:
        nqubits = TEST_QUBITS
    elif isinstance(nqubits, int):
        nqubits = (nqubits,)
    elif isinstance(nqubits, tuple):
        pass
    else:
        raise ValueError('nqubits must be of type tuple or int')

    if bsizes is None:
        bsizes = TEST_BATCHES
    elif isinstance(bsizes, int):
        bsizes = (bsizes,)
    elif isinstance(bsizes, tuple):
        pass
    else:
        raise ValueError('batch_size must be of type tuple or int')
    for q, B in itertools.product(nqubits, bsizes):
        fix_random_seeds(seed)
        env = QubitsEnvironment(num_qubits=q, epsi=epsi, batch_size=B)
        env.set_random_states(copy=False)
        yield env


import matplotlib.pyplot as plt

def plot_qubits_2d(psi, filename='sample'):
    qmap = psi.reshape(-1, 1) / psi.reshape(1, -1)
    fig, axs = plt.subplots(1, 2)
    pmesh0 = axs[0].pcolormesh(qmap.real)
    pmesh1 = axs[1].pcolormesh(qmap.imag)
    axs[0].set_aspect(1.0)
    axs[1].set_aspect(1.0)
    plt.colorbar(pmesh0, ax=axs[0], use_gridspec=True, orientation='horizontal')
    plt.colorbar(pmesh1, ax=axs[1], use_gridspec=True, orientation='horizontal')
    fig.savefig(filename + '.png', dpi=120)
    plt.close(fig)

psi = make_product_state(4)
plot_qubits_2d(psi, filename='dissentangled')
phi = make_semientagled_state(4, 0)
plot_qubits_2d(phi, filename='entangled')