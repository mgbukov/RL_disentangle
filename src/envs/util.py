import numpy as np
from itertools import permutations

#----------------------------------- System variables -----------------------------------#
def _generate_permutation_maps(L):
    qubits_permutations = np.zeros((L, L, L), dtype=np.int32)
    qubits_inverse_permutations = np.zeros_like(qubits_permutations)
    for q0, q1 in permutations(range(L), 2):
        sysA = [q0, q1]
        sysB = [q for q in range(L) if q not in sysA]
        P = sysA + sysB
        qubits_permutations[q0, q1] = np.array(P, dtype=np.int32)
        qubits_inverse_permutations[q0, q1] = np.argsort(P).astype(np.int32)
    return qubits_permutations, qubits_inverse_permutations

_QUBITS = {}
_QSYSTEMS_P = {}
_QSYSTEMS_INV_P = {}
for L in range(2, 15):
    _QUBITS[L] = (2,)*L
    _QSYSTEMS_P[L], _QSYSTEMS_INV_P[L] = _generate_permutation_maps(L)


#---------------------------------- Utility functions -----------------------------------#
def _random_pure_state(L):
    # Sample from Gaussian distribution, as it gives uniformly distributed
    # points in L dimensional unit sphere
    psi = np.random.randn(2 ** L) + 1j * np.random.randn(2 ** L)
    psi /= np.linalg.norm(psi)
    return psi.astype(np.complex64)

def _random_batch(L, batch_size=1):
    states = np.random.randn(batch_size, 2 ** L) + 1j * np.random.randn(batch_size, 2 ** L)
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return states.astype(np.complex64)

def _transpose_batch_inplace(batch, qubits_indices, L, inverse=False):
    P = _QSYSTEMS_INV_P[L] if inverse else _QSYSTEMS_P[L]
    for i, (q0, q1) in enumerate(qubits_indices):
        batch[i] = np.transpose(batch[i], P[q0, q1])

def _calculate_q0_q1_entropy_from_rhos(rhos):
    a = rhos[:, 0] + rhos[:, 1]  # rho_{q0-1}
    b = rhos[:, 2] + rhos[:, 3]  # rho_{q0-2}
    c = rhos[:, 0] + rhos[:, 2]  # rho_{q1-1}
    d = rhos[:, 1] + rhos[:, 3]  # rho_{q1-2}
    Sent_q0 = -a*np.log(a + np.finfo(a.dtype).eps) - b*np.log(b + np.finfo(b.dtype).eps)
    Sent_q1 = -c*np.log(c + np.finfo(c.dtype).eps) - d*np.log(d + np.finfo(d.dtype).eps)
    return Sent_q0, Sent_q1

def _phase_norm(states):
    """Normalizes the relative phase shift between different qubits in one state."""
    B = states.shape[0]
    L = states.ndim - 1
    first = states.reshape(B, -1)[:, 0]
    phi = np.angle(first)
    z = np.cos(phi) - 1j * np.sin(phi)
    return states * z.reshape((B,) + (1,) * L)

def _ent_entropy(states, subsys_A):
    """Returns the entanglement entropy for every state in the batch w.r.t. `subsys_A`.

    Args:
        states (np.array): A numpy array of shape (b, 2,2,...,2), giving the states in the
            batch.
        subsys_A (list[int]): A list of ints specifying the indices of the qubits to be
            considered as a subsystem. The subsystem is the same for every state in the
            batch. If None, defaults to half of the system.

    Returns:
        entropies (np.Array): A numpy array of shape (b,), giving the entropy of each
            state in the batch.
    """
    L = states.ndim - 1
    subsys_B = [i for i in range(L) if i not in subsys_A]
    system = subsys_A + subsys_B
    subsys_A_size = len(subsys_A)
    subsys_B_size = L - subsys_A_size
    states = np.transpose(states, (0,) + tuple(t + 1 for t in system))
    states = states.reshape((-1, 2 ** subsys_A_size, 2 ** subsys_B_size))
    lmbda = np.linalg.svd(states, full_matrices=False, compute_uv=False)
    lmbda += np.finfo(lmbda.dtype).eps # shift lmbda to be positive within machine precision
    return -2.0 / subsys_A_size * np.einsum('ai, ai->a', lmbda ** 2, np.log(lmbda))

def _entropy(states):
    """For each state in the batch compute the entanglement entropies by considering each
    qubit as a subsystem.

    Args:
        states (np.array): A numpy array of shape (b, 2,2,...,2), giving the states in the
            batch.

    Returns:
        entropies (np.array): A numpy array of shape (b, L), giving single-qubit entropies.
    """
    L = states.ndim - 1
    entropies = [_ent_entropy(states, [i]) for i in range(L)]
    return np.stack(entropies).T

#