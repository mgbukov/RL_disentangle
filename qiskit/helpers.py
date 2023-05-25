""" Helper functions for disentanglement of 4-qubit systems."""
import numpy as np
import os
import torch
import torch.nn.functional as F
from itertools import cycle, permutations
from typing import Sequence, Tuple, Literal


# Circuit for 4 qubits
# NOTE
# This circuit is not cyclic!
# ---------------------------
# For example if gates are applied on qubits
# (2, 3), (0, 2), (1, 2), (2, 3), (0, 1) in that order
# one should not expect to disentangle a 4-qubit state in 5 steps !
# 
# To disentangle a system in 5 steps, one must apply exactly 5 gates to
# qubits indices from cycle starting from (0, 1) !
UNIVERSAL_CIRCUIT = cycle([(0, 1), (2, 3), (0, 2), (1, 2), (2, 3)])
# Action set on which RL agents are currently trained (may change in future)
ACTION_SET_FULL = list(permutations(range(4), 2))
ACTION_SET_REDUCED = [x for x in ACTION_SET_FULL if x[0] < x[1]]
# Path to TorchScript file with policy from RL agent.
PE_POLICY_PATH = os.path.join(os.path.dirname(__file__), 'pe-policy-2.0.pts')
TRANSFORMER_POLICY_PATH = os.path.join(os.path.dirname(__file__), 'tfm-policy-4q.torchscript')
# Load policy networks
PE_POLICY = torch.jit.load(PE_POLICY_PATH).eval()
TRANSFORMER_POLICY = torch.jit.load(TRANSFORMER_POLICY_PATH).eval()


@torch.no_grad()
def policy_function(inputs, policy):
    # Add batch dimension
    x = torch.from_numpy(inputs[None, :, :])
    logits = policy(x)[0]
    distr = torch.distributions.Categorical(logits=logits)
    action = torch.argmax(distr.probs).item()
    return action


def get_action_4q(
        rdms6 :Sequence[np.ndarray],
        policy :Literal['universal', 'equivariant'] = 'universal') -> Tuple[np.ndarray, int, int]:
    """
    Returns gate and indices of qubits (starting from 0) on which to apply it.

    Parameters:
    -----------
    rdms6: Sequence[numpy.ndarray]
        Sequence of reduced density matrices in order:
            rho_01, rho_02, rho_03, rho_12, rho_13, rho_23
    policy: str, default="universal"
        "universal" uses a predefined circuit, "equivariant" uses equivariant
        policy from trained RL agent. "transformer" uses equivariant transformer
        architecture.

    Returns: Sequence[numpy.ndarray, int, int]
        Gate U_{ij} to be applied, i, j
    """

    if policy == 'universal':
        i, j = next(UNIVERSAL_CIRCUIT)
        rdm = rdms6[ACTION_SET_REDUCED.index((i,j))]
    elif policy == 'equivariant':
        x = _prepare_all_rdms_input(rdms6)
        a = policy_function(x, PE_POLICY)
        i, j = ACTION_SET_FULL[a]
        rdm = x[ACTION_SET_FULL.index((i, j))].reshape(4, 4)
    elif policy == 'transformer':
        x = _prepare_reduced_real_input(rdms6)
        a = policy_function(x, TRANSFORMER_POLICY)
        i, j = ACTION_SET_REDUCED[a]
        rdm = rdms6[a]
    else:
        raise ValueError("`policy` must be one of (None, 'equivariant', 'transformer').")

    # QUESTIONABLE
    # Rounding the near 0 elements in `rdm` matrix solved problems with
    # numerical precision in RL training. But should we clip them here ?
    rdm[np.abs(rdm) < 1e-7] = 0.0
    _, U = np.linalg.eigh(rdm)
    phase = np.exp(-1j * np.angle(np.diagonal(U)))
    np.einsum('ij,j->ij', U, phase, out=U)
    U = U.conj().T
    print(U, i, j)
    return U, i, j


def _prepare_all_rdms_input(rdms: Sequence[np.ndarray]) -> np.ndarray:
    """
    Construct all 12 RDMs from list of 6 RDMs.

    Parameters:
    -----------
    rdms: Sequence[numpy.ndarray]
        Sequence of reduced density matrices in order:
            rho_01, rho_02, rho_03, rho_12, rho_13, rho_23

    Returns: numpy.ndarray
        Array with 12 RDMs in order:
            rho_01, rho_02, rho_03, rho_10, rho_12, rho_13, rho_20, rho_21, rho_23
    """
    rdms = np.asarray(list(rdms))
    if rdms.shape != (6, 4, 4):
        raise ValueError("Expected sequence of 6 RDMs of size 4x4.")

    indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rdms_dict = dict(zip(indices, rdms))
    for k in indices:
        rdm = rdms_dict[k]
        newkey = (k[1], k[0])
        flipped = rdm[[0, 2, 1, 3], :][:, [0, 2, 1, 3]].copy()
        rdms_dict[newkey] = flipped

    result = []
    for k in ACTION_SET_FULL:
        result.append(rdms_dict[k])
    return np.array(result).reshape(12, 16)


def _prepare_reduced_real_input(rdms: Sequence[np.ndarray]) -> np.ndarray:
    """
    Construct all 6 inputs for ACTION_SET_REDUCED.

    Parameters:
    -----------
    rdms: Sequence[numpy.ndarray]
        Sequence of reduced density matrices in order:
            rho_01, rho_02, rho_03, rho_12, rho_13, rho_23

    Returns: numpy.ndarray
        Array with 6 inputs
    """
    rdms = np.asarray(list(rdms))
    if rdms.shape != (6, 4, 4):
        raise ValueError("Expected sequence of 6 RDMs of size 4x4.")

    indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rdms_dict = {}
    for idx, rdm in zip(indices, rdms):
        flipped = rdm[[0, 2, 1, 3], :][:, [0, 2, 1, 3]].copy()
        rdms_dict[idx] = 0.5 * (rdm + flipped)

    result = []
    for k in ACTION_SET_REDUCED:
        result.append(rdms_dict[k])
    x = np.array(result).reshape(6, 16)
    return np.vstack([x.real, x.imag]).reshape(6, 32)


def peek_next_4q(state :np.ndarray, U :np.ndarray, i :int, j :int) -> Tuple[np.ndarray, int]:
    """
    Applies gate `U` on qubits `i` and `j` in `state`.

    Parameters:
    ----------
    state : numpy.ndarray
        Quantum state of 4 qubits.
    U : numpy.ndarray
        Unitary gate
    i : int
        Index of the first qubit on which U is applied
    j : int
        Index of the second qubit on which U is applied

    Returns: Tuple[numpy.ndarray, numpy.ndarray, int]
        The resulting next state, entanglement, and the RL reward.
    """
    state = np.asarray(state).ravel()
    if state.shape != (16,):
        raise ValueError("Expected a 4-qubit state vector with 16 elements.")

    psi = state.reshape(2,2,2,2)
    other = tuple(k for k in range(4) if k not in (i, j))
    P = (i, j) + other
    psi = np.transpose(psi, P)

    phi = U @ psi.reshape(4, -1)
    phi = np.transpose(phi.reshape(2,2,2,2), np.argsort(P))

    ent = _entanglement(phi)
    done = np.all(ent < 1e-3)
    reward = 100 if done else -1

    return phi.ravel(), ent, reward


def _qubit_entanglement(state :np.ndarray, i :int) -> float:
    """
    Returns the entanglement entropy of qubit `i` in `state`.

    Parameters:
    -----------
    state : numpy.ndarray
        Quantum state vector - shape == (2,2,...,2)
    i : int
        Index of the qubit for which entanglement is calculated and returned.

    Returns: float
        The entanglement of qubit `i` wtih rest of the system.
    """
    L = state.ndim
    subsys_A = [i]
    subsys_B = [j for j in range(L) if j != i]
    system = subsys_A + subsys_B

    psi = np.transpose(state, system).reshape(2, 2 ** (L - 1))
    lmbda = np.linalg.svd(psi, full_matrices=False, compute_uv=False)
    # shift lmbda to be positive within machine precision
    lmbda += np.finfo(lmbda.dtype).eps
    return -2.0 * np.sum((lmbda ** 2) * np.log(lmbda))


def _entanglement(state :np.ndarray) -> np.ndarray:
    """
    Compute the entanglement entropies of `state` by considering
    each qubit as a subsystem.

    Parameters:
    -----------
    states : np.array
        A numpy array of shape (2,2,...,2).

    Returns: np.ndarray
        A numpy array with single-qubit entropies for each qubit in `state`.
    """
    n = state.size
    if (n & (n - 1)):
        raise ValueError("Expected array with size == power of 2.")
    
    L = int(np.log2(n))
    state = state.reshape((2,) * L)
    entropies = np.array([_qubit_entanglement(state, i) for i in range(L)])
    return entropies
