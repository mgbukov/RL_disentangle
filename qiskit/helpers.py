""" Helper functions for disentanglement of 4-qubit systems."""
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from itertools import cycle, permutations, combinations
from typing import Sequence, Tuple, Literal

# Find the the absolute path to project directory
dirname = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.realpath(os.path.join(dirname, os.path.pardir))
sys.path.append(project_root)


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
ACTION_SET_4Q_FULL = list(permutations(range(4), 2))
ACTION_SET_4Q = [x for x in ACTION_SET_4Q_FULL if x[0] < x[1]]
ACTION_SET_5Q = list(combinations(range(5), 2))
ACTION_SET_6Q = list(combinations(range(6), 2))


def get_action_index_from_ij(rdms, i, j):
    if len(rdms) == 6:
        return ACTION_SET_4Q.index((i,j))
    elif len(rdms) == 10:
        return ACTION_SET_5Q.index((i,j))
    elif len(rdms) == 15:
        return ACTION_SET_6Q.index((i,j))
    else:
        raise ValueError('Actions are defined only for 4,5,6 qubit systems. ' \
                         'Expected `rdms.shape[0]` to be one of (6, 10, 15).')

def get_ij_from_action_index(a, L):
    if L == 4:
        return ACTION_SET_4Q[a]
    elif L == 5:
        return ACTION_SET_5Q[a]
    elif L == 6:
        return ACTION_SET_6Q[a]
    else:
        raise ValueError('Actions are defined only for 4,5,6 qubit systems. ' \
                         'Expected `L` to be one of (4, 5, 6).')

def load_policy(path):
    agent = torch.load(path, map_location='cpu')
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()
    return agent.policy_network

@torch.no_grad()
def eval_policy(inputs, policy):
    # Add batch dimension
    x = torch.from_numpy(inputs[None, :, :])
    logits = policy(x)[0]
    distr = torch.distributions.Categorical(logits=logits)
    action = torch.argmax(distr.probs).item()
    return action


# Load all policies in global constants
# -----------------------------------------------------------------------------
#   Policy with permutation equivariant network
PE_POLICY = torch.jit.load(os.path.join(dirname, "pe-policy-2.0.pts")).eval()
#   Policty with transformer network
TRANSFORMER_POLICY = load_policy(
    os.path.join(
        project_root,
        "logs/4q_pGen_0.9_attnHeads_2_tLayers_2_ppoBatch_512_entReg_0.1_embed_128_mlp_256/agent.pt"
    )
)
#   Policy with transformer network + constrain that preserves the order
#   of entanglements, i.e if S_i < S_j then S_i' < S_j' and vice versa, where
#   S are the entanglements before applying action, S' after action.
ORDERED_POLICY_4Q = load_policy(
    os.path.join(
        project_root, "logs/4q_10000_iters_haar_unif2_1024envs/agent.pt"
))

ORDERED_POLICY_5Q = load_policy(
    os.path.join(
        project_root, "logs/5q_20000_iters_haar_unif2_128envs/agent.pt"
))

ORDERED_POLICY_6Q = load_policy(
    os.path.join(
        project_root, "logs/6q_4000iters_haar_unif3_512envs_seed7_3rd/agent.pt"
))


def get_action_4q(
        rdms :Sequence[np.ndarray],
        policy :Literal['universal', 'equivariant', \
                        'transformer', 'ordered'] = 'universal') \
                -> Tuple[np.ndarray, int, int]:
    """
    Returns gate and indices of qubits (starting from 0) on which to apply it.
    The returned gate is multiplied with pre-swap gate if S(q_i) < S(q_j).

    Parameters:
    -----------
    rdms6: Sequence[numpy.ndarray], dtype=np.complex64
        Sequence of reduced density matrices in order:
            rho_01, rho_02, rho_03, rho_12, rho_13, rho_23, ..., rho_45
    policy: str, default="universal"
        "universal" uses a predefined circuit, "equivariant" uses equivariant
        policy from trained RL agent, "transformer" uses transformer
        architecture, "ordered" uses transformer and requires that S(q_i) > S(q_j)

    Returns: Sequence[numpy.ndarray, int, int]
        Gate U_{ij} to be applied, i, j
    """
    assert rdms.shape[0] in (6, 10, 15)
    L = {6:4, 10:5, 15:6}[rdms.shape[0]]    # index dictionary with num of RDMs

    if policy == 'universal':
        assert rdms.shape[0] == 6
        i, j = next(UNIVERSAL_CIRCUIT)
    elif policy == 'equivariant':
        assert rdms.shape[0] == 6
        x = _prepare_all_rdms_input(rdms)
        a = eval_policy(x, PE_POLICY)
        i, j = ACTION_SET_4Q_FULL[a]
    elif policy == 'transformer':
        x = _prepare_reduced_real_input(rdms)
        a = eval_policy(x, TRANSFORMER_POLICY)
        i, j = get_ij_from_action_index(a, L)
    elif policy == 'ordered':
        x = _prepare_reduced_real_input(rdms)
        if L == 4:
            a = eval_policy(x, ORDERED_POLICY_4Q)
        elif L == 5:
            a = eval_policy(x, ORDERED_POLICY_5Q)
        elif L == 6:
            a = eval_policy(x, ORDERED_POLICY_6Q)
        i, j = get_ij_from_action_index(a, L)
    else:
        raise ValueError("`policy` must be one of ('universal', " \
                         "'equivariant', 'transformer', 'ordered').")
    # Gate + preswap + postswap
    use_swaps = (policy == 'ordered')
    U = get_U(rdms, i, j, apply_preswap=use_swaps, apply_postswap=use_swaps)
    return U, i, j


def get_U(rdms, i, j, apply_preswap=False, apply_postswap=False):
    rdm = rdms[get_action_index_from_ij(rdms, i, j)].copy()
    # Rounding the near 0 elements in `rdm` matrix solved problems with
    # numerical precision in RL training. But should we clip them here?
    if apply_preswap:
        S = get_preswap_gate(rdms, i, j)
        rdm = S @ rdm @ S.conj().T
    rdm += np.finfo(rdm.dtype).eps * np.diag([0, 1, 2, 4]).astype(rdm.dtype)
    _, U = np.linalg.eigh(rdm)
    max_col = np.abs(U).argmax(axis=0)
    for k in range(4):
        U[:, k] *= np.exp(-1j * np.angle(U[max_col[k], k]))

    U = U.conj().T
    if apply_preswap:
        # Swap qubits, apply gate and then swap back again
        U = S @ U @ S
    if apply_postswap:
        P = get_postswap_gate(rdms, i, j)
        U = P @ U
    return U


def get_preswap_gate(rdms :Sequence[np.ndarray], i :int, j :int):
    """
    Returns a swap gate for qubits (i,j). The swap gate's purpose is to
    enforce the relation S(q_i) > S(q_j).

    Parameters:
    -----------
    rdms6: Sequence[numpy.ndarray], dtype=np.complex64
        Sequence of reduced density matrices in order:
            rho_01, rho_02, rho_03, rho_12, rho_13, rho_23
    i: int
        Index of first qubit
    j: int
        Index of second qubit
    """
    rdm_ij = rdms[get_action_index_from_ij(rdms, i, j)]
    S_i, S_j = _get_sqe_rdm(rdm_ij)
    # Make an Identity or Swap gate, depending on `S_i`, `S_j`
    I = np.eye(4, 4, dtype=np.complex64)
    P = I.copy()
    P[[1,2]] = P[[2,1]]
    if S_i > S_j + np.finfo(S_i.dtype).eps:
        return I
    return P


def get_postswap_gate(rdms :Sequence[np.ndarray], i :int, j :int):
    _dtype = rdms.dtype
    _eps = np.finfo(rdms.dtype).eps
    I = np.eye(4, 4, dtype=np.complex64)
    P = I.copy()
    P[[1, 2]] = P[[2, 1]]

    # Calculate current entanglements
    rdm_ij = rdms[get_action_index_from_ij(rdms, i, j)]
    S_i, S_j = _get_sqe_rdm(rdm_ij)

    # Calculate U
    preswap_gate = I if S_i > S_j + _eps else P
    rdm_temp = preswap_gate @ rdm_ij @ preswap_gate.conj().T
    rdm_temp += np.finfo(rdm_temp.dtype).eps * np.diag([0, 1, 2, 4])
    _, U = np.linalg.eigh(rdm_temp)
    max_col = np.abs(U).argmax(axis=0)
    for k in range(4):
        U[:, k] *= np.exp(-1j * np.angle(U[max_col[k], k]))

    U = U.conj().T
    # Swap qubits, apply gate and then swap back again
    U =  preswap_gate @ U @ preswap_gate

    # Calculate single qubit entanglements after action
    rdm_ij_next = U @ rdm_ij @ U.conj().T
    S_i_new, S_j_new = _get_sqe_rdm(rdm_ij_next)
    flag = (S_i >= S_j + _eps) ^ (S_i_new >= S_j_new + _eps)
    return P if flag else I


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
    L = int(np.log2(state.size))
    if L not in (4, 5, 6):
        raise ValueError("Expected either 4, 5 or 6 qubit systems state vector.")

    tshape = (2,) * L       # Tensor-like shape
    psi = state.reshape(tshape)
    P = (i, j) + tuple(k for k in range(L) if k not in (i, j))
    psi = np.transpose(psi, P)

    phi = U @ psi.reshape(4, -1)
    phi = np.transpose(phi.reshape(tshape), np.argsort(P))

    ent = get_entanglements(phi)
    done = np.all(ent < 1e-3)
    reward = 100 if done else -1

    return phi.ravel(), ent, reward


def get_entanglements(state :np.ndarray) -> np.ndarray:
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


def _prepare_all_rdms_input(rdms: Sequence[np.ndarray]) -> np.ndarray:
    """
    Construct all 12 RDMs from list of 6 RDMs.

    Parameters:
    -----------
    rdms: Sequence[numpy.ndarray], dtype=np.complex64
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
    for k in ACTION_SET_4Q_FULL:
        result.append(rdms_dict[k])
    return np.array(result).reshape(12, 16)


def _prepare_reduced_real_input(rdms: Sequence[np.ndarray]) -> np.ndarray:
    """
    Construct inputs for reduced action sets.

    Parameters:
    -----------
    rdms: Sequence[numpy.ndarray], dtype=np.complex64
        Sequence of reduced density matrices in order:
            rho_01, rho_02, rho_03, rho_12, ..., rho_45

    Returns: numpy.ndarray
        Array with (6 | 10 | 15) inputs
    """
    rdms = np.asarray(list(rdms))
    if rdms.shape[0] not in (6, 10, 15):
        raise ValueError("Expected sequence of (6 | 10 | 15) RDMs of size 4x4.")

    L = {6:4, 10:5, 15:6}[rdms.shape[0]]
    if L == 4:
        indices = ACTION_SET_4Q
    elif L == 5:
        indices = ACTION_SET_5Q
    elif L == 6:
        indices = ACTION_SET_6Q

    rdms_dict = {}
    for idx, rdm in zip(indices, rdms):
        flipped = rdm[[0, 2, 1, 3], :][:, [0, 2, 1, 3]].copy()
        rdms_dict[idx] = 0.5 * (rdm + flipped)

    result = []
    for k in indices:
        result.append(rdms_dict[k])
    x = np.array(result).reshape(rdms.shape[0], 16)
    return np.hstack([x.real, x.imag])


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


def _get_sqe_rdm(rdm):
    """Calculates single qubit entanglements from 2-qubit RDM."""
    # Calculate single qubit RDMs
    rdm_i = np.trace(rdm.reshape(2,2,2,2), axis1=1, axis2=3)
    rdm_j = np.trace(rdm.reshape(2,2,2,2), axis1=0, axis2=2)
    # Calculate single qubit entanglements
    lambdas = np.linalg.svd(rdm_i, compute_uv=False, full_matrices=False)
    S_i = -np.sum(lambdas * np.log(lambdas + np.finfo(lambdas.dtype).eps), axis=-1)
    lambdas = np.linalg.svd(rdm_j, full_matrices=False, compute_uv=False)
    S_j = -np.sum(lambdas * np.log(lambdas + np.finfo(lambdas.dtype).eps), axis=-1)
    return S_i, S_j
