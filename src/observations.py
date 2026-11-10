from itertools import combinations, permutations
from typing import List, Optional, Tuple

import numpy as np
import torch



SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def vec(states: torch.Tensor, device: str|torch.device = "cpu"):
    N = states.shape[0]
    batch = states.reshape(N, -1)
    batch = torch.hstack([batch.real, batch.imag])
    return batch


def rdm2m(psi: torch.Tensor,
          rdms: Optional[torch.Tensor] = None,
          modified_qubits: Optional[List[Tuple[int,int]]] = None,
          device: str|torch.device = "cpu"):

    N = psi.shape[0]
    Q = psi.ndim - 1

    # --- Slow path / Compute all RDMs
    if rdms is None and modified_qubits is None:
        return _rdm2m_(psi.to(device=device), N, Q)

    # --- Fast path / Compute only the RDMs that touch modified qubits
    elif rdms is not None and modified_qubits is not None:
        # Unstack real and imaginary parts of `psi``
        assert rdms.ndim == 3
        assert rdms.shape[-1] == 32
        rdms_real = rdms[..., :16]
        rdms_imag = rdms[..., 16:]
        rdms = (rdms_real + 1.0j * rdms_imag).to(dtype=torch.complex64)
        return _rdm2m_partial_(psi.to(device=device), N, Q,
                               rdms.to(device=device), modified_qubits)

    # --- Error
    else:
        raise ValueError("Both `rdms` and `modified_qubits` must be "
                         "passed or be left as `None`")


def _rdm2m_(x: torch.Tensor, num_envs: int, num_qubits: int):
    # Infer device
    device = x.device

    # Allocate output tensor
    num_rdms = (num_qubits * (num_qubits - 1)) // 2
    rdms = torch.empty((num_rdms, num_envs, 4, 4), dtype=torch.complex64, device=device)

    # Row and column indexes for fanncy indexing `rdm_ji` from `rdm_ij`
    col_index = ((0,0,0,0), (2,2,2,2), (1,1,1,1), (3,3,3,3))
    row_index = ((0,2,1,3), (0,2,1,3), (0,2,1,3), (0,2,1,3))

    # Iterate over qubit pairs for all states in the batch
    for i, pair in enumerate(combinations(range(1, num_qubits + 1), 2)):
        # Compute the RDM for qubits (i, j)
        psi = x.movedim(pair, (1, 2)).reshape(num_envs, 4, -1)
        rdm_ij = torch.einsum("...ik,...jk->...ij", psi, psi.conj())
        # Create the RDM for (j, i) by fancy indexing the (i, j)-th RDM
        rdm_ji = rdm_ij[:, col_index, row_index]
        # Average the rdms and assign
        rdms[i] =  (0.5 * (rdm_ij + rdm_ji))

    # Swap the dimensions of `num_envs` and `num_rdms`
    rdms = torch.swapaxes(rdms, 0, 1).reshape(num_envs, num_rdms, 16)

    # Stack real and imaginary parts and return
    return torch.dstack([rdms.real, rdms.imag]).to(dtype=torch.float32)


def _rdm2m_partial_(x: torch.Tensor, num_envs: int, num_qubits: int,
                    rdms: torch.Tensor, modified_qubits: List[Tuple[int,int]]):
    """
    Args:
        x (torch.Tensor):
            Tensor with shape [batch_size, num_actions, 4, 4]
    """

    # Allocate output tensor
    num_rdms = (num_qubits * (num_qubits - 1)) // 2
    new_rdms = torch.empty((num_envs, num_rdms, 16), dtype=torch.complex64,
                           device=x.device)

    # Row and column indexes for fancy indexing `rdm_ji` from `rdm_ij`
    col_index = ((0,0,0,0), (2,2,2,2), (1,1,1,1), (3,3,3,3))
    row_index = ((0,2,1,3), (0,2,1,3), (0,2,1,3), (0,2,1,3))

    # Iterate over states in the batch
    for n in range(num_envs):
        nth_rdms = rdms[n].clone()
        assert nth_rdms.shape == (num_rdms, 16)
        modified = modified_qubits[n]
        qubit_pairs = combinations(range(num_qubits), 2)
        for k, pair in enumerate(qubit_pairs):
            if pair[0] in modified or pair[1] in modified:
                # Isolate state
                state = x[n].clone()
                # Compute the RDM for qubits (i, j)
                psi = state.movedim(pair, (0, 1)).reshape(4, -1)
                rdm_ij = torch.einsum("ik, jk -> ij", psi, psi.conj())
                # Create the RDM for qubit pair (j, i) by fancy indexing
                # the (i,j)-th RDM
                rdm_ji = rdm_ij[col_index, row_index]
                # Average the rdms and assign
                nth_rdms[k] = (0.5 * (rdm_ij + rdm_ji)).ravel()
            else:
                pass
        new_rdms[n] = nth_rdms

    # Stack real and imaginary parts and return
    return torch.dstack([new_rdms.real, new_rdms.imag]).to(dtype=torch.float32)


def phase_norm(states, device=None):
    N = states.shape[0]
    batch = states.reshape(N, -1)
    batch = np.hstack([batch.real, batch.imag])
    return batch

def rdm_1q(states, device=None):
    """Returns an observation of the states of the vector environment.
    The observation for a single quantum system is arrived at by calculating
    all single qubit rdm for each and every qubit of the system separately.

        Returns:
            obs: np.Array
                Numpy array of shape (N, Q, 8), giving the observations for the
                current quantum systems, where N = number of environments,
                and Q = number of qubits.
        """
    N = states.shape[0]
    Q = len(states.shape[1:])
    rdms = []
    for q in range(Q):
        sysA = (q+1,)
        sysB = tuple(p+1 for p in range(Q) if p != q)
        permutation = (0,) + sysA + sysB

        psi = np.transpose(states, permutation).reshape(N, 2, -1)
        rdm = psi @ np.transpose(psi, (0, 2, 1)).conj()
        rdms.append(rdm)
    rdms = np.array(rdms)               # rdms.shape == (Q, B, 2, 2)

    # The single qubit rmd is a complex matrix of shape (2, 2). The matrix is
    # flattened and the real and imaginary parts of each number are stacked.
    rdms = rdms.transpose((1, 0, 2, 3)).reshape(N, -1, 4)
    obs = np.dstack([rdms.real, rdms.imag])
    return rdms

def rdm_2q_complex(states, device=None):
    """
    Returns 2-qubit RDM observations with complex64 dtype.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, Q*(Q-1), 16), where N = number of episodes,
            Q = number of qubits
    """
    N = states.shape[0]
    Q = len(states.shape[1:])
    rdms = []
    qubit_pairs = permutations(range(Q), 2)

    for qubits in qubit_pairs:
        sysA = tuple(q+1 for q in qubits)
        sysB = tuple(q+1 for q in range(Q) if q not in qubits)
        permutation = (0,) + sysA + sysB
        psi = np.transpose(states, permutation).reshape(N, 4, -1)
        rdm = psi @ np.transpose(psi, (0, 2, 1)).conj()
        rdms.append(rdm)
    rdms = np.array(rdms)                # rdms.shape == (Q*(Q-1), N, 4, 4)
    rdms = rdms.transpose((1, 0, 2, 3))  # rdms.shape == (N, Q*(Q-1), 4, 4)
    obs = rdms.reshape(N, Q*(Q - 1), 16) # obs.shape  == (N, Q*(Q-1), 16)
    return obs

def rdm_2q_real(states, device=None):
    """
    Returns 2-qubit RDM observations with float32 dtype.

    Returns:
        obs: np.ndarray, dtype=np.float32
            Numpy tensor with shape (N, Q*(Q-1), 32), where N = number of episodes,
            Q = number of qubits.
    """
    rdms = rdm_2q_complex(states)           # rdms.shape = (N, Q*(Q-1), 16)
    return np.dstack([rdms.real, rdms.imag])

def rdm_2q_half(states, device=None):
    """
    Returns 2-qubit RDM observations with complex64 dtype.
    Only RDMs for qubit indices i < j are returned.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, Q*(Q-1)/2, 16), where N = number of episodes,
            Q = number of qubits
    """
    full = rdm_2q_complex(states)
    Q = len(states.shape[1:])
    idx = np.array([i < j for i,j in permutations(range(Q), 2)])
    return full[:, idx, :]


def rdm_2q_mean_complex(states, rdms=None, modified_qubits=None, device=None):
    """
    Returns 2-qubit RDM observations with complex64 dtype.
    The rdms resulting from the two different combinations of qubits (i,j) and
    (j, i) are averaged. This function supports fast path of computation,
    in which the RDMs are computed only for a subset of qubit pairs (i,j) and
    the rest of them are copied from `previous_rdms`.

    Parameters:
        states (np.ndarray) :
            Batched tensor of quantum vector states. Shape = (B, 2, 2, ..., 2)

        rdms (Optional[np.ndarray]) :
            Batched tensor of RDM observations. Shape = (B, Q*(Q-1)/2, 16)

        modified_qubits (Optional[List[tuple]]) :
            Batched list of modified qubits. The modified qubits are passed as
            tuple of indices in the system. For a given pair (i, j), if any of
            `i` or `j` is in this list, then the RDM for (i, j) are recomputed
            from `states`, otherwise it is copied from `previous_rdms`.

    Returns:
        obs (np.ndarray[np.complex64]) :
            Numpy tensor with shape (N, Q*(Q-1)/2, 16),
            where N = batch dim, Q = number of qubits
    """

    assert  (rdms is None and modified_qubits is None) or \
            (rdms is not None and modified_qubits is not None)
    if rdms is not None:
        assert len(states) == len(rdms) == len(modified_qubits)

    N = states.shape[0]
    Q = len(states.shape[1:])
    new_rdms = []

    if modified_qubits is None:
        qubit_pairs = combinations(range(Q), 2)
        # Iterate over qubit pairs for all states in the batch
        for qubits in qubit_pairs:
            sysA = tuple(q+1 for q in qubits)
            sysB = tuple(q+1 for q in range(Q) if q not in qubits)

            # qubit pair (i, j)
            permutation = (0,) + sysA + sysB
            psi = np.transpose(states, permutation).reshape(N, 4, -1)
            rdm = psi @ np.transpose(psi, (0, 2, 1)).conj() # rdm.shape = (N, 4, 4)
            rdm = rdm.reshape(N, 16)

            # qubit pair (j, i)
            permutation_rev = (0,) + tuple(reversed(sysA)) + sysB
            psi = np.transpose(states, permutation_rev).reshape(N, 4, -1)
            rdm_rev = psi @ np.transpose(psi, (0, 2, 1)).conj() # rdm_rev.shape = (N, 4, 4)
            rdm_rev = rdm_rev.reshape(N, 16)

            # Average the rdms
            rdm_avg = 0.5 * (rdm + rdm_rev)

            new_rdms.append(rdm_avg)


        # At this point `new_rdms` is with shape (Q*(Q-1)/2, N, 16).
        # We need to transpose the first 2 dimensions and convert it to NumPy array
        obs = np.array(new_rdms).transpose((1, 0, 2))

    else:
        # Iterate trough each state in the batch
        for n in range(N):
            nth_rdms = rdms[n]
            modified = modified_qubits[n]
            qubit_pairs = combinations(range(Q), 2)

            for k, pair in enumerate(qubit_pairs):
                # If one of the qubits in the `ij` pair is modified, then
                # recalculate it's RDM
                if pair[0] in modified or pair[1] in modified:
                    sysA = tuple(q for q in pair)
                    sysB = tuple(q for q in range(Q) if q not in pair)

                    # qubit pair (i, j)
                    permutation_ij = sysA + sysB
                    psi = np.transpose(states[n], permutation_ij).reshape(4, -1)
                    rdm_ij = psi @ psi.T.conj()

                    # qubit pair (j, i)
                    permutation_ji = tuple(reversed(sysA)) + sysB
                    psi = np.transpose(states[n], permutation_ji).reshape(4, -1)
                    rdm_ji = psi @ psi.T.conj()

                    nth_rdms[k] = 0.5 * (rdm_ij + rdm_ji).ravel()
                # Otherwise do nothing
                else:
                    pass

            new_rdms.append(nth_rdms)

        # At this point `new_rdms` is with shape (N, Q*(Q-1)/2, 16).
        # Turn it into NumPy array and we're done.
        obs = np.asarray(new_rdms)

    assert obs.shape == (N, Q*(Q-1)/2, 16)
    return obs


def rdm_2q_mean_real(states, rdms=None, modified_qubits=None, device=None):
    """
    Returns 2-qubit RDM observations with float32 dtype. The rdms resulting
    from the two different combinations of qubits (i,j) and (j,i) are averaged.
    See `rdm_2q_mean_complex` for more details.

    Returns:
        obs (np.ndarray[np.float32]) :
            Tensor with shape (N, Q*(Q-1)/2, 32),
            where N = batch dim, Q = number of qubits
    """
    # Unstack real and imaginary parts of `rmds`
    if rdms is not None:
        assert rdms.ndim == 3
        assert rdms.shape[-1] == 32
        rdms_real = rdms[..., :16]
        rdms_imag = rdms[..., 16:]
        rdms = (rdms_real + 1.0j * rdms_imag).astype(np.complex64)
    new_rdms = rdm_2q_mean_complex(states, rdms, modified_qubits)
    # new_rdms.shape = (N, Q*(Q-1)/2, 32)
    return np.dstack([new_rdms.real, new_rdms.imag])


def rdm_2q_nisq_mean(states, max_bitflip_noise=0.15, ampdeph=True):
    """
    Returns 2-qubit RDM observations with simulated NISQ noise added.
    Only RDMs for qubit indices i < j are returned. The RDMS resulting
    from the qubits (i,j) and (j,i) are averaged.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, Q*(Q-1)/2, 16), where N = number of
            episodes, Q = number of qubits
    """
    rdms = rdm_2q_half(states)
    N, Q, _ = rdms.shape
    _rdms = rdms.reshape(-1, 4, 4)

    # Bit-flip errors constants
    X = np.array([[0,1], [1,0]], dtype=np.complex64)
    I = np.eye(2,2, dtype=np.complex64)
    X1  = np.kron(X, I)                             # Flip qubit 1
    X2  = np.kron(I, X)                             # Flip qubit 2
    X12 = np.kron(X, X)                             # Flip qbuit 1 & 2
    bitflip_noise = np.random.uniform(0.0, max_bitflip_noise)
    p00 = (1 - bitflip_noise) ** 2                  # P(no flips)
    p01 = (1 - bitflip_noise) * bitflip_noise       # P(flip 1 qubit)
    p11 = bitflip_noise ** 2                        # P(flip 2 qubits)

    # Amplitude dampening constants
    num_qubits = int(np.ceil(Q ** 0.5))
    timesteps = np.random.randint(0, 2 ** num_qubits)
    tau = 3 * timesteps * 0.640      # 3 * nsteps * tau_2(us)
    T1 = 81.80
    T2 = 29.64
    E0a = np.array([[1, 0], [0, np.sqrt(np.exp(-tau / T1))]], dtype=np.complex64)
    E1a = np.array([[0, np.sqrt(1 - np.exp(-tau/T1))], [0, 0]], dtype=np.complex64)
    E00a = np.kron(E0a, E0a)
    E01a = np.kron(E0a, E1a)
    E10a = np.kron(E1a, E0a)
    E11a = np.kron(E1a, E1a)

    # Dephasing constants
    Tphi = (2 * T1 * T2) / (2 * T1 - T2)
    E0d = np.array([[1, 0], [0, np.exp(-tau/Tphi)]], dtype=np.complex64)
    E1d = np.array([[0, 0], [0, np.sqrt(1 - np.exp(-2*tau/Tphi))]], dtype=np.complex64)
    E00d = np.kron(E0d, E0d)
    E01d = np.kron(E0d, E1d)
    E10d = np.kron(E1d, E0d)
    E11d = np.kron(E1d, E1d)

    result = []
    for rdm in _rdms:
        if ampdeph:
            # Amplitude damping
            ad_rdm = E00a @ rdm @ E00a.conj().T + \
                     E01a @ rdm @ E01a.conj().T + \
                     E10a @ rdm @ E10a.conj().T + \
                     E11a @ rdm @ E11a.conj().T
            # Dephasing
            df_rdm = E00d @ ad_rdm @ E00d.conj().T + \
                     E01d @ ad_rdm @ E01d.conj().T + \
                     E10d @ ad_rdm @ E10d.conj().T + \
                     E11d @ ad_rdm @ E11d.conj().T
        else:
            df_rdm = rdm
        # Bit flip errors
        bf_rdm = p00 * df_rdm + \
                 2*p01*(X1 @ df_rdm @ X1.conj().T + X2 @ df_rdm @ X2.conj().T) + \
                 p11 * (X12 @ df_rdm @ X12.conj().T)
        # -----
        noisy_rdm_ij = bf_rdm                                     # rho(i,j) i<j
        noisy_rdm_ji = noisy_rdm_ij[[0,2,1,3], :][:, [0,2,1,3]]   # rho(j,i)
        # Average
        result.append(0.5 * (noisy_rdm_ij + noisy_rdm_ji))

    return np.array(result).reshape(N, Q, 16)


def rdm_2q_nisq_mean_real(states, max_bitflip_noise=0.15, ampdeph=True):
    """
    Returns 2-qubit RDM observations with simulated NISQ noise added.
    Only RDMs for qubit indices i < j are returned. The real and imaginary
    parts are stacked in the last dimension.

    Returns:
        obs: np.ndarray, dtype=np.float32
            Numpy tensor with shape (N, Q*(Q-1)/2, 32), where N = number of
            episodes, Q = number of qubits
    """
    rdms = rdm_2q_nisq_mean(states, max_bitflip_noise, ampdeph)
    return np.dstack([rdms.real, rdms.imag])


def rsqr(states, max_angle=np.pi/6, max_rotated_qubits=1):
    """
    Returns vector states with random single qubit rotations applied
    over at most `max_rotated_qubits`. The axis of rotation is arbitrary,
    the angle is at most ±`max_angle`.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, 2**Q), where N = number of
            episodes, Q = number of qubits
    """
    from scipy.linalg import expm

    N = states.shape[0]
    Q = states.ndim - 1
    choices = np.arange(Q, dtype=np.int32)
    shape = states.shape[1:]

    # Apply random rotations
    rotated_states = states.copy()
    for n in range(N):
        s = rotated_states[n]
        num_rotated_qubits = np.random.randint(1, max_rotated_qubits+1)
        indices = np.random.choice(choices, num_rotated_qubits, replace=False)
        for i in indices:
            phi = np.random.uniform(0.0, 2*np.pi)
            theta = np.random.uniform(0.0, np.pi)
            angle = np.random.uniform(-max_angle, max_angle)
            h = (np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta))
            M = h[0] * SIGMA_X + h[1] * SIGMA_Y + h[2] * SIGMA_Z
            U = expm(-1.0j * (angle / 2) * M)
            P = (i,) + tuple(j for j in range(Q) if i != j)
            s_rotated = U @ np.transpose(s, P).reshape(2, -1)
            s_rotated = np.transpose(s_rotated.reshape(shape), np.argsort(P))
            s = s_rotated
        rotated_states[n] = s
    return np.array(rotated_states)


def rdm_2q_rsqr_mean(states, max_angle=np.pi/6, max_rotated_qubits=1):
    """
    Returns 2-qubit RDM observations with random single qubit rotations applied
    over at most `max_rotated_qubits`. The axis of rotation is arbitrary,
    the angle is at most ±`max_angle`. RDMs for (i,j) and (j,i) are averaged.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, Q*(Q-1)/2, 16), where N = number of
            episodes, Q = number of qubits
    """
    rotated_states = rsqr(states, max_angle, max_rotated_qubits)
    return rdm_2q_mean_complex(rotated_states)


def rdm_2q_rsqr_mean_real(states, max_angle=np.pi/6, max_rotated_qubits=1):
    """
    Returns 2-qubit RDM observations with random single qubit rotations applied
    over at most `max_rotated_qubits`. The axis of rotation is arbitrary,
    the angle is at most ±`max_angle`. RDMs for (i,j) and (j,i) pairs are
    averaged. Real and imaginary parts are stacked in the last dimension.

    Returns:
        obs: np.ndarray, dtype=np.float32
            Numpy tensor with shape (N, Q*(Q-1)/2, 32), where N = number of
            episodes, Q = number of qubits
    """
    rdms = rdm_2q_rsqr_mean(states, max_angle, max_rotated_qubits)
    return np.dstack([rdms.real, rdms.imag])


def rdm_2q_rsqr_nisq_mean(states, max_angle=np.pi/6, max_rotated_qubits=1,
                          max_bitflip_noise=0.15, ampdeph=True):
    """
    Returns 2-qubit RDM observations with random single qubit rotations applied
    and simulated NISQ noise. RDMs for (i,j) and (j,i) pairs are averaged.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, Q*(Q-1)/2, 16), where N = number of
            episodes, Q = number of qubits
    """
    rotated_states = rsqr(states, max_angle, max_rotated_qubits)
    return rdm_2q_nisq_mean(rotated_states, max_bitflip_noise, ampdeph)


def rdm_2q_rsqr_nisq_mean_real(states, max_angle=np.pi/6, max_rotated_qubits=1,
                               max_bitflip_noise=0.15, ampdeph=True):
    """
    Returns 2-qubit RDM observations with random single qubit rotations applied
    and simulated NISQ noise. RDMs for (i,j) and (j,i) pairs are averaged.

    Returns:
        obs: np.ndarray, dtype=np.float32
            Numpy tensor with shape (N, Q*(Q-1)/2, 32), where N = number of
            episodes, Q = number of qubits
    """
    rotated_states = rsqr(states, max_angle, max_rotated_qubits)
    return rdm_2q_nisq_mean_real(rotated_states, max_bitflip_noise, ampdeph)
