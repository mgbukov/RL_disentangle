from itertools import combinations, permutations
from typing import List, Optional, Tuple

import torch


def vec(states: torch.Tensor):
    N = states.shape[0]
    batch = states.reshape(N, -1)
    batch = torch.hstack([batch.real, batch.imag])
    return batch


def rdm2m(psi: torch.Tensor,
          rdms: Optional[torch.Tensor] = None,
          modified_qubits: Optional[List[Tuple[int,int]]] = None,
          device: str|torch.device = "cuda"):

    N = psi.shape[0]
    Q = psi.ndim - 1

    # --- Slow path / Compute all RDMs
    if rdms is None and modified_qubits is None:
        return _rdm2m_(psi.to(device=device), N, Q)
        # if D == torch.device("cpu"):
        #     return _rdm2m_cpu(psi.to(device=D), N, Q)
        # elif D == torch.device("cuda"):
        #     return _rdm2m_cuda(psi.to(device=D), N, Q)
        # else:
        #     raise ValueError(f"Unsupported device: '{D}'")

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
    rdms = torch.zeros((num_rdms, num_envs, 4, 4), dtype=torch.complex64, device=device)

    # Row and column indexes for fanncy indexing `rdm_ji` from `rdm_ij`
    col_index = ((0,0,0,0), (2,2,2,2), (1,1,1,1), (3,3,3,3))
    row_index = ((0,2,1,3), (0,2,1,3), (0,2,1,3), (0,2,1,3))

    # Iterate over qubit pairs for all states in the batch
    for i, pair in enumerate(combinations(range(num_qubits), 2)):
        # Compute the RDM for qubits (i, j)
        psi = x.movedim(pair, (1, 2)).reshape(num_envs, 4, -1)
        rdm_ij = torch.einsum("...ik,...jk->...ij", psi, psi.conj())
        # Create the RDM for (j, i) by fancy indexing the (i, j)-th RDM
        rdm_ji = rdm_ij[:, col_index, row_index]
        # Average the rdms and assign
        rdms[i] =  (0.5 * (rdm_ij + rdm_ji))

    # Swap the dimenstions of `num_envs` and `num_rdms`
    rdms = torch.swapaxes(rdms, 0, 1).reshape(num_envs, num_rdms, 16)

    # Stack real and imaginary parts and return
    return torch.dstack([rdms.real, rdms.imag]).to(dtype=torch.float32)


def _rdm2m_cuda(x: torch.Tensor, num_envs: int, num_qubits: int):

    # Allocate output tensor
    num_rdms = (num_qubits * (num_qubits - 1)) // 2
    rdms = torch.empty((num_rdms, num_envs, 4, 4), dtype=torch.complex64, device="cuda")

    # Row and column indexes for fanncy indexing `rdm_ji` from `rdm_ij`
    col_index = ((0,0,0,0), (2,2,2,2), (1,1,1,1), (3,3,3,3))
    row_index = ((0,2,1,3), (0,2,1,3), (0,2,1,3), (0,2,1,3))

    # Iterate over qubit pairs for all states in the batch
    for i, pair in enumerate(combinations(range(1, num_qubits + 1), 2)):
        # Compute the RDM for qubits (i, j)
        psi = x.movedim(pair, (1, 2)).reshape(num_envs, 4, -1)
        rdm_ij = torch.einsum("...ik,...jk->...ij", psi, psi.conj())
        # Create the RDM for qubit pair (j, i) by fancy indexing the (i,j)-th RDM
        rdm_ji = rdm_ij[:, col_index, row_index]
        # Average the rdms and assign
        rdms[i] =  (0.5 * (rdm_ij + rdm_ji))

    # Swap the dimenstions of `num_envs` and `num_rdms` and ravel
    rdms = torch.swapaxes(rdms, 0, 1).reshape(num_envs, num_rdms, 16)

    # Stack real and imaginary parts and return
    return torch.dstack([rdms.real, rdms.imag]).to(dtype=torch.float32)



def _rdm2m_partial_(x: torch.Tensor, num_envs: int, num_qubits: int,
                    rdms: torch.Tensor, modified_qubits: List[Tuple[int,int]]):

    # Allocate output tensor
    num_rdms = (num_qubits * (num_qubits - 1)) // 2
    new_rdms = torch.empty((num_envs, num_rdms, 16), dtype=torch.complex64,
                           device=x.device)

    # Row and column indexes for fanncy indexing `rdm_ji` from `rdm_ij`
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
                state = x[n]
                # Compute the RDM for qubits (i, j)
                psi = state.movedim(pair, (0, 1)).reshape(4, -1)
                rdm_ij = torch.einsum("...ik, ...jk -> ...ij", psi, psi.conj())
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


def _cuda_rdm_2q_complex(x: torch.Tensor):
    """
    Returns 2-qubit RDM observations with complex64 dtype.

    Parameters
    ----------
    x: torch.tensor, dtype=torch.complex64
        Batch of state vectors, shape = (N, 2, 2, ..., 2). The number of qubits
        is infered as `x.ndim - 1`

    Returns
    -------
    obs: torch.tensor, dtype=torch.complex64
        Tensor with shape (N, Q*(Q-1), 16), where N = number of states
        in the batch, Q = number of qubits
    """
    assert x.dtype == torch.complex64, f"Expected complex64, got {x.dtype}"

    # Get number of states (batch size) and number of qubits
    N = x.shape[0]
    Q = x.ndim - 1

    rdms_shape = (Q*(Q-1), N, 4, 4)

    # Allocate output tensor
    rdms = torch.zeros(rdms_shape, dtype=torch.complex64, device="cuda")

    for i, (q0,q1) in enumerate(permutations(range(Q), 2)):
        psi = torch.moveaxis(x, (q0+1,q1+1), (1,2)).reshape(N, 4, -1)
        # The following line does matrix multiplication of Ψ and Ψ†
        rdm = torch.einsum("...ik,...jk->...ij", psi, psi.conj())
        rdms[i, :, :, :] = rdm

    return torch.swapaxes(rdms, 0, 1).reshape(N, Q*(Q-1), 16)


def rdm_2q_mean_complex(x: torch.Tensor, rdms: Optional[torch.Tensor] = None,
                        modified_qubits: Optional[List[Tuple]] = None):
    """
    Returns 2-qubit RDM observations with complex64 dtype.
    The rdms resulting from the two different combinations of qubits (i,j) and
    (j, i) are averaged. This function supports fast path of computation,
    in which the RDMs are computed only for a subset of qubit pairs (i,j) and
    the rest of them are copied from `previous_rdms`.

    Parameters
    ----------
    states (np.ndarray) :
        Batched tensor of quantum vector states. Shape = (B, 2, 2, ..., 2)

    rdms (Optional[np.ndarray]) :
        Batched tensor of RDM observations. Shape = (B, Q*(Q-1)/2, 16)

    modified_qubits (Optional[List[tuple]]) :
        Batched list of modified qubits. The modified qubits are passed as
        tuple of indices in the system. For a given pair (i, j), if any of
        `i` or `j` is in this list, then the RDM for (i, j) are recomputed
        from `states`, otherwise it is copied from `previous_rdms`.

    Returns
    -------
    obs (np.ndarray[np.complex64]) :
        Numpy tensor with shape (N, Q*(Q-1)/2, 16),
        where N = batch dim, Q = number of qubits
    """

    assert  (rdms is None and modified_qubits is None) or \
            (rdms is not None and modified_qubits is not None)
    if rdms is not None and modified_qubits is not None:
        assert len(x) == len(rdms) == len(modified_qubits)

    N = x.shape[0]
    Q = x.ndim - 1

    # Allocate output tensor
    rdms_shape = (Q*(Q-1)//2, N, 16)
    new_rdms = torch.zeros(rdms_shape, dtype=torch.complex64, device="cuda")

    # Move `x` to GPU device
    x = x.to(device=GPU_DEVICE)

    ### Case 1: Prevous RDM observations not given. This is the slow path
    if modified_qubits is None:
        # Iterate over qubit pairs for all states in the batch
        for i, qubits in enumerate(combinations(range(Q), 2)):
            sysA = tuple(q+1 for q in qubits)
            sysB = tuple(q+1 for q in range(Q) if q not in qubits)

            # qubit pair (i, j)
            P_ij = (0,) + sysA + sysB
            psi = torch.permute(x, P_ij).reshape(N, 4, -1)
            rdm_ij = (psi @ torch.swapaxes(psi, 1, 2).conj()).reshape(N, 16)

            # qubit pair (j, i)
            P_ji = (0,) + tuple(reversed(sysA)) + sysB
            psi = torch.permute(x, P_ji).reshape(N, 4, -1)
            rdm_ji = (psi @ torch.swapaxes(psi, 1, 2).conj()).reshape(N, 16)

            # Average the rdms and assign
            new_rdms[i] =  0.5 * (rdm_ij + rdm_ji)

        # `new_rdms` is with shape (Q*(Q-1)/2, N, 16).
        # We need to transpose the first 2 dimensions
        return torch.swapaxes(new_rdms, 0, 1).reshape(N, Q*(Q-1)//2, 16)

    ### Case 2: Previous RDM observations are given. This is the fast path
    else:
        # Swap the positions of Q*(Q-1) and N axes
        new_rdms = torch.swapaxes(new_rdms, 0, 1)

        # Move `rdms` to GPU device
        rdms = rdms.to(device=GPU_DEVICE)

        # Iterate trough each state in the batch
        for n in range(N):
            nth_rdms = rdms[n]
            modified = modified_qubits[n]

            for k, pair in enumerate(combinations(range(Q), 2)):
                # If one of the qubits in the `ij` pair is modified, then
                # recalculate it's RDM
                if pair[0] in modified or pair[1] in modified:
                    sysA = tuple(q for q in pair)
                    sysB = tuple(q for q in range(Q) if q not in pair)

                    # qubit pair (i, j)
                    P_ij = sysA + sysB
                    psi = torch.permute(x[n], P_ij).reshape(4, -1)
                    rdm_ij = psi @ psi.T.conj()

                    # qubit pair (j, i)
                    P_ji = tuple(reversed(sysA)) + sysB
                    psi = torch.permute(x[n], P_ji).reshape(4, -1)
                    rdm_ji = psi @ psi.T.conj()

                    nth_rdms[k] = 0.5 * (rdm_ij + rdm_ji).ravel()

            new_rdms[n] = nth_rdms

        # `new_rdms` is with shape (N, Q*(Q-1)/2, 16).
        return new_rdms



def rdm_2q_mean_real(x: torch.Tensor, rdms: Optional[torch.Tensor] = None,
                     modified_qubits: Optional[List[Tuple]] = None):
    """
    Returns 2-qubit RDM observations with float32 dtype. The rdms resulting
    from the two different combinations of qubits (i,j) and (j,i) are averaged.
    See `rdm_2q_mean_complex` for more details.

    Returns
    -------
    obs: torch.tensor, dtype=torch.float32]
        Tensor with shape (N, Q*(Q-1)/2, 32), where N = batch size, Q = number of qubits
    """
    assert x.dtype == torch.complex64, f"Expected complex64, got {x.dtype}"

    # Unstack real and imaginary parts of `rmds`
    if rdms is not None:
        assert rdms.ndim == 3
        assert rdms.shape[-1] == 32
        rdms_real = rdms[..., :16]
        rdms_imag = rdms[..., 16:]
        rdms = (rdms_real + 1.0j * rdms_imag).to(dtype=torch.complex64)

    new_rdms = rdm_2q_mean_complex(x, rdms, modified_qubits)
    # new_rdms.shape = (N, Q*(Q-1)/2, 32)
    return torch.dstack([new_rdms.real, new_rdms.imag])


# @torch.compile(backend="cudagraphs")
def cuda_rdm_2q_mean_complex_fast(x: torch.Tensor, rdms: torch.Tensor, modified_qubits: List[Tuple]):
    """
    Returns 2-qubit RDM observations with complex64 dtype.
    The rdms resulting from the two different combinations of qubits (i,j) and
    (j, i) are averaged. This function supports fast path of computation,
    in which the RDMs are computed only for a subset of qubit pairs (i,j) and
    the rest of them are copied from `previous_rdms`.

    Parameters
    ----------
    states (np.ndarray) :
        Batched tensor of quantum vector states. Shape = (B, 2, 2, ..., 2)

    rdms (Optional[np.ndarray]) :
        Batched tensor of RDM observations. Shape = (B, Q*(Q-1)/2, 16)

    modified_qubits (Optional[List[tuple]]) :
        Batched list of modified qubits. The modified qubits are passed as
        tuple of indices in the system. For a given pair (i, j), if any of
        `i` or `j` is in this list, then the RDM for (i, j) are recomputed
        from `states`, otherwise it is copied from `previous_rdms`.

    Returns
    -------
    obs (np.ndarray[np.complex64]) :
        Numpy tensor with shape (N, Q*(Q-1)/2, 16),
        where N = batch dim, Q = number of qubits
    """

    assert  (rdms is None and modified_qubits is None) or \
            (rdms is not None and modified_qubits is not None)
    if rdms is not None and modified_qubits is not None:
        assert len(x) == len(rdms) == len(modified_qubits)

    N = x.shape[0]
    Q = x.ndim - 1

    # Allocate output tensor
    rdms_shape = (Q*(Q-1)//2, N, 16)
    new_rdms = torch.zeros(rdms_shape, dtype=torch.complex64, device="cuda")

    # Move `x` to GPU device
    x = x.to(device=GPU_DEVICE)

    ### Case 1: Prevous RDM observations not given. This is the slow path
    if modified_qubits is None:
        # Iterate over qubit pairs for all states in the batch
        for i, qubits in enumerate(combinations(range(Q), 2)):
            sysA = tuple(q+1 for q in qubits)
            sysB = tuple(q+1 for q in range(Q) if q not in qubits)

            # qubit pair (i, j)
            P_ij = (0,) + sysA + sysB
            psi = torch.permute(x, P_ij).reshape(N, 4, -1)
            rdm_ij = (psi @ torch.swapaxes(psi, 1, 2).conj()).reshape(N, 16)

            # qubit pair (j, i)
            P_ji = (0,) + tuple(reversed(sysA)) + sysB
            psi = torch.permute(x, P_ji).reshape(N, 4, -1)
            rdm_ji = (psi @ torch.swapaxes(psi, 1, 2).conj()).reshape(N, 16)

            # Average the rdms and assign
            new_rdms[i] =  0.5 * (rdm_ij + rdm_ji)

        # `new_rdms` is with shape (Q*(Q-1)/2, N, 16).
        # We need to transpose the first 2 dimensions
        return torch.swapaxes(new_rdms, 0, 1).reshape(N, Q*(Q-1)//2, 16)

    ### Case 2: Previous RDM observations are given. This is the fast path
    else:
        # Swap the positions of Q*(Q-1) and N axes
        new_rdms = torch.swapaxes(new_rdms, 0, 1)

        # Move `rdms` to GPU device
        rdms = rdms.to(device=GPU_DEVICE)

        # Iterate trough each state in the batch
        for n in range(N):
            nth_rdms = rdms[n]
            modified = modified_qubits[n]

            for k, pair in enumerate(combinations(range(Q), 2)):
                # If one of the qubits in the `ij` pair is modified, then
                # recalculate it's RDM
                if pair[0] in modified or pair[1] in modified:
                    sysA = tuple(q for q in pair)
                    sysB = tuple(q for q in range(Q) if q not in pair)

                    # qubit pair (i, j)
                    P_ij = sysA + sysB
                    psi = torch.permute(x[n], P_ij).reshape(4, -1)
                    rdm_ij = psi @ psi.T.conj()

                    # qubit pair (j, i)
                    P_ji = tuple(reversed(sysA)) + sysB
                    psi = torch.permute(x[n], P_ji).reshape(4, -1)
                    rdm_ji = psi @ psi.T.conj()

                    nth_rdms[k] = 0.5 * (rdm_ij + rdm_ji).ravel()

            new_rdms[n] = nth_rdms

        # `new_rdms` is with shape (N, Q*(Q-1)/2, 16).
        return new_rdms


def rdm_2q_mean_real(x: torch.Tensor, rdms: Optional[torch.Tensor] = None,
                     modified_qubits: Optional[List[Tuple]] = None):
    """
    Returns 2-qubit RDM observations with float32 dtype. The rdms resulting
    from the two different combinations of qubits (i,j) and (j,i) are averaged.
    See `rdm_2q_mean_complex` for more details.

    Returns
    -------
    obs: torch.tensor, dtype=torch.float32]
        Tensor with shape (N, Q*(Q-1)/2, 32), where N = batch size, Q = number of qubits
    """
    assert x.dtype == torch.complex64, f"Expected complex64, got {x.dtype}"

    # Unstack real and imaginary parts of `rmds`
    if rdms is not None:
        assert rdms.ndim == 3
        assert rdms.shape[-1] == 32
        rdms_real = rdms[..., :16]
        rdms_imag = rdms[..., 16:]
        rdms = (rdms_real + 1.0j * rdms_imag).to(dtype=torch.complex64)

    new_rdms = rdm_2q_mean_complex(x, rdms, modified_qubits)
    # new_rdms.shape = (N, Q*(Q-1)/2, 32)
    return torch.dstack([new_rdms.real, new_rdms.imag])
