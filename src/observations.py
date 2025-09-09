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
