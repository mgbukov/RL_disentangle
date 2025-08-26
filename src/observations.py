from itertools import combinations, permutations
from typing import List, Optional, Tuple

import torch


GPU_DEVICE = "cuda"


def rdm_2q_complex(x: torch.tensor):
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

    x = x.to(device=GPU_DEVICE)
    x.requires_grad_ = False

    rdms_shape = (Q*(Q-1), N, 4, 4)
    rdms = torch.zeros(rdms_shape, dtype=torch.complex64, requires_grad=False,
        device=GPU_DEVICE)

    for i, qubits in enumerate(permutations(range(Q), 2)):
        sysA = tuple(q+1 for q in qubits)
        sysB = tuple(q+1 for q in range(Q) if q not in qubits)
        permutation = (0,) + sysA + sysB
        psi = torch.permute(x, permutation).reshape(N, 4, -1)
        rdm = psi @ torch.permute(psi, (0, 2, 1)).conj()
        rdms[i, :, :, :] = rdm

    return torch.swapaxes(rdms, 0, 1).reshape(N, Q*(Q-1), 16)



def rdm_2q_mean_complex(x: torch.tensor, rdms: Optional[torch.tensor] = None,
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
    if rdms is not None:
        assert len(x) == len(rdms) == len(modified_qubits)

    N = x.shape[0]
    Q = x.ndim - 1

    # Allocate output tensor
    rdms_shape = (Q*(Q-1), N, 4, 4)
    new_rdms = torch.zeros(
        rdms_shape, dtype=torch.complex64, requires_grad=False, device=GPU_DEVICE
    )

    # Move `x` to GPU device
    x.requires_grad_ = False
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
        return torch.swapaxes(new_rdms, 0, 1).reshape(N, Q*(Q-1), 16)

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


def rdm_2q_mean_real(x: torch.tensor, rdms: Optional[torch.tensor] = None,
                     modified_qubits: Optional[torch.tensor] = None):
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
