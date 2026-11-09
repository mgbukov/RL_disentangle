import numpy as np
import torch


def sparse(entanglements: torch.Tensor|np.ndarray,
           prev_entanglements: torch.Tensor|np.ndarray,
           epsi: torch.Tensor|np.ndarray
    ):
    if isinstance(entanglements, torch.Tensor):
        terminated = torch.all(entanglements <= epsi, dim=1)
    elif isinstance(entanglements, np.ndarray):
        terminated = np.all(entanglements <= epsi, axis=1)
    else:
        raise ValueError("Unexpected type for `entanglements`")
    rewards = -5. + 10. * terminated
    return rewards


def relative_delta(entanglements: torch.Tensor|np.ndarray,
                   prev_entanglements: torch.Tensor|np.ndarray,
                   epsi: torch.Tensor|np.ndarray
    ):
    # We should probably use both deltas. If the agent starts applying a gate
    # to a disentangled pair it results in swapping the entanglements and thus
    # can achieve the max reward at each step. So we have to penalize for the
    # other entanglement.
    if isinstance(entanglements, torch.Tensor) and isinstance(prev_entanglements, torch.Tensor):
        entanglements = torch.maximum(entanglements, torch.tensor([epsi]))
        prev_entanglements = torch.maximum(prev_entanglements, torch.tensor([epsi]))
        deltas = (prev_entanglements - entanglements) / torch.maximum(prev_entanglements, entanglements)
        entangled_qubits = (entanglements > torch.tensor([epsi])).sum(dim=1)
        rewards = deltas.sum(dim=1) - entangled_qubits

    elif isinstance(entanglements, np.ndarray) and isinstance(prev_entanglements, np.ndarray):
        entanglements = np.maximum(entanglements, epsi)
        prev_entanglements = np.maximum(prev_entanglements, epsi)
        deltas = (prev_entanglements - entanglements) / np.maximum(prev_entanglements, entanglements)
        entangled_qubits = (entanglements > epsi).sum(axis=1)
        rewards = deltas.sum(axis=1) - entangled_qubits

    else:
        raise ValueError("Unexpected types for `entanglements` and `prev_entanglements`")
    return rewards
