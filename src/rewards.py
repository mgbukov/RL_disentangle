import numpy as np
import torch


def sparse(entanglements, prev_entanglements, epsi):
    terminated = torch.all(entanglements <= epsi, axis=1)
    rewards = -5. + 10. * terminated
    return rewards

def relative_delta(entanglements, prev_entanglements, epsi):
    # We should probably use both deltas. If the agent starts applying a gate
    # to a disentangled pair it results in swapping the entanglements and thus
    # can achieve the max reward at each step. So we have to penalize for the
    # other entanglement.
    entanglements = torch.maximum(entanglements, torch.tensor([epsi]))
    prev_entanglements = torch.maximum(prev_entanglements, torch.tensor([epsi]))
    deltas = (prev_entanglements - entanglements) / torch.maximum(prev_entanglements, entanglements)
    entangled_qubits = (entanglements > torch.tensor([epsi])).sum(axis=1)
    rewards = deltas.sum(dim=1) - entangled_qubits
    return rewards
