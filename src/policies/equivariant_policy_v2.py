from itertools import permutations
from math import log2

import torch
import torch.nn as nn

from src.policies.base_policy import BasePolicy


class EquivariantPolicyV2(nn.Module, BasePolicy):

    def __init__(self, actions, policy, **kwargs):
        super().__init__()

        # Store arguments for model initialization.
        # Kwargs dict is used to save and restore the model.
        self.kwargs = dict(actions=actions,
                           policy=policy,
                           kwargs=kwargs)

        self.input_size = kwargs["input_size"]
        self.out_size = kwargs["out_size"]

        # Bookkeeping for the group of permutations.
        self.L = int(log2(self.input_size)) - 1
        self.size_G = fact(self.L)
        self.qubit_perms = {id: perm for id, perm in enumerate(permutations(range(self.L)))}
        self.perm_index = {perm: id for id, perm in self.qubit_perms.items()}
        self.cayley_table = {(i, j): self.perm_index[_perm_prod(pi, pj)]
                for i, pi in self.qubit_perms.items()
                for j, pj in self.qubit_perms.items()}
        self.inverse_table = {i: self.perm_index[_perm_inverse(pi)]
                for i, pi in self.qubit_perms.items()}
        self.qubit_act_cosets = {i: [g_id for g_id, g in self.qubit_perms.items() if (g[0], g[1])==act]
                for i, act in actions.items()}

        # Create an ensemble of models.
        self.models = nn.ModuleList(policy(**kwargs) for _ in range(self.size_G))

    def forward(self, x):
        # Make sure the input is a 3D tensor -- a batch of sequences.
        initial_shape = x.shape
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=1)
        if len(x.shape) != 3:
            raise ValueError("Unknown input shape")
        b, t, q = x.shape

        # y is the output tensor that we will return.
        y = torch.zeros(size=(b, t, self.out_size), device=self.device)
        for i in range(self.out_size):
            y_i = torch.zeros(size=(b, t), device=self.device)
            for u in self.qubit_act_cosets[i]:
                # Run a forward pass for every member of the coset.
                result = torch.zeros(size=(b, t, self.out_size), device=self.device)
                for g in self.qubit_perms.keys():
                    ug_inv = self._prod_with_inverse(u, g)
                    x_p = _qubit_perm_apply(x, ug_inv)
                    out = self.models[g](x_p)
                    result += out
                result = result / self.size_G

                # After the forward pass we get output of shape (b, t, o).
                # We need to produce one single number for our current action y[i]. Thus,
                # we need to sum the outputs (or take mean, w/e). The reason for this is
                # that the values in this output vector are not associated in any way with
                # the other output classes, but only with class `i`.
                y_i += result.mean(dim=-1)

            # After iterating over all the groups in the coset of action `y_i` we need to
            # "project down" to the action space.
            # y[i] = 1/|H| * Sum_u y^G(u) = 1/|H| * Sum_u Sum_G (f*w)(u)
            y[:, :, i] = y_i / len(self.qubit_act_cosets[i])

        # Squeeze the shape of the output if the input was not a sequence.
        if len(initial_shape) == 2:
            y = torch.squeeze(y, dim=1)

        return y

    @property
    def device(self): return self.models[0].device

    @torch.no_grad()
    def _prod_with_inverse(self, u, g):
        return self.qubit_perms[self.cayley_table[u, self.inverse_table[g]]]


#----------------------------------- Helper functions -----------------------------------#
fact = lambda n: 1 if n<=1 else n*fact(n-1)

def _perm_inverse(p):
    """Compute the inverse of a permutation.

    Args:
        p (tuple[int]): A tuple of integers representing the permutation pi,
            e.g. (2, 0, 3, 1).

    Result:
        p_inv (tuple[int]): A tuple of integers representing the permutation inverse, i.e.
            p_inv * p = id, e.g. (1, 3, 0, 2).
    """
    p_inv = [None] * len(p)
    for i, t in enumerate(p):
        p_inv[t] = i
    return tuple(p_inv)

def _perm_prod(pi, pj):
    """Compute the product of two permutations.

    Args:
        pi (tuple[int]): A tuple of integers representing the permutation pi,
            e.g. (3, 1, 2, 0).
        pi (tuple[int]): A tuple of integers representing the permutation pj,
            e.g. (2, 0, 1, 3).

    Result:
        p (tuple[int]): A tuple of integers representing the permutation p = pi * pj,
            e.g. (3, 0, 1, 2).
    """
    return tuple(pi[i] for i in pj)

def _qubit_perm_apply(x, g):
    """Applying permutation `g` to input `x`.
    Our input has a shape of (b, t, q), consisting of q "features". The input represents a
    quantum system of L qubits. Here q = 2 ** (L+1). The permutation `g` should be applied
    at the `qubit-level` instead of at the `feature-level`.

    Args:
        x (torch.Tensor): A tensor of shape (b, t, 2**(L+1)), giving the input to be permuted.
        g (tuple[int]): A tuple of integers representing the permutation g.

    Result:
        x_p (torch.Tensor): A tensor of shape (b, t, 2**(L+1)), giving the permuted input.
    """
    b, t, q = x.shape
    L = int(log2(q)) - 1
    #                                           /------- L ------\
    # Reshape the input into the shape (b, t, 2, 2, 2, 2, ...., 2).
    x_p = x.reshape((b, t) + (2,)*(L+1))
    #
    # Having the input reshaped in this form, we can perform the permutation transformation
    # using transposition of the tensor axes. We need to permute the last L axes.
    x_p = x_p.permute(0, 1, 2, *list(map(lambda x: x+3, g)))
    #
    # After transposing the tensor, reshape back into the original shape.
    return x_p.reshape(b, t, 2**(L+1))

#