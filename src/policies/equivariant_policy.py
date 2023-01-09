from itertools import permutations
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.policies.base_policy import BasePolicy


class EquivariantPolicy(nn.Module, BasePolicy):
    def __init__(self, input_size, hidden_sizes, actions, dropout_rate=0.0):
        """Initialize a policy model.

        Args:
            input_size (int): Size of the environment state.
            hidden_sizes (list[int]): A list of sizes for the hidden layers.
            actions (dict): A mapping representing the actions of the agent.
            dropout_rate (float, optional): Dropout probability. Default value is 0.0.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.out_size = len(actions)
        self.dropout_rate = dropout_rate

        # Store arguments for model initialization.
        # Kwargs dict is used to save and restore the model.
        self.kwargs = dict(input_size=input_size,
                           hidden_sizes=hidden_sizes,
                           actions=actions,
                           dropout_rate=dropout_rate)

        # Bookkeeping for the group of permutations.
        self.L = int(log2(input_size)) - 1
        self.size_G = fact(self.L)
        self.qubit_perms = {id: perm for id, perm in enumerate(permutations(range(self.L)))}
        self.qubit_act_cosets = {i: [g_id for g_id, g in self.qubit_perms.items() if (g[0], g[1])==act]
                for i, act in actions.items()}

        # Initialize the model architecture.
        self.num_layers = len(hidden_sizes)
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        if len(hidden_sizes) > 0:
            h = hidden_sizes[0]
            self.hidden_layers.append(EquivariantLayer(input_size, h))
            fan_in = h
            fan_out = h
            for fan_out in hidden_sizes[1:]:
                self.hidden_layers.append(nn.Linear(fan_in, fan_out))
                self.dropout_layers.append(nn.Dropout(dropout_rate))
                fan_in = fan_out
            self.output_layer = nn.Linear(fan_out, self.out_size)
        else: # If there are no hidden layers, then the output layer must be equivariant.
            self.output_layer = EquivariantLayer(input_size, self.out_size)

        # Initialize model parameters.
        for param in self.parameters():
            if len(param.shape) == 2: # weight
                nn.init.kaiming_uniform_(param)
            elif len(param.shape) == 1: # biases
                nn.init.uniform_(param, -0.01, 0.01)

    def forward(self, x):
        """Take a mini-batch of environment states and compute scores over the possible
        actions.

        Args:
            x (torch.Tensor): Tensor of shape (b, q), or (b, t, q), giving the current
                state of the environment, where b = batch size, t = number of time steps,
                q = size of the quantum system (2 ** num_qubits).

        Returns:
            y (torch.Tensor): Tensor of shape (b, num_actions), or (b, t, num_acts),
                giving a score to every action from the action space.
        """
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
                out = self._forward(x, u) # shape (b, t, o)

                # After the forward pass we get output of shape (b, t, o).
                # We need to produce one single number for our current action y[i]. Thus,
                # we need to sum the outputs (or take mean, w/e). The reason for this is
                # that the values in this output vector are not associated in any way with
                # the other output classes, but only with class `i`.
                y_i += out.mean(dim=-1)

            # After iterating over all the groups in the coset of action `y_i` we need to
            # "project down" to the action space.
            # y[i] = 1/|H| * Sum_u y^G(u) = 1/|H| * Sum_u Sum_G (f*w)(u)
            y[:, :, i] = y_i / len(self.qubit_act_cosets[i])

        # Squeeze the shape of the output if the input was not a sequence.
        if len(initial_shape) == 2:
            y = torch.squeeze(y, dim=1)

        return y

    def _forward(self, x, u):
        out = x
        for idx in range(len(self.hidden_layers)):
            if idx == 0:
                out = self.hidden_layers[idx](out, u)
            else:
                out = self.hidden_layers[idx](out)
            out = F.relu(out)
            # out = self.dropout_layers[idx](out)

        if len(self.hidden_layers) > 0:
            out = self.output_layer(out)
        else:
            out = self.output_layer(x, u)

        return out

class EquivariantLayer(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.input_size = input_size
        self.out_size = out_size

        # Bookkeeping for the group of permutations.
        self.L = int(log2(input_size)) - 1
        self.size_G = fact(self.L)
        self.qubit_perms = {id: perm for id, perm in enumerate(permutations(range(self.L)))}
        self.perm_index = {perm: id for id, perm in self.qubit_perms.items()}
        self.cayley_table = {(i, j): self.perm_index[_perm_prod(pi, pj)]
                for i, pi in self.qubit_perms.items()
                for j, pj in self.qubit_perms.items()}
        self.inverse_table = {i: self.perm_index[_perm_inverse(pi)]
                for i, pi in self.qubit_perms.items()}

        # The layer weights are a `lifted` set of tensors os shape (input_size, out_size).
        # A different weight tensor is associated to every group member.
        #
        # Note that the weights object field must be named `weight`! This is due to the
        # fact that there is a nasty dependency with the base policy object. The `device`
        # property of policy objects is defined as: device = self.output_layer.weight.device.
        # If the Equivariant layer is the only layer of our model, that is if we have a
        # linear model, then it itself is the output layer.
        #
        self.weight = nn.Parameter(torch.empty(self.size_G, self.input_size, out_size))
        self.bias = nn.Parameter(torch.empty(self.size_G, out_size))

        # Initialize layer weights.
        # A transposed view of the weights is passed as an argument to `kaiming_uniform_`,
        # because of the way `fan_in` and `fan_out` are calculated in PyTorch - `fan_in`
        # is weight.shape[0], `fan_out` is weight.shape[1]
        nn.init.kaiming_uniform_(self.weight, mode='fan_out')
        nn.init.uniform_(self.bias, -0.01, 0.01)

    def forward(self, x, u):
        """
        """
        assert u in range(self.size_G)
        b, t, q = x.shape

        # Perform a convolution over the group elements.
        # (f*w)(u) = Sum_g f (ug-1) * w(g) (formula 10.6)
        # According to the formula we have to permute the input with the permutation ug-1
        # and permute the weights with the permutation g.
        # Instead of permuting w, however, we are associating a different w to every g.
        # Finally, the outputs of the layer are summed over all group elements.
        # with torch.no_grad():
        #     x_lift = torch.stack([_qubit_perm_apply(x, g)
        #         for g in self.qubit_perms.values()])
        #     perms_to_consider = [self.cayley_table[u, self.inverse_table[g]]
        #         for g in self.qubit_perms.keys()]

        # out = x_lift[perms_to_consider] @ self.weight.unsqueeze(dim=1)   # shape = (size_G, b, t, h)
        # return out.mean(dim=0)

        result = torch.zeros(size=(b, t, self.out_size), device=self.weight.device)
        for g in self.qubit_perms.keys():
            ug_inv = self._prod_with_inverse(u, g)
            x_p = _qubit_perm_apply(x, ug_inv)
            w_p, b_p = self.weight[g], self.bias[g]
            out = x_p @ w_p + b_p
            result += out
        return result / self.size_G

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