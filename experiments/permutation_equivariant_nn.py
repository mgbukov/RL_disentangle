import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from permutation import Permutation
from typing import List


class PEFullyConnected(nn.Module):

    def __init__(self, input_dim :int, hidden_dims :List[int]):
        super().__init__()
        # Asssert that the input is a qubits system
        assert (input_dim & (input_dim - 1)) == 0
        self._input_dim = input_dim
        self._n_qubits = int(np.log2(input_dim))
        self.Y = list(itertools.combinations(range(1, self._n_qubits+1), 2))
        self.G = list(Permutation().group(self._n_qubits))
        self.G_inverse = [g.inverse() for g in self.G]
        self.G_size = len(self.G)
        self.G_toint = {k: v for v, k in enumerate(self.G)}
        self.G_act_Y = GroupActionY(self.G, self.Y, self._n_qubits)
        # Compute stabilizer
        self.y0 = (1, 2)
        self.stabilizer = []
        for g in self.G:
            gy = self.G_act_Y(g, self.y0)
            if gy == self.y0:
                self.stabilizer.append(g)
        # Initialize weights
        weights = []
        layer_dims = [input_dim] + hidden_dims + [1]
        for in_dim, out_dim in zip(layer_dims, layer_dims[1:]):
            W = torch.zeros(size=(self.G_size, in_dim, out_dim), requires_grad=True)
            # A transposed view of W is passed as an argument to
            # `kaiming_uniform_`, because of the way `fan_in` and `fan_out` are
            # calculated in PyTorch - `fan_in` is W.shape[0], `fan_out` is
            # W.shape[1]
            nn.init.kaiming_uniform_(W.view(in_dim, out_dim, self.G_size), mode='fan_out')
            weights.append(nn.Parameter(W))
        self.weights = nn.ParameterList(weights)

    def num_parameters(self):
        return sum(np.multiply.reduce(w.shape) for w in self.weights)

    def _f_weights(self, W, g):
        if isinstance(g, Iterable):
            return W[[self.G_toint[x] for x in g]]
        else:
            return W[self.G_toint[g]]

    def _f_data(self, X_lift, g):
        if isinstance(g, Iterable):
            return X_lift[:, [self.G_toint[x] for x in g]]
        else:
            return X_lift[:, self.G_toint[g]]

    def forward(self, x):
        result = []
        with torch.no_grad():
            # Lift `x` on G
            shape = (-1,) + (2,) * self._n_qubits
            lifted = _lift(x.view(shape), self.G)
        for y in self.Y:
            p = self.project(lifted, y)
            result.append(p.view(-1))
        return torch.stack(result, dim=1)

    def project(self, x, y):
        assert y in self.Y
        # Find group element from coset
        # TODO : Precompute
        for g in self.G:
            gy = self.G_act_Y(g, self.y0)
            if gy == y:
                break
        assert y == gy
        # Project
        result = []
        # TODO : Precompute g * s
        for u in (g * s for s in self.stabilizer):
            result.append(self._G_convolution(x, u))
        result = torch.stack(result, dim=1)
        return result.mean(dim=1)

    def _G_convolution(self, X: torch.tensor, u: Permutation):
        assert u in self.G
        with torch.no_grad():
            _prod = [u * invg for invg in self.G_inverse]
            # Lift X on G
            # shape = (-1,) + (2,) * self._n_qubits
            # lifted = _lift(X.view(shape), self.G)
            _input = self._f_data(X, _prod) # (B, |G|, 2,2,2...)
            _input = _input.view(-1, self.G_size, 2 ** self._n_qubits)  # (B, |G|, input_dim)

        result = []
        for g in self.G:
            i = self.G_toint[g]
            x = _input[:, i, ...].squeeze(1)  # (B, input_dim)
            for W in self.weights:
                omega = self._f_weights(W, g)
                x = F.leaky_relu(x @ omega)
            result.append(x)
        result = torch.stack(result, dim=1).sum(dim=1)  # (B, |G|, o)
        return result


class GroupActionY:

    def __init__(self, G, Y, k):
        self.G = G
        self.Y = Y
        self.k = k 
        self._action_y_mapping = {}
        for g in G:
            x = g.to_image(k)
            for i, j in Y:
                inew, jnew = x[i-1], x[j-1]
                ynew = (inew, jnew) if inew < jnew else (jnew, inew)
                self._action_y_mapping[g, (i,j)] = ynew

    def __str__(self):
        leadspace = 3 * self.k + 4
        header = 'g \ y'.ljust(leadspace) + \
                 ''.join('{:<8}'.format(str(y)) for y in self.Y) + '\n\n'
        rows = []
        for g in self.G:
            row = [str(g.to_image(self.k)).ljust(leadspace)]
            for y in self.Y:
                gy = self._action_y_mapping[(g, y)]
                row.append('{:<8}'.format(str(gy)))
            rows.append(''.join(row))
        return header + '\n'.join(rows) + '\n'

    def __call__(self, g, y):
        return self._action_y_mapping[(g, y)]


def _lift(X, G):
    """Applies each element from `G` to every element x in `X`"""
    L = len(X.shape) - 1
    shape = (X.shape[0], len(G)) + ((2,) * L)  # (B, |G|, 2,2,...,2)
    result = torch.zeros(shape)
    for i in range(len(X)):
        for j in range(len(G)):
            newdims = tuple(i - 1 for i in G[j].to_image(L))
            result[i, j] = torch.permute(X[i], newdims)
    return result  # shape: (B, |G|, 2,2,...,2)


class PEConvolutionalNet(torch.nn.Module):
    
    def _G_convolution(self):
        pass
            # activations = []
        # x = X
        # for W in self.weights:
        #     shape = (-1,) + (2,) * self._n_qubits
        #     lifted = _lift(x.view(shape), self.G)
        #     _input = self._f_data(lifted, _prod) # (B, |G|, 2,2,2...)
        #     _input = _input.view(-1, self.sizeG, 2 ** self._n_qubits)  # (B, |G|, input_dim)
        #     result = []
        #     activations = []
        #     x = _input
        #     accum = []
        #     for g in self.G:
        #         i = self.Gint[g]
        #         xg = x[:, i, ...]
        #         omega = self._f_weights(W, g)
        #         accum.append(xg @ omega)
        #     accum = torch.stack(accum, dim=1)
        #     accum = torch.sum(accum, dim=1)
        #     x = accum
