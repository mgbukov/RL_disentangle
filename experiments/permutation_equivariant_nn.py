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

    def project(self, x_lifted, y):
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
            result.append(self._G_convolution(x_lifted, u))
        result = torch.stack(result, dim=1)
        return result.mean(dim=1)

    def _G_convolution(self, x_lifted: torch.tensor, u: Permutation):
        assert u in self.G
        with torch.no_grad():
            _prod = [u * invg for invg in self.G_inverse]
            # Lift X on G
            # shape = (-1,) + (2,) * self._n_qubits
            # lifted = _lift(X.view(shape), self.G)
            _input = self._f_data(x_lifted, _prod) # (B, |G|, 2,2,2...)
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


class GroupActionX:

    def __init__(self, G, k):
        self.G = G
        self.k = k
    
    def __call__(self, g, x):
        x = torch.atleast_2d(x)
        oldshape = tuple(x.shape)
        shape = (-1,) + (2,) * self.k
        assert g in self.G
        dims = [0] + list(g.to_image(self.k))
        return torch.permute(x.reshape(shape), dims).reshape(oldshape)


class GroupActionX2:

    def __init__(self, G, k):
        self.G = G
        self.k = k
    
    def __call__(self, g, x):
        x = torch.atleast_2d(x)
        oldshape = tuple(x.shape)
        shape = (-1,) + (2,) * (self.k + 1)
        assert g in self.G
        dims = [0, 1] + [i+1 for i in g.to_image(self.k)]
        return torch.permute(x.reshape(shape), dims).reshape(oldshape) 



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


class GConvLayer(nn.Module):

    def __init__(self, input_dim, output_dim, group, group_action):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.G = group
        self.G_size = len(self.G)
        self.G_toint = {k: v for v, k in enumerate(self.G)}
        self.G_action = group_action
        # Initialize weights
        weights = []
        for _ in range(self.G_size):
            W = torch.zeros((input_dim, output_dim), dtype=torch.float32, requires_grad=True)
            nn.init.kaiming_uniform_(W, mode='fan_out')
            weights.append(nn.Parameter(W))
        self.weights = nn.ParameterList(weights)

    def forward(self, x, u):
        assert u in self.G
        result = []
        for g, W in zip(self.G, self.parameters()):
            h = u * g.inverse()
            xg = self.G_action(h, x).reshape(-1, self.input_dim)
            result.append(xg @ W)
        result = F.leaky_relu(torch.stack(result, dim=0).sum(dim=0))
        return result


class PENetwork(nn.Module):

    def __init__(self, input_dim, hidden_dims, group_action_X, group_action_Y):
        super().__init__()
        self.input_dim = input_dim
        self._n_qubits = int(np.log2(input_dim))
        self._vshape = (-1,) + (2,) * self._n_qubits
        self.Y = list(itertools.combinations(range(1, self._n_qubits+1), 2))
        self.G = list(Permutation.group(self._n_qubits))
        self.G_size = len(self.G)
        self.G_toint = {k: v for v, k in enumerate(self.G)}
        self.G_action_X = group_action_X
        self.G_action_Y = group_action_Y
        # Compute stabilizer
        self.y0 = self.Y[0]
        self.stabilizer = []
        for g in self.G:
            gy = group_action_Y(g, self.y0)
            if gy == self.y0:
                self.stabilizer.append(g)
        # Initialize layers
        layers = []
        layer_dims = [input_dim] + hidden_dims + [1]
        it = zip(layer_dims, layer_dims[1:])
        in_dim, out_dim = next(it)
        layers.append(GConvLayer(in_dim, out_dim, self.G, group_action_X))
        for in_dim, out_dim in it:
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
        self.layers = layers

    def num_parameters(self):
        res = sum(np.multiply.reduce(p.shape) for p in self.layers[0].parameters())
        for lay in self.layers[1:]:
            res += np.multiply.reduce(lay.weight.shape)
        return res

    def forward(self, x):
        result = []
        for y in self.Y:
            p = self.project(x, y)
            result.append(p.view(-1))
        return torch.stack(result, dim=1)

    def project(self, x, y):
        assert y in self.Y
        # Find group element from coset
        # TODO : Precompute
        for g in self.G:
            gy = self.G_action_Y(g, self.y0)
            if gy == y: break
        assert y == gy
        # Project
        result = []
        # TODO : Precompute g * s
        for u in (g * s for s in self.stabilizer):
            out = self.layers[0](x.reshape(self._vshape), u)
            for lay in self.layers[1:]:
                out = lay(out)
            result.append(out)
        result = torch.stack(result, dim=1)
        return result.mean(dim=1)
