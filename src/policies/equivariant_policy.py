from itertools import permutations, product
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


class PermutationLayer(nn.Module):

    def __init__(self, n_inputs, in_features, out_features,
                 hidden_dims=()):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weights_outer = nn.Linear(
            self.in_features ** 2, 128, bias=False, dtype=torch.complex64)
        self.weights_proj = nn.Linear(
            self.in_features, 64, bias=False, dtype=torch.complex64)
        fan_in = 256
        self.hidden_layers = nn.ModuleList()
        for fan_out in hidden_dims:
            layer = nn.Linear(fan_in, fan_out, dtype=torch.complex64)
            self.hidden_layers.append(layer)
            fan_in = fan_out
        self.output_layer = nn.Linear(fan_in, out_features, dtype=torch.complex64)

    def forward(self, inputs):
        assert inputs.ndim == 3
        assert inputs.dtype == torch.complex64
        batch_size = inputs.shape[0]
        assert inputs.shape[1] == self.n_inputs
        assert inputs.shape[2] == self.in_features
        y = {}
        for i,j in product(range(self.n_inputs), range(self.n_inputs)):
            in_A, in_B = inputs[:, i, :], inputs[:, j, :]
            assert in_A.shape == in_B.shape == (batch_size, self.in_features)
            # Calculate the outer product of input pairs
            outer_prod = torch.einsum('Bi,Bj->Bij', in_A, in_B)
            assert outer_prod.shape == \
                (batch_size, self.in_features, self.in_features)
            outer_prod = outer_prod.reshape(batch_size, -1)
            assert outer_prod.shape == (batch_size, self.in_features ** 2)
            # (B, f**2) x (f**2, 128) => (B, 128)
            enc_outer = self.weights_outer(outer_prod)
            assert enc_outer.shape == (batch_size, 128)
            # Calculate embeddings
            # (B, f) x (f, 64) => (B, 64)
            enc_A = self.weights_proj(in_A)
            enc_B = self.weights_proj(in_B)
            assert enc_A.shape == enc_B.shape == (batch_size, 64)
            # Stack outer embedding and projections
            z = torch.hstack([enc_outer, enc_A, enc_B])
            # Apply hidden layers
            for layer in self.hidden_layers:
                z = layer(z)
                z = PermutationLayer.phase_amplitude_relu(z)
            z = self.output_layer(z)
            assert z.shape == (batch_size, self.out_features)
            y.setdefault(i, []).append(z)

        # Average outputs
        result = []
        for k, v in y.items():
            assert len(v) == self.n_inputs
            v_mean = torch.stack(v, dim=0).mean(axis=0)
            assert v_mean.shape == (batch_size, self.out_features)
            result.append(v_mean)
        result = torch.stack(result, dim=1)
        assert result.shape == (batch_size, self.n_inputs, self.out_features)
        return result

    @staticmethod
    def phase_amplitude_relu(z):
        return F.relu(torch.abs(z)) * torch.exp(1j * torch.angle(z))



class PermutationLayer2(nn.Module, BasePolicy):

    def __init__(self, subnet, pooling='mean'):
        super().__init__()
        self.subnet = subnet
        self.pooling = pooling

    def forward(self, inputs):
        assert inputs.ndim == 3
        # inputs.shape == (B, n_inputs, in_features)
        # We transform the input to (B, n_inputs**2, 2 * n_features)
        B, n_inputs, in_features = inputs.shape
        x1 = torch.tile(inputs, (1, n_inputs, 1))
        # z1.shape == (B, n_inputs**2, in_features)
        x2 = torch.tile(inputs, (1, 1, n_inputs)).view(B, -1, in_features)
        # z2.shape == (B, n_inputs**2, in_features)
        assert x1.shape == x2.shape == (B, n_inputs ** 2, in_features)
        x = torch.concatenate([x1, x2], dim=2)
        # x.shape == (B, n_inputs**2, 2 * in_features)
        assert x.shape == (B, n_inputs ** 2, 2 * in_features)
        # Apply subnet
        outputs = self.subnet(x)
        # outputs.shape == (B, n_inputs ** 2, out_features)
        outputs = outputs.view(B, n_inputs, n_inputs, -1)
        if self.pooling == 'mean':
            pooled = torch.mean(outputs, dim=2)
        elif self.pooling == 'max':
            pooled = torch.max(outputs, dim=2).values
        else:
            pooled = self.pooling(outputs, dim=2)
        # `pooled` shape should be (B, n_inputs, out_features)
        assert pooled.shape == (B, n_inputs, pooled.shape[2])
        return pooled


class PEPolicy(nn.Module, BasePolicy):

    def __init__(self, n_inputs, n_features):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.n_features = list(n_features)
        self.pe_layers = nn.ModuleList()
        for in_f, out_f in zip(self.n_features[:-1], self.n_features[1:]):
            pe_layer = PermutationLayer(self.n_inputs, in_f, out_f, (1024, 512, 128))
            self.pe_layers.append(pe_layer)
        self.output_layer = PermutationLayer(
            self.n_inputs, self.n_features[-1], 1, (1024, 512, 128)
        )

    @property
    def device(self):
        return self.output_layer.output_layer.weight.device

    def forward(self, inputs):
        if inputs.shape[1:] != (self.n_inputs, self.n_features[0]):
            oshape = inputs.shape[:-1] + (self.n_inputs,)
            inputs = inputs.reshape(-1, self.n_inputs, self.n_features[0])
        else:
            oshape = (inputs.shape[0], self.n_inputs)
        out = inputs
        for layer in self.pe_layers:
            out = layer(out)
        out = torch.abs(self.output_layer(out)).squeeze(dim=-1)
        out = out.reshape(oshape)
        return out


class ComplexDenseNet(nn.Module):

    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        units = [inputs] + list(hidden) + [outputs]
        self.layers = nn.ModuleList()
        for _in, _out in zip(units[:-1], units[1:]):
            layer = nn.Linear(_in, _out, dtype=torch.complex64)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = ComplexDenseNet.real_imaginary_relu(x)
        x = self.layers[-1](x)
        return x

    @staticmethod
    def phase_amplitude_relu(z):
        return F.relu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))

    @staticmethod
    def real_imaginary_relu(z):
        return F.relu(z.real) + 1.0j * F.relu(z.imag)


class PEPolicy2(nn.Module, BasePolicy):

    def __init__(self, n_inputs, in_features=16, n_hidden=1,
                 hidden_units=(512, 256, 128)):
        
        super().__init__()

        self.n_inputs     = int(n_inputs)
        self.in_features  = int(in_features)
        self.n_hidden     = int(n_hidden)
        self.hidden_units = tuple(hidden_units)
        # Save __init__()'s arguments
        self.kwargs = dict(
            n_inputs     = self.n_inputs,
            in_features  = self.in_features,
            n_hidden     = self.n_hidden,
            hidden_units = self.hidden_units
        )

        self.layers = nn.ModuleList()
        # Initialize hidden layers
        _in = self.in_features
        for i in range(n_hidden):
            _out = hidden_units[-1]
            subnet = ComplexDenseNet(2 * _in, hidden_units[:-1], hidden_units[-1])
            layer = PermutationLayer2(subnet)
            self.layers.append(layer)
            _in = _out
        # Initialize output layer
        subnet = ComplexDenseNet(_in, hidden_units, 1)
        self.output_layer = PermutationLayer2(subnet)


    def forward(self, x):
        # x.shape == (batch_size, n_inputs * in_features)
        x = x.view(-1, self.n_inputs, self.in_features)
        batch_size, n_inputs, _ = x.shape

        for layer in self.layers:
            x = layer(x)
        out = self.output_layer(x)
        assert out.shape == (batch_size, n_inputs, 1)
        return torch.abs(out).square(-1)
