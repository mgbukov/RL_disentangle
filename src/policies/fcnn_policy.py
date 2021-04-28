import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .base_policy import BasePolicy


class FCNNPolicy(nn.Module, BasePolicy):
    """ A policy function parametrized by a fully-connected neural network.
    The model architecture uses fully-connected layers. Every hidden layer is
    followed by a batch normalization layer.
    After applying the non-linearity a dropout layer is applied.

    For a network with L layers the architecture will be:
    {affine - [BatchNorm] - leaky-ReLU - [dropout]} x (L - 1) - affine

    The weights of the layers are initialized using Kaiming He uniform distribution.
    """

    def __init__(self, input_size, hidden_sizes, out_size, dropout_rate=0.0, batchnorm=False):
        """ Initialize a policy model.

        @param input_size (int): Size of the environment state.
        @param hidden_sizes (List[int]): A list of sizes for the hidden layers.
        @param out_size (int): Number of possible actions the agent can choose from.
        @param dropout_rate (float): Dropout probability.
        @param batchnorm (bool): If true, then use batch normalization layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm

        # Store arguments for model initialization.
        # Args dict is used to save and restore the model.
        self.args = dict(input_size=self.input_size,
                         hidden_sizes=self.hidden_sizes,
                         out_size=self.out_size,
                         dropout_rate=self.dropout_rate,
                         batchnorm=self.batchnorm)

        # Initialize the model architecture.
        self.num_layers = len(hidden_sizes)
        self.hidden_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        fan_in = input_size
        for fan_out in hidden_sizes:
            self.hidden_layers.append(nn.Linear(fan_in, fan_out))
            self.batchnorm_layers.append(nn.BatchNorm1d(fan_out))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            fan_in = fan_out
        self.output_layer = nn.Linear(fan_out, out_size)

        # Initialize model parameters.
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_uniform_(param)         # weight
            else:
                nn.init.uniform_(param, -0.01, 0.01)    # bias


    def forward(self, x):
        """ Take a mini-batch of environment states and compute scores over
        the possible actions.

        @param x (Tensor): Tensor of shape (b, q), giving the current state of
                the environment, where b = batch size,
                q = size of the quantum system (2 ** num_qubits).
        @return out (Tensor): Tensor of shape (batch_size, num_actions), giving
                a score to every action from the action space.
        """
        out = x
        for idx in range(self.num_layers):
            out = self.hidden_layers[idx](out)
            if self.batchnorm:
                out = self.batchnorm_layers[idx](out)
            out = F.leaky_relu(out)
            out = self.dropout_layers[idx](out)
        out = self.output_layer(out)
        return out

#