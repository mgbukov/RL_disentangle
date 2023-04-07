import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected multi-layer perceptron."""

    def __init__(self, in_shape, hidden_sizes, out_size):
        """Init a multi-layer perceptron neural net.

        Args:
            in_shape: list[int]
                Shape of the input. E.g. (3, 32, 32) for images.
            hidden_sizes: list[int]
                List of sizes for the hidden layers of the network.
            out_size: int
                Size of the output.
        """
        super().__init__()
        self.in_size = np.prod(in_shape)
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size

        # Initialize the model architecture.
        layers = []
        sizes = [self.in_size] + hidden_sizes
        layers.append(nn.Flatten())
        for fan_in ,fan_out in zip(sizes[:-1], sizes[1:]):
            layers.extend([
                nn.Linear(fan_in, fan_out),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(fan_out, out_size))
        self.net = nn.Sequential(*layers)

        # Initialize model parameters.
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_uniform_(param)         # weight
            else:
                nn.init.uniform_(param, -0.01, 0.01)    # bias

    def forward(self, x):
        """Forward the input through the neural network.
        The first dimension of the input is considered to be the batch dimension
        and all other dimensions will be flattened.

        Args:
            x: torch.Tensor
                Tensor of shape (B, d1, d2, ..., dk).

        Returns:
            out: torch.Tensor
                Tensor of shape (B, 1), giving the resulting output.
        """
        x = x.float().contiguous().to(self.device)
        return self.net(x)

    @property
    def device(self):
        """str: Determine on which device is the model placed upon, CPU or GPU."""
        return next(self.parameters()).device


class Transformer(nn.Module):
    """Transformer is a stack of transformer encoder layers."""

    def __init__(self, in_shape, embed_dim, hid_dim, out_dim, dim_mlp, n_heads, n_layers):
        """Init a Transformer neural net.

        Args:
            in_shape: tuple(int)
                The expected shape of the inputs to the network.
            embed_dim: int
                The size of the embedding dimension used for self-attention.
            hid_dim: int
                The size of the final fully-connected layer.
            out_dim: int
                The size of the output.
            dim_mlp: int
                The size of the hidden layers used in the encoder layers.
            n_heads: int
                Number of attention heads.
            n_layers: int
                Number of Transformer encoder layers.
        """
        super().__init__()
        T, in_dim = in_shape

        # Define the stack of encoder layers. Note that by default the encoder
        # layer uses dropout=0.1.
        self.embed = nn.Linear(in_dim, embed_dim, bias=False)
        self.encoder = nn.Sequential(
            *[nn.TransformerEncoderLayer(embed_dim, n_heads, dim_mlp, batch_first=True)
                for _ in range(n_layers)],
        )

        # The output of the encoder layers will be flattened and projected
        # through a fully-connected layer.
        self.hid_proj = nn.Linear(T * embed_dim, hid_dim)
        self.out_proj = nn.Linear(hid_dim, out_dim)

        # Initialize the weights of the embedding and projection layers.
        # For the encoder layers: MHA is initialized with xavier normal and MLP
        # is initialized with xavier uniform by default.
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.hid_proj.weight)
        nn.init.zeros_(self.hid_proj.bias)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """Forward the input through the Transformer network.

        Args:
            x: torch.Tensor
                Tensor of shape (B, T, d), giving a batch of sequential input
                data.

        Returns:
            out: torch.Tensor
                Tensor of shape (B, T*(T-1)).
        """
        x = x.float().contiguous().to(self.device)

        # Encode the inputs with the transformer encoders.
        z = self.embed(x)
        z = self.encoder(z)

        # Flatten and pass through the fully-connected layer.
        z = torch.flatten(z, start_dim=1)
        out = self.out_proj(torch.relu(self.hid_proj(z)))

        return out

    @property
    def device(self):
        """str: Determine on which device is the model placed upon, CPU or GPU."""
        return next(self.parameters()).device


class TransformerPE(nn.Module):
    """Permutation-equivariant Transformer net."""

    def __init__(self, in_dim, embed_dim, dim_mlp, n_heads, n_layers):
        super().__init__()
        self.n_heads = n_heads

        # Define a stack of transformer encoder layers.
        self.embed = nn.Linear(in_dim, embed_dim, bias=False)
        self.encoder = nn.Sequential(
            *[nn.TransformerEncoderLayer(embed_dim, n_heads, dim_mlp, batch_first=True)
                for _ in range(n_layers)],
        )

        # The final output projection will be the self-attention scores.
        self.qk = nn.Linear(embed_dim, 2 * embed_dim)
        nn.init.normal_(self.qk.weight, std=np.sqrt(2 / (embed_dim + embed_dim//n_heads)))
        nn.init.zeros_(self.qk.bias)

    def forward(self, x):
        B, T, _ = x.shape
        nh = self.n_heads
        x = x.float().contiguous().to(self.device)

        # Encode the inputs with the transformer encoders.
        z = self.embed(x)
        z = self.encoder(z)

        # Compute the attention scores from the encoder output embeddings.
        queries, keys = self.qk(z).chunk(chunks=2, dim=-1)
        queries = queries.view(B, T, nh, -1).transpose(1, 2)
        keys = keys.view(B, T, nh, -1).transpose(1, 2)
        attn = torch.matmul(queries, keys.transpose(2, 3)) # XV @ (XK)^T and shape (B, nh, T, T)
        d_k = keys.shape[-1]
        attn /= np.sqrt(d_k)

        # Exclude the diagonal entries from the attention scores matrix.
        out = attn.reshape(B, nh, -1)[:, :, 1:].view(B, nh, T-1, T+1)[:, :, :, :-1].reshape(B, nh, -1)
        out = out.mean(dim=1) # average over the heads
        return out

    @property
    def device(self):
        """str: Determine on which device is the model placed upon, CPU or GPU."""
        return next(self.parameters()).device

#