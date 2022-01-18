import torch

from src.policies.base_policy import BasePolicy


class UniformPolicy(BasePolicy):
    """A policy function returning equal scores over all actions in the action space."""

    def __init__(self, out_size):
        """Initialize a uniform policy object.

        Args:
            out_size (int): Number of possible actions the agent can choose from.
        """
        self.out_size = out_size

    def __call__(self, x):
        """Take a mini-batch of environment states and return equal socres over the
        possible actions.

        Args:
            x (torch.Tensor): Tensor of shape (b, q), or (b, t, q), giving the current
                state of the environment, where b = batch size, t = number of time steps,
                q = size of the quantum system (2 ** num_qubits).
        
        Returns:
            out (torch.Tensor): Tensor of shape (b, num_actions), or (b, t, num_acts),
                giving a score to every action from the action space.
        """
        shape = x.shape[:-1] + (self.out_size,)
        return torch.ones(shape)

    @property
    def device(self): return torch.device("cpu")

#