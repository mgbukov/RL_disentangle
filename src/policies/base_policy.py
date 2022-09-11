import torch
import torch.nn.functional as F


class BasePolicy:
    """An abstract class implementation of a policy function parametrization.
    The function takes as input the current state of the environment and returns a score
    for every action in the action space.
    """

    def __init__(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @torch.no_grad()
    def get_action(self, states, disallowed=None, greedy=False, beta=1.0):
        """Return the action selected by the policy.
        Using the scores returned by the network compute a boltzmann probability
        distribution over the actions from the action space. Select the next action
        probabilistically, or deterministically returning the action with the highest
        probability.

        Args:
            states (torch.Tensor): Tensor of shape (batch_size, system_size), giving the
                current states of the environment.
            disallowed (torch.Tensor, optional): Tensor of shape (batch_size,), giving the
                disallowed actions of the agent. Default value is None.
            greedy (bool, optional): If true, select the next action deterministically.
                If false, select the next action probabilistically. Default value is False.
            beta (float, optional): Inverse value of the temperature for the boltzmann
                distribution. Default value is 1.0.
        
        Returns:
            acts (torch.Tensor): Tensor of shape (batch_size,), giving actions selected by
                the policy for every states of the batch.
        """
        states = states.to(self.device)
        logits = self(states) * beta
        probs = F.softmax(logits, dim=-1)
        if greedy:
            acts = torch.argmax(probs, dim=1, keepdim=True)
        else:
            acts = torch.multinomial(probs, 1)
        return acts.squeeze(dim=-1)

    @property
    def device(self):
        """str: Determine which device to place the Tensors upon, CPU or GPU."""
        return self.output_layer.weight.device

    @property
    def num_params(self):
        """int: Number of trainable parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def load(cls, model_path):
        """Load the model from a file."""
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        kwargs = params["kwargs"]
        model = cls(**kwargs)
        model.load_state_dict(params["state_dict"])
        return model

    def save(self, path):
        """Save the model to a file."""
        params = {"kwargs": self.kwargs,
                  "state_dict": self.state_dict()
                 }
        torch.save(params, path)

#