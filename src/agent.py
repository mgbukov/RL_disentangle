import os
import pickle

import torch
from torch.distributions import Categorical


class PGAgent:
    """A base class implementation of a policy gradient reinforcement learning
    agent. Concrete classes must implement their own update strategies.
    """

    def __init__(self, policy_network, value_network=None, config={}):
        """Init a policy gradient agent.
        Set up the configuration parameters for training the model and initialize
        the optimizers for updating the neural networks.

        Args:
            policy_network: torch.nn Module
            value_network: torch.nn Module, optional
                Value network used for computing the baseline.
            config: dict, optional
                Dictionary with configuration parameters, containing:
                pi_lr: float, optional
                    Learning rate parameter for the policy network. Default: 3e-4
                vf_lr: float, optional
                    Learning rate parameter for the value network. Default: 3e-4
                discount: float, optional
                    Discount factor for future rewards. Default: 1.
                batch_size: int, optional
                    Batch size for iterating over the set of experiences. Default: 128.
                clip_grad: float, optional
                    Threshold for gradient norm clipping. Default: 1.
                entropy_reg: float, optional
                    Entropy regularization factor. Default: 0.
        """
        # The networks should already be moved to device.
        self.policy_network = policy_network
        self.value_network = value_network

        # The training history is a list of dictionaries. At every update step
        # we will write the update stats to a dictionary and we will store that
        # dictionary in this list.
        self.train_history = []

        # Unpack the config parameters to configure the agent for training.
        pi_lr = config.get("pi_lr", 3e-4)
        vf_lr = config.get("vf_lr", 3e-4)
        self.discount = config.get("discount", 1.)
        self.batch_size = config.get("batch_size", 128)
        self.clip_grad = config.get("clip_grad", 1.)
        self.entropy_reg = config.get("entropy_reg", 0.)

        # Initialize the optimizers.
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=pi_lr)
        if self.value_network is not None:
            self.value_optim = torch.optim.Adam(self.value_network.parameters(), lr=vf_lr)

    @torch.no_grad()
    def policy(self, obs):
        self.policy_network.eval()
        return Categorical(logits=self.policy_network(obs))

    @torch.no_grad()
    def value(self, obs):
        self.value_network.eval()
        return self.value_network(obs).squeeze(dim=-1)

    def update(self, obs, acts, rewards, done):
        raise NotImplementedError("This method must be implemented by the subclass")

    def save(self, dir):
        """Save the agent at the provided folder location."""
        os.makedirs(dir, exist_ok=True)
        # Save the agent.
        torch.save(self, os.path.join(dir, "agent.pt"))
        # Save the training history as a pickle file.
        with open(os.path.join(dir, "train_history.pickle"), "wb") as f:
            pickle.dump(self.train_history, f, protocol=pickle.HIGHEST_PROTOCOL)

#