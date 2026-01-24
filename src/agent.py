import os
import warnings
from typing import *

import torch
from torch.distributions import Categorical


class PGAgent:
    """A base class implementation of a policy gradient reinforcement learning
    agent. Concrete classes must implement their own update strategies.
    """

    def __init__(self, policy_network: torch.nn.Module,
                 value_network: Optional[torch.nn.Module] = None, config: dict = {}):
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
            tracker: optional
                Instance that will keep track of the training stats.
                Must provide `add_scalar()` method.
        """
        # The networks should already be moved to device.
        self.policy_network = policy_network
        self.value_network = value_network

        # The training history is a list of dictionaries. At every update step
        # we will write the update stats to a dictionary and we will store that
        # dictionary in this list.
        # self.train_history = []

        # Unpack the config parameters to configure the agent for training.
        self.pi_freeze_iters = config.get("pi_freeze_iters", 0)
        pi_lr = config.get("pi_lr", 3e-4)
        pi_lr_milestones = config.get("pi_lr_milestones", None)
        pi_lr_gamma = config.get("pi_lr_gamma", None)

        self.vf_freeze_iters = config.get("vf_freeze_iters", 0)
        vf_lr = config.get("vf_lr", 3e-4)
        vf_lr_milestones = config.get("vf_lr_milestones", None)
        vf_lr_gamma = config.get("vf_lr_gamma", None)
        vf_warmup_iters = config.get("vf_warmup_iters", 0)

        self.discount = config.get("discount", 1.)
        self.batch_size = config.get("batch_size", 128)
        self.clip_grad = config.get("clip_grad", 1.)
        self.entropy_reg = config.get("entropy_reg", 0.)

        # Initialize the optimizers.
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=pi_lr)
        if self.value_network is not None:
            self.value_optim = torch.optim.Adam(self.value_network.parameters(), lr=vf_lr)

        # Initialize LR schedulers
        # * Policy function scheduler
        if pi_lr_milestones is not None and pi_lr_gamma is not None:
            self.pi_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.policy_optim,
                pi_lr_milestones,
                pi_lr_gamma
            )
        else:
            self.pi_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.policy_optim, factor=1.0,
            )
        # * Value function base scheduler
        if vf_lr_milestones is not None and vf_lr_gamma is not None:
            vf_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.value_optim,
                vf_lr_milestones,
                vf_lr_gamma
            )
        else:
            vf_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.value_optim, factor=1.0
            )
        if vf_warmup_iters > 0:
            # * Value function warmup scheduler
            vf_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.value_optim,
                start_factor=1e-2,
                total_iters=vf_warmup_iters
            )
            self.vf_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                [vf_warmup_scheduler, vf_scheduler],
                milestones=[vf_warmup_iters]
            )
        else:
            self.vf_lr_scheduler = vf_scheduler

    @torch.no_grad()
    def policy(self, obs) -> Categorical:
        self.policy_network.eval()
        return Categorical(logits=self.policy_network(obs))

    @torch.no_grad()
    def value(self, obs):
        self.value_network.eval()
        return self.value_network(obs).squeeze(dim=-1)

    def update(self, obs, acts, rewards, done):
        raise NotImplementedError("This method must be implemented by the subclass")

    def save(self, dir, increment=None):
        """Save the agent at the provided folder location."""
        os.makedirs(dir, exist_ok=True)

        basename = f"agent{increment}" if increment else "agent"
        # Save the complete agent ( Non-Portable !!! )
        torch.save(self, os.path.join(dir, basename + '.pt'))

    def state_dict(self):
        return {
            "policy_fn":    self.policy_network.state_dict(),
            "value_fn":     self.value_network.state_dict() \
                                if self.value_network is not None else {},
            "policy_optim": self.policy_optim.state_dict(),
            "value_optim":  self.value_optim.state_dict() \
                                if self.value_network is not None else {},
            "pi_lr_scheduler": self.pi_lr_scheduler,
            "vf_lr_scheduler": self.vf_lr_scheduler,
            "discount":     self.discount,
            "batch_size":   self.batch_size,
            "clip_grad":    self.clip_grad,
            "entropy_reg":  self.entropy_reg
        }

    def load_state_dict(self, state_dict: dict):
        self.policy_network.load_state_dict(state_dict["policy_fn"])
        self.policy_optim.load_state_dict(state_dict["policy_optim"])
        if self.value_network is not None:
            self.value_network.load_state_dict(state_dict["value_fn"])
            self.value_optim.load_state_dict(state_dict["value_optim"])
        try:
            self.pi_lr_scheduler.load_state_dict(state_dict["pi_lr_scheduler"])
        except:
            warnings.warn("Cannot load LR scheduler state for policy function")
        try:
            self.vf_lr_scheduler.load_state_dict(state_dict["vf_lr_scheduler"])
        except:
            warnings.warn("Cannot load LR scheduler state for value function.")
        self.discount = state_dict["discount"]
        self.batch_size = state_dict["batch_size"]
        self.clip_grad = state_dict["clip_grad"]
        self.entropy_reg = state_dict["entropy_reg"]


class RandomAgent:

    def __init__(self, num_actions):
        self.num_actions = int(num_actions)

    def policy(self, obs):
        batch_size = len(obs)
        probs = torch.zeros((batch_size, self.num_actions), dtype=torch.float32)
        a = torch.randint(0, self.num_actions, (batch_size,))
        probs[torch.arange(batch_size), a] = 1.0
        return Categorical(probs=probs)