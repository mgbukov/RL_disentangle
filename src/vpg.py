import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions import Categorical

from src.agent import PGAgent


class VPGAgent(PGAgent):
    """Vanilla policy gradient implementation of a reinforcement learning agent.
    The updates for the policy network are computed using sample episodes
    generated from simulations. A Monte-carlo estimate of the gradients is
    computed and a single policy update step is performed before the experiences
    are discarded.
    """

    def update(self, obs, acts, rewards, done):
        """Update the agent policy network using the provided experiences.
        If the agent uses a value network, then it will also be updated.

        Args:
            obs: torch.Tensor
                Tensor of shape (N, T, *), giving the observations produced by
                the agent during multiple roll-outs.
            acts: torch.Tensor
                Tensor of shape (N, T), giving the actions selected by the agent.
            rewards: torch.Tensor
                Tensor of shape (N, T), giving the obtained rewards.
            done: torch.Tensor
                Boolean tensor of shape (N, T), indicating which of the
                observations are terminal states for the environment.
        """
        # Extend the training history with a dict of statistics.
        self.train_history.append({})
        N, T = rewards.shape
        returns = torch.zeros_like(rewards)

        # Deal with unfinished episodes; either bootstrap or mask.
        mask = torch.ones_like(done) # no masking
        if self.value_network is not None:
            # Bootstrap the last reward of unfinished episodes.
            bootstrap = self.value(obs[:, -1]).to(rewards.device) # uses torch.no_grad
            returns[:, -1] = torch.where(done[:, -1], rewards[:, -1], bootstrap)
            mask[:, -1] = True # if bootstrap then mark as finished
        else:
            returns[:, -1] = torch.where(done[:, -1], rewards[:, -1], 0.)
            mask[:, -1] = done[:, -1]

        # Compute the discounted returns.
        for t in range(T-2, -1, -1): # O(T)  \_("/)_/
            returns[:, t] = rewards[:, t] + self.discount * returns[:, t+1] * ~done[:, t]

            # Maybe mask unfinished episodes
            mask[:, t] = mask[:, t+1] | done[:, t]
            returns[:, t] *= mask[:, t]

        # Reshape the inputs for the neural networks. Maybe discard masked experiences.
        obs = obs.reshape(N*T, *obs.shape[2:])[mask.ravel()]
        acts = acts.reshape(N*T)[mask.ravel()]
        returns = returns.reshape(N*T)[mask.ravel()]

        # Maybe update the value network and baseline the returns.
        if self.value_network is not None:
            self.update_value(obs, returns)
            baseline = self.value(obs).to(returns.device) # uses torch.no_grad
            returns -= baseline

        # Update the policy network.
        self.update_policy(obs, acts, returns)

    def update_policy(self, obs, acts, returns):
        """Perform one gradient update step on the policy network.

        Args:
            obs: torch.Tensor
                Tensor of shape (N, *), giving the observations produced by the
                agent during rollout.
            acts: torch.Tensor
                Tensor of shape (N,), giving the actions selected by the agent.
            returns: torch.Tensor
                Tensor of shape (N,), giving the obtained returns.
        """
        # Forward pass.
        self.policy_network.train()
        logits = self.policy_network(obs)
        logp = F.cross_entropy(logits, acts.to(logits.device), reduction="none")

        # Normalize the returns and compute the pseudo-loss.
        eps = torch.finfo(torch.float32).eps
        returns = (returns - returns.mean()) / (returns.std() + eps)
        returns = returns.to(logp.device)
        pi_loss = torch.mean(logp * returns)

        # Add entropy regularization. Augment the loss with the mean entropy of
        # the policy calculated over the sampled observations.
        avg_policy_ent = Categorical(logits=logits).entropy().mean(dim=-1)
        total_loss = pi_loss - self.entropy_reg * avg_policy_ent

        # Backward pass.
        self.policy_optim.zero_grad()
        total_loss.backward()
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.clip_grad)
        self.policy_optim.step()

        # Store the stats.
        self.train_history[-1].update({
            "policy_loss"            : pi_loss.item(),
            "total_loss"             : total_loss.item(),
            "policy_entropy"         : avg_policy_ent.item(),
            "policy_total_grad_norm" : total_norm.item(),
        })

    def update_value(self, obs, returns):
        """Update the value network to fit the value function of the current
        policy `V_pi`. This functions performs a single iteration over the
        set of experiences drawing mini-batches of examples and fits the value
        network using MSE loss.

        Args:
            obs: torch.Tensor
                Tensor of shape (N, *), giving the observations of the agent.
            returns: torch.Tensor
                Tensor of shape (N,), giving the obtained returns.
        """
        # Create a dataloader object for iterating through the examples.
        returns = returns.reshape(-1, 1) # match the output shape of the net (B, 1)
        dataset = data.TensorDataset(obs, returns)
        train_dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Iterate over the collected experiences and update the value network.
        self.value_network.train()
        total_loss, total_norm, j = 0., 0., 0
        for o, r in train_dataloader:
            # Forward pass.
            pred = self.value_network(o)
            vf_loss = F.mse_loss(pred, r.to(pred.device))
            # Backward pass.
            self.value_optim.zero_grad()
            vf_loss.backward()
            grad_norm = torch.norm(torch.stack(
                [torch.norm(p.grad) for p in self.value_network.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.clip_grad)
            self.value_optim.step()

            # Bookkeeping.
            total_loss += vf_loss.item() * o.shape[0]
            total_norm += grad_norm.item() * o.shape[0]
            j += o.shape[0]

        # Store the stats.
        self.train_history[-1].update({
            "value_avg_loss"        : total_loss / j,
            "value_avg_grad_norm"   : total_norm / j,
        })

#