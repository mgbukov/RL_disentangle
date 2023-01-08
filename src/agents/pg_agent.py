from collections import defaultdict
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.infrastructure.logging import logText, log_train_stats, log_test_stats


class PGAgent(BaseAgent):
    """Policy-gradient agent implementation of a reinforcement learning agent.
    The agent uses vanilla policy gradient update to improve its policy.

    Attributes:
        env (QubitsEnvironment): Environment object that the agent interacts with.
        policy (Policy): Policy object that the agent uses to decide on the next action.
        train_history (dict): A dict object used for bookkeeping.
        test_history (dict): A dict object used for bookkeeping.
    """

    def __init__(self, env, policy):
        """Initialize policy gradient agent.

        Args:
            env (QubitsEnvironment object): Environment object.
            policy (Policy object): Policy object.
        """
        self.env = env
        self.policy = policy
        self.train_history = {}
        self.test_history = {}

    def sum_to_go(self, t):
        """Sum-to-go returns the sum of the values starting from the current index. Given
        an array `arr = {a_0, a_1, ..., a_(T-1)}` the sum-to-go is an array `s` such that:
            `s[0] = a_0 + a_1 + ... + a_(T-1)`
            `s[1] = a_1 + ... + a_(T-1)`
            ...
            `s[i] = a_i + a_(i+1) + ... + a_(T-1)`

        Args:
            t (torch.Tensor): Tensor of shape (N1, N2, ..., Nk, steps), where the values
                to be summed are along the last dimension.

        Returns:
            sum_to_go (torch.Tensor): Tensor of shape (N1, N2, ..., Nk, steps)
        """
        return t + torch.sum(t, keepdims=True, dim=-1) - torch.cumsum(t, dim=-1)

    def reward_to_go(self, rewards):
        """Compute the reward-to-go at every timestep t.
        "Don't let the past destract you"
        Taking a step with the gradient pushes up the log-probabilities of each action in
        proportion to the sum of all rewards ever obtained. However, agents should only
        reinforce actions based on rewards obtained after they are taken.
        Check out: https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof1.html

        Args:
            rewards (torch.Tensor): Tensor of shape (batch_size, steps), containing the
                rewards obtained at every step.

        Returns:
            reward_to_go (torch.Tensor): Tensor of shape (batch_size, steps).
        """
        return self.sum_to_go(rewards)

    def reward_baseline(self, returns, mask):
        """Compute the baseline as the average return at timestep t.

        The baseline is usually computed as the mean total return:
            `b = E[sum(r_1, r_2, ..., r_t)]`.
        Subtracting the baseline from the total return has the effect of centering the
        return, giving positive values to good trajectories and negative values to bad
        trajectories. However, when using reward-to-go, subtracting the mean total return
        won't have the same effect. The most common choice of baseline is the value
        function V(s_t). An approximation of the value function at time step `i` is
        computed as the mean reward-to-go:
            `b[i] = E[sum(r_i, r_(i+1), ..., r_T)]`.

        Args:
            returns (torch.Tensor): Tensor of shape (batch_size, steps), containing the
                rewards-to-go obtained at every step.
            mask (torch.Tensor): Boolean tensor of shape (batch_size, steps), that masks
                out the part of the trajectory after it has finished.

        Returns:
            baselines (torch.Tensor): Tensor of shape (batch_size, steps), giving the
                baseline term for every timestep. The tensor is expanded over the first
                dimension to match the shape of the input.
        """
        # When working with a batch of trajectories, only the active trajectories are
        # considered for calculating the mean baseline.
        batch_size, _ = returns.shape
        baseline = torch.sum(mask * returns, dim=0, keepdim=True) / torch.maximum(
            torch.sum(mask, dim=0), torch.Tensor([1]).to(self.policy.device))
        baseline = mask * torch.tile(baseline, dims=(batch_size, 1))
        return baseline

    def entropy_term(self, logits, actions, mask):
        """Compute the entropy regularization term.
        Check out: https://arxiv.org/pdf/1805.00909.pdf

        Args:
            logits (torch.Tensor): Tensor of shape (batch_size, steps, num_act), giving
                the logits for every action at every time step.
            actions (torch.Tensor): Tensor of shape (b, t), giving the actions selected by
                the policy during rollout.
            mask (torch.Tensor): Boolean tensor of shape (batch_size, steps), that masks
                out the part of the trajectory after it has finished.

        Returns:
            entropy_term (torch.Tensor): Tensor of shape (b, t), giving the entropy
                regularization terms for the entire episodes. For every episode the entropy
                of the entire trajectory is copied over all time steps.
        """
        # log_probs = F.log_softmax(logits, dim=-1)
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # step_entropy = log_probs.gather(index=actions.unsqueeze(dim=2), dim=2).squeeze(dim=2)

        # The `cross_entropy` function returns the negative log-likelihood (nll). Taking
        # the negative of the result gives the entropy.
        step_entropy = -F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")

        # The episode entropy is computed as the sum of entropies for the individual steps.
        # The true length of the epsiode is taken into account by masking-out the finished
        # part. The result is a 1D Tensor of shape (b,) giving the entropies for the
        # different trajectories.
        # This tensor is then broadcast into the shape (b, t) and the part of the episodes
        # that is finished is again masked.
        _, steps = actions.shape
        episode_entropy = torch.sum(mask * step_entropy, dim=-1, keepdim=True)
        episode_entropy = mask * torch.tile(episode_entropy, dims=(1, steps))
        return episode_entropy

    def _train(self, num_iter, steps, learning_rate, lr_decay=1.0, clip_grad=10.0, reg=0.0,
              entropy_reg=0.0, log_every=1, test_every=100, save_every=1000000,
              log_dir=".", logfile=""):
        """Train the agent using vanilla policy-gradient algorithm.

        Args:
            num_iter (int): Number of iterations to train the agent for.
            steps (int): Number of steps to rollout the policy for.
            learning_rate (float): Learning rate for gradient decent.
            lr_decay (float, optional): Multiplicative factor of learning rate decay.
                Default value is 1.0.
            clip_grad (float, optional): Threshold for gradient norm during backpropagation.
                Default value is 10.0.
            reg (float, optional): L2 regularization strength. Default value is 0.0.
            entropy_reg (float, optional): Entropy regularization strength.
                Default value is 0.0.
            log_every (int, optional): Every `log_every` iterations write the results to
                the log file. Default value is 100.
            save_every (int, optional): Every `save_every` epochs save a checkpoint for the
                current weights of the model. Default value is 1.
            log_dir (str, optional): Path to the directory where save checkpoints should be
                stored. Default value is the current directory.
            logfile (str, optional): File path to the file where logging information should
                be written. If empty the logging information is printed to the console.
                Default value is empty string.
        """
        optimizer = self.optimizer
        scheduler = self.scheduler

        # Start the training loop.
        for i in tqdm(range(num_iter)):
            tic = time.time()

            # Set the initial state and perform policy rollout.
            self.env.set_random_states()
            states, actions, rewards, done = self.rollout(steps)
            mask = self.generate_mask(done)

            # The shape of the states tensor is (b, steps+1, q). Discard the final states
            # from the trajectories. We have no information about the returns when starting
            # in these states!
            states = states[:, :-1, :]

            # Compute the loss.
            logits = self.policy(states)
            episode_entropy = self.entropy_term(logits, actions, mask)
            q_values = self.reward_to_go(rewards) - 0.5 * entropy_reg * episode_entropy
            q_values -= self.reward_baseline(q_values, mask)
            nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
            weighted_nll = torch.mul(mask * nll, q_values)
            loss = torch.sum(weighted_nll) / torch.sum(mask)

            # Perform backward pass.
            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.policy.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()

            # Compute average policy entropy.
            probs = F.softmax(logits, dim=-1) + torch.finfo(torch.float32).eps
            avg_policy_ent = -(torch.mean(torch.sum(probs*torch.log(probs), dim=-1))
                        / torch.log(torch.Tensor([self.env.num_actions]).to(self.policy.device)))

            # Book-keeping.
            mask_hard = np.any(self.env.entropy() > 0.6, axis=1)
            mask_easy = np.any(~mask.cpu().numpy(), axis=1)
            self.train_history[i] = defaultdict(lambda: np.nan)
            self.train_history[i].update({
                "entropies"         : self.env.entropy(),
                "rewards"           : rewards.cpu().numpy(),
                "exploration"       : episode_entropy[:, 0].detach().cpu().numpy(),
                "policy_entropy"    : avg_policy_ent.item(),
                "policy_loss"       : loss.item(),
                "policy_total_norm" : total_norm.item(),
                "nsolved"           : sum(self.env.disentangled()),
                "nsteps"            : (torch.sum(mask, dim=1)).cpu().numpy(),
                # "easy_states"       : states.detach().cpu().numpy()[mask_easy, 0][:32],
                # "hard_states"       : states.detach().cpu().numpy()[mask_hard, 0][:32],
            })
            toc = time.time()

            # Log results to file.
            if i % log_every == 0:
                logText(f"Iteration ({i}/{num_iter}) took {toc-tic:.3f} seconds.", logfile)
                log_train_stats(self.train_history[i], logfile)

            # Test the agent.
            if i % test_every == 0:
                tic = time.time()
                entropies, returns, nsolved, nsteps = self.test_accuracy(10, steps)
                self.test_history[i] = {
                    "entropies" : entropies,
                    "returns" : returns,
                    "nsolved" : nsolved,
                    "nsteps"  : nsteps,
                }
                toc = time.time()
                logText(f"Iteration {i}\nTesting agent accuracy for {steps} steps...", logfile)
                logText(f"Testing took {toc-tic:.3f} seconds.", logfile)
                log_test_stats(self.test_history[i], logfile)

            # Save checkpoint.
            if i % save_every == 0:
                os.makedirs(os.path.join(log_dir, "policies"), exist_ok=True)
                self.policy.save(os.path.join(log_dir, "policies", f"policy_{i}.bin"))
                torch.save({
                    "optim_state_dict": optimizer.state_dict(),
                    "optim_kwargs": {"lr": learning_rate, "weight_decay": reg},
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scheduler_kwargs": {"step_size": 1, "gamma": lr_decay},
                    "train_kwargs": {
                        "learning_rate" : learning_rate,
                        "lr_decay" : lr_decay,
                        "clip_grad" : clip_grad,
                        "reg" : reg,
                        "entropy_reg" : entropy_reg,
                        "log_every" : log_every,
                        "test_every" : test_every,
                        "save_every" : save_every,
                        "log_dir" : log_dir,
                        "logfile" : logfile,
                    }
                }, os.path.join(log_dir, "policies", f"checkpoint_{i}"))

    def train(self, num_iter, steps, learning_rate, lr_decay=1.0, clip_grad=10.0, reg=0.0,
              entropy_reg=0.0, log_every=1, test_every=100, save_every=1000000,
              log_dir=".", logfile=""):
        # Move the neural network to device and prepare for training.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device("cpu")
        logText(f"Using device: {device}\n", logfile)
        self.policy.train()
        self.policy = self.policy.to(device)

        # Initialize the optimizer and the scheduler.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=reg)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay)
        logText(f"Using optimizer:\n{str(self.optimizer)}\n", logfile)

        # Start training.
        self._train(num_iter, steps, learning_rate, lr_decay, clip_grad, reg, entropy_reg,
            log_every, test_every, save_every, log_dir, logfile)

    def train_from_checkpoint(self, checkpoint_path, num_iter, steps):
        # Move the neural network to device and prepare for training.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device("cpu")
        self.policy.train()
        self.policy = self.policy.to(device)

        # Load the optimizer and scheduler.
        params = torch.load(checkpoint_path)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), **params["optim_kwargs"])
        self.optimizer.load_state_dict(params["optim_state_dict"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **params["scheduler_kwargs"])
        self.scheduler.load_state_dict(params["scheduler_state_dict"])

        # Start training.
        params["train_kwargs"]["num_iter"] = num_iter
        params["train_kwargs"]["steps"] = steps # we maybe want to train with shorter/longer episodes
        self._train(**params["train_kwargs"])

#