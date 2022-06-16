from collections import defaultdict
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.infrastructure.logging import logText, log_train_stats, log_test_stats


class ACAgent(BaseAgent):
    """Actor-Critic agent implementation of a reinforcement learning agent.
    This is an implementation of an online advantage actor critic algorithm.
    The agent uses the current observations to update its value network.
    After that it uses the value network and the bootstrap formula to update its
    policy.

    Attributes:
        env (QubitsEnvironment): Environment object that the agent interacts with.
        policy_network (NN object): A neural network that the agent uses as a policy.
        value_network (NN object): A neural network that the agent uses to score
            the environment states.
        train_history (dict): A dict object used for bookkeeping.
        test_history (dict): A dict object used for bookkeeping.
    """

    def __init__(self, env, policy_network, value_network, discount=1.):
        """Initialize policy gradient agent.

        Args:
            env (QubitsEnvironment object): Environment object.
            policy_network (NN object): Policy network object.
            value_network (NN object): Value network object.
        """
        self.env = env
        self.policy = policy_network
        self.value_network = value_network
        self.discount = discount
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

    def train(self, num_iter, steps, policy_lr, value_lr, batch_size, clip_grad=10.0,
        policy_reg=0.0, value_reg=0.0, log_every=1, test_every=100, save_every=100,
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
            logfile (str, optional): File path to the file where logging information should
                be written. If empty the logging information is printed to the console.
                Default value is empty string.
        """
        discount = self.discount
        # Move the neural network to device and prepare for training.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device("cpu")
        logText(f"Using device: {device}\n", logfile)
        self.policy.train()
        self.policy = self.policy.to(device)
        self.value_network.train()
        self.value_network = self.value_network.to(device)

        # Initialize the optimizers and the schedulers.
        policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=policy_lr, weight_decay=policy_reg)
        value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=value_lr, weight_decay=value_reg)
        logText(f"Using optimizers:\nPolicy optimizer:{str(policy_optimizer)}\n"+
            f"Value optimizer:{str(value_optimizer)}", logfile)

        # Start the training loop.
        for i in tqdm(range(num_iter)):
            tic = time.time()

            # Set the initial state and perform policy rollout.
            self.env.set_random_states()
            states, actions, rewards, done = self.rollout(steps)
            mask = self.generate_mask(done)

            # Fit the value network.
            # This is bad! There must be a way to draw random samples without reshaping.
            # Also this reshaping does not take into account that there might be trajectories
            # that have finished before reaching the maximum number of steps.
            #
            # system_size = states.shape[-1]
            # prev_obs = states[:,  :-1, :].reshape(-1, system_size)
            # next_obs = states[:, 1:  , :].reshape(-1, system_size)
            # rwrd = rewards[:, -1].reshape(-1)
            #
            # A correct but not very good solution.
            prev_obs, next_obs, rwrd, ddone = [], [], [], []
            for idx, traject in enumerate(states):
                length = torch.sum(mask[idx]).item()
                prev_obs.append(traject[:length-1])
                next_obs.append(traject[1:length])
                rwrd.append(rewards[idx, :length-1])
                ddone.append(done[idx, :length-1])
            prev_obs = torch.vstack(prev_obs)
            next_obs = torch.vstack(next_obs)
            rwrd = torch.hstack(rwrd)
            done = torch.hstack(ddone)
            #
            # Find a torch implementation of np.random.choice()
            # np.random.choice(np.ndarray, mask) -- works out perfectly!

            total_loss, total_grad_norm, j = 0.0, 0.0, 0
            for idxs in torch.randperm(len(prev_obs)).to(device).split(batch_size):
                # Compute the targets for the value network using one-step bootstrapping.
                # Values of terminal states are set to 0.
                prev_values = self.value_network(prev_obs[idxs]).squeeze(dim=-1)
                next_values = self.value_network(next_obs[idxs]).squeeze(dim=-1) * ~done[idxs]
                targets = rwrd[idxs] + discount * next_values

                # Compute the loss for the value network.
                value_loss = 0.5 * F.mse_loss(prev_values, targets, reduction="mean")

                # Perform backward pass for the value network.
                value_optimizer.zero_grad()
                value_loss.backward()
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad) for p in self.value_network.parameters()]))
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), clip_grad)
                value_optimizer.step()

                total_loss += value_loss
                total_grad_norm += total_norm
                j += 1

            # Update the policy network.
            # Compute the advantages using the critic,
            prev_obs = states[:,  :-1, :]
            next_obs = states[:, 1:  , :]
            prev_values = self.value_network(prev_obs).squeeze(dim=-1)
            next_values = self.value_network(next_obs).squeeze(dim=-1)
            advantages = rewards + discount * next_values - prev_values

            # Compute the loss for the policy gradient.
            logits = self.policy(prev_obs)
            nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
            weighted_nll = torch.mul(mask * nll, advantages)
            loss = torch.sum(weighted_nll) / torch.sum(mask)

            # Perform backward pass.
            policy_optimizer.zero_grad()
            loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.policy.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_grad)
            policy_optimizer.step()

            # Compute average policy entropy.
            probs = F.softmax(logits, dim=-1) + torch.finfo(torch.float32).eps
            avg_policy_ent = -torch.mean(torch.sum(probs*torch.log(probs),axis=-1))

            # Book-keeping.
            mask_hard = np.any(self.env.entropy() > 0.6, axis=1)
            mask_easy = np.any(~mask.cpu().numpy(), axis=1)
            self.train_history[i] = defaultdict(lambda: np.nan)
            self.train_history[i].update({
                "entropy"           : self.env.entropy(),
                "rewards"           : rewards.cpu().numpy(),
                # "exploration"       : episode_entropy[:, 0].detach().cpu().numpy(),
                "value_loss"        : total_loss.item() / j,
                "value_total_norm"  : total_grad_norm.item() / j,
                "policy_entropy"    : avg_policy_ent.item(),
                "policy_loss"       : loss.item(),
                "policy_total_norm" : total_norm.item(),
                "nsolved"           : sum(self.env.disentangled()),
                "nsteps"            : ((~mask[:,-1])*torch.sum(mask, axis=1)).cpu().numpy(),
                "easy_states"       : states.detach().cpu().numpy()[mask_easy, 0][:32],
                "hard_states"       : states.detach().cpu().numpy()[mask_hard, 0][:32],
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

            if i % save_every == 0:
                self.policy.save(os.path.join(log_dir, f"policy_{i}.bin"))
                self.value_network.save(os.path.join(log_dir, f"value_net_{i}.bin"))
                torch.save(policy_optimizer.state_dict, os.path.join(log_dir, f"policy_{i}.optim"))
                torch.save(value_optimizer.state_dict, os.path.join(log_dir, f"value_{i}.optim"))

#