import time

import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from utils.logger import FileLogger


class PGAgent(BaseAgent):
    """ Policy-gradient agent implementation of a reinforcement learning agent.
    The agent uses vanilla policy gradient update to improve its policy.
    """

    def __init__(self, env, policy):
        """ Initialize policy gradient agent.

        @param env (QubitsEnvironment object): Environment object.
        @param policy (Policy object): Policy object.
        """
        self.env = env
        self.policy = policy
        self.train_history = {}
        self.logger = FileLogger("logs", "log_final.txt")


    @torch.no_grad()
    def reward_to_go(self, rewards):
        return rewards + torch.sum(rewards, keepdims=True, dim=1) - torch.cumsum(rewards,dim=1)


    @torch.no_grad()
    def reward_baseline(self, rewards, done):
        """ Compute the baseline as the average return at time-step t.

        @param rewards (torch.Tensor): Tensor of shape (batch_size, steps),
                containing the rewards obtained at every step.
        @param done (torch.Tensor): Tensor of shape (batch_size, steps).
        """
        return torch.mean(self.reward_to_go(rewards), dim=0, keepdim=True)


    def train(self, num_episodes, steps, learning_rate, lr_decay=1.0, reg=0.0,
              log_every=1, verbose=False):
        """ Train the agent using vanilla policy-gradient algorithm.

        @param num_episodes (int): Number of episodes to train the agent for.
        @param steps (int): Number of steps to rollout the policy for.
        @param learning_rate (float): Learning rate for gradient decent.
        @param lr_decay (float): Multiplicative factor of learning rate decay.
        @param reg (float): L2 regularization strength.
        @param log_every (int): Every @log_every iterations write the results
                to the log file.
        @param verbose (bool): If true, prinout logging information.
        """
        self.policy.train()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.logger.verboseLogging(verbose)
        self.logger.logTxt("Using device: {}".format(device))
        self.policy = self.policy.to(device)

        # Initialize the optimizer and the scheduler.
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=lr_decay)

        trajects = self.env.batch_size   # number of trajectories
        for i in range(num_episodes):
            tic = time.time()

            # Perform policy roll-out.
            self.env.set_random_state()
            states, actions, rewards, done = self.rollout(steps)

            # Compute the loss.
            q_values = self.reward_to_go(rewards) - self.reward_baseline(rewards, done)
            logits = self.policy(states)
            logits = self.policy.mask_logits(logits, actions)
            nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
            weighted_nll = torch.mul(nll, q_values)
            loss = torch.mean(torch.sum(weighted_nll, dim=1))

            # Perform backward pass.
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Book-keeping.
            entropies = self.env.entropy()
            self.train_history[i] = {
                "entropy": (entropies.min(), entropies.max(), entropies.mean(), entropies.std()),
                "states" : states.cpu().numpy(),
                "loss" : loss.item(),
                "nsteps" : np.argmax(done.cpu().numpy(), axis=1) + 1
            }

            toc = time.time()

            # Log results to file.
            if (i + 1) % log_every == 0:
                self.logger.logTxt("Episode ({}/{}) took {:.3f} seconds.".format(
                    i + 1, num_episodes, (toc-tic)))
                self.logger.logTxt("  Mean final reward: {:.4f}".format(
                    torch.mean(rewards[:,-1])))
                self.logger.logTxt("  Mean return: {:.4f}".format(
                    torch.mean(torch.sum(rewards, dim=1), dim=0)))
                self.logger.logTxt("  Mean final entropy: {:.4f}".format(entropies.mean()))
                self.logger.logTxt("  Max final entropy: {:.4f}".format(entropies.max()))
                self.logger.logTxt("  Pseudo loss: {:.5f}".format(loss))
                self.logger.logTxt("  Solved trajectories: {} / {}".format(
                    sum(done[:, -1]).item(), trajects))
                self.logger.logTxt("  Avg steps to disentangle: {:.3f}".format(
                    (np.argmax(done.cpu().numpy(), axis=1) + 1).mean()))

#