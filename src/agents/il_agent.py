import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.infrastructure.logging import log_test_stats


class ILAgent(BaseAgent):
    """Imitation learning agent implementation of a reinforcement learning agent.
    The agent uses a dataset of example actions to train a policy network.

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

    def train(self, dataset, num_epochs, batch_size, learning_rate, lr_decay=1.0,
              clip_grad=10.0, reg=0.0, log_every=1, test_every=100, stdout=sys.stdout):
        """Train the agent.

        Args:
            dataset (dict): A dataset dictionary containing:
                states (torch.Tensor): A tensor of shape (N, q), where N is the number of
                    examples in the dataset and q is the shape of the environment state.
                actions (torch.Tensor): A tensor of shape (N,), giving the action to be
                    selected for each state in the dataset.
            num_iter (int): Number of iterations to train the agent for.
            batch_size (int): Batch size parameter for updating the policy network.
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
            stdout (file, optional): File object (stream) used for standard output of
                logging information. Default value is `sys.stdout`.
        """
        # Move the neural network to device and prepare for training.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device("cpu")
        print(f"Using device: {device}\n", file=stdout)
        self.policy.train()
        self.policy = self.policy.to(device)

        # Initialize the optimizer and the scheduler.
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
        print(f"Using optimizer:\n{str(optimizer)}\n", file=stdout)

        data_size, _ = dataset["states"].shape
        # Fit the policy network.
        for i in tqdm(range(num_epochs)):
            tic = time.time()

            # Loop over the entire dataset in random order.
            total_loss, total_grad_norm, j = 0.0, 0.0, 0
            for idxs in torch.randperm(data_size).to(device).split(batch_size):
                # Draw a random mini-batch of samples from the dataset.
                states = dataset["states"][idxs].to(device)
                actions = dataset["actions"][idxs].to(device)

                # Compute the loss.
                logits = self.policy(states)
                loss = F.cross_entropy(logits, actions, reduction="mean")

                # Perform backward pass.
                optimizer.zero_grad()
                loss.backward()
                total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.policy.parameters()]))
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()

                # Bookkeeping.
                total_loss += loss.item()
                total_grad_norm += total_norm
                j += 1

            self.train_history[i] = {
                "loss" : total_loss / j,
            }
            toc = time.time()

            # Log results to file.
            if i % log_every == 0:
                print(f"Epoch ({i}/{num_epochs}) took {toc-tic:.3f} seconds.", file=stdout)
                print(f"  Avg Loss:  {total_loss / j:.5f}", file=stdout)
                print(f"  Avg Grad norm:   {total_grad_norm / j:.5f}", file=stdout)

            # Test the agent.
            if i % test_every == 0:
                tic = time.time()
                steps=30
                entropies, returns, nsolved = self.test_accuracy(100, steps)
                self.test_history[i] = {
                    "entropy" : entropies,
                    "returns" : returns,
                    "nsolved" : nsolved,
                }
                toc = time.time()
                print(f"Epoch {i}\nTesting agent accuracy for {steps} steps...", file=stdout)
                print(f"Testing took {toc-tic:.3f} seconds.", file=stdout)
                log_test_stats((entropies, returns, nsolved), stdout)

#