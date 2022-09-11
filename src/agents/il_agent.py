import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.infrastructure.logging import logText, log_test_stats


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

    @torch.no_grad()
    def test_action_accuracy(self, dataset):
        """Test the fraction of times that the correct action was chosen.

        Args:
            dataset (dict): A test dataset containing samples never seen by the
            agent before. The dictionary contains:
                states (torch.Tensor): A tensor of shape (N, q), where N is the number of
                    examples in the dataset and q is the shape of the environment state.
                actions (torch.Tensor): A tensor of shape (N,), giving the action to be
                    selected for each state in the dataset.
        """
        device = self.policy.device
        test_size, _ = dataset["states"].shape
        batch_size = 1024 # iterate the dataset on batches
        predictions, targets = [], []
        with torch.no_grad():
            for i in range(0, test_size, batch_size):
                states = dataset['states'][i:i+batch_size].to(device)
                actions = dataset['actions'][i:i+batch_size].to(device)
                logits = self.policy(states)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(probs, dim=1)
                predictions.append(preds.cpu().numpy())
                targets.append(actions.detach().cpu().numpy())
        predictions = np.hstack(predictions)
        targets = np.hstack(targets)
        accuracy = np.sum(predictions == targets) / len(predictions)
        return accuracy

    def train(self, dataset, num_epochs, batch_size, learning_rate, lr_decay=1.0,
              clip_grad=10.0, reg=0.0, log_every=1, test_every=100, save_every=1,
              log_dir=".", logfile=""):
        """Train the agent.

        Args:
            dataset (dict): A dataset dictionary containing:
                states (torch.Tensor): A tensor of shape (N, q), where N is the number of
                    examples in the dataset and q is the shape of the environment state.
                actions (torch.Tensor): A tensor of shape (N,), giving the action to be
                    selected for each state in the dataset.
            num_epochs (int): Number of epochs to train the model for.
            batch_size (int): Batch size parameter for updating the policy network.
            learning_rate (float): Learning rate for gradient decent.
            lr_decay (float, optional): Multiplicative factor of learning rate decay.
                Default value is 1.0.
            clip_grad (float, optional): Threshold for gradient norm during backpropagation.
                Default value is 10.0.
            reg (float, optional): L2 regularization strength. Default value is 0.0.
            entropy_reg (float, optional): Entropy regularization strength.
                Default value is 0.0.
            log_every (int, optional): Every `log_every` epochs write the results to
                the log file. Default value is 100.
            save_every (int, optional): Every `save_every` epochs save a checkpoint for the
                current weights of the model. Default value is 1.
            log_dir (str, optional): Path to the directory where save checkpoints should be
                stored. Default value is the current directory.
            logfile (str, optional): File path to the file where logging information should
                be written. If empty the logging information is printed to the console.
                Default value is empty string.
        """
        # Move the neural network to device and prepare for training.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device("cpu")
        logText(f"Using device: {device}\n", logfile)
        self.policy.train()
        self.policy = self.policy.to(device)

        # Initialize the optimizer and the scheduler.
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
        logText(f"Using optimizer:\n{str(optimizer)}\n", logfile)

        # Split the dataset into a training set and a test set.
        data_size, _ = dataset["states"].shape
        test_size = data_size // 10
        train_size = data_size - test_size
        test_dataset = {
            "states": dataset["states"][train_size:],
            "actions": dataset["actions"][train_size:],
        }

        # Fit the policy network.
        for i in tqdm(range(num_epochs)):
            tic = time.time()

            # Loop over the entire dataset in random order.
            total_loss, total_grad_norm, total_policy_ent, j = 0.0, 0.0, 0.0, 0
            for idxs in torch.randperm(train_size).to(device).split(batch_size):
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
                probs = F.softmax(logits, dim=-1) + torch.finfo(torch.float32).eps
                total_policy_ent +=-torch.mean(torch.sum(probs*torch.log(probs),dim=-1)).item()
                j += 1

            self.train_history[i] = {
                "policy_loss" : total_loss / j,
                "policy_entropy" : total_policy_ent / j,
            }

            toc = time.time()

            # Log results to file.
            if i % log_every == 0:
                logText(f"Epoch ({i}/{num_epochs}) took {toc-tic:.3f} seconds.", logfile)
                logText(f"  Avg Loss:  {total_loss / j:.5f}", logfile)
                logText(f"  Avg Grad norm:   {total_grad_norm / j:.5f}", logfile)

            # Test the agent.
            if i % test_every == 0:
                tic = time.time()
                steps = 30
                entropies, returns, nsolved, nsteps = self.test_accuracy(1000, steps, greedy=False)
                accuracy = self.test_action_accuracy(test_dataset)
                self.test_history[i] = {
                    "entropies"   : entropies,
                    "returns"   : returns,
                    "nsolved"   : nsolved,
                    "nsteps"    : nsteps,
                    "accuracy"  : accuracy,
                }
                toc = time.time()
                logText(f"Epoch {i}\nTesting agent accuracy for {steps} steps...", logfile)
                logText(f"Testing took {toc-tic:.3f} seconds.", logfile)
                logText(f"    Action classification accuracy: {accuracy:.3f}", logfile)
                log_test_stats(self.test_history[i], logfile)

            # Checkpoint save.
            if i % save_every == 0:
                self.save_policy(log_dir, filename=f"policy_{i}.bin")
                self.save_history(log_dir)

#