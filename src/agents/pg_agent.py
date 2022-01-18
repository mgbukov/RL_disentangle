import sys
import time

import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.infrastructure.logging import log_train_stats, log_test_stats


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

    def new_baseline(self, rewards, masks):
        """Compute the baseline as the average return at timestep t.

        The baseline is usually computed as the mean total return.
            `b = E[sum(r_1, r_2, ..., r_t)]`
        Subtracting the baseline from the total return has the effect of centering the
        return, giving positive values to good trajectories and negative values to bad
        trajectories. However, when using reward-to-go, subtracting the mean total return
        won't have the same effect. The most common choice of baseline is the value
        function V(s_t). An approximation of V(s_t) is computed as the mean reward-to-go.
            `b[i] = E[sum(r_i, r_(i+1), ..., r_T)]`

        Args:
            rewards (torch.Tensor): Tensor of shape (batch_size, steps), containing the
                rewards obtained at every step.
            masks (torch.Tensor): Boolean tensor of shape (batch_size, steps), that masks
                out the part of the trajectory after it has finished.

        Returns:
            baselines (torch.Tensor): Tensor of shape (steps,), giving the baseline term
                for every timestep.
        """
        steps = rewards.shape[-1]
        b = torch.mean(torch.sum(rewards, dim=-1))
        lengths = torch.sum(masks, dim=-1, keepdim=True)
        mod_rewards = masks * (b/lengths).repeat(1, steps)
        return self.sum_to_go(mod_rewards)

    def reward_baseline(self, rewards, masks):
        # When working with a batch of trajectories, only the active trajectories are
        # considered for calculating the mean baseline. The reward-to-go sum of finished
        # trajectories is 0.
        baselines = torch.sum(self.reward_to_go(rewards), dim=0) / torch.maximum(
                        torch.sum(masks, dim=0), torch.Tensor([1]).to(self.policy.device))

        # Additionally, if there is only 1 active trajectory in the batch, then the
        # the baseline for that trajectory should be 0.
        return (torch.sum(masks, dim=0) > 1) * baselines

    def entropy_term(self, logits, actions):
        """Compute the entropy regularization term.
        Check out: https://arxiv.org/pdf/1805.00909.pdf

        Args:
            logits (torch.Tensor): Tensor of shape (batch_size, steps, num_act), giving
                the logits for every action at every time step.
            actions (torch.Tensor): Tensor of shape (b, t), giving the actions selected by
                the policy during rollout.

        Returns:
            entropy_term (torch.Tensor): Tensor of shape (b, t), giving the entropy
                regularization term for every timestep.
        """
        # log_probs = F.log_softmax(logits, dim=-1)
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # ent = log_probs.gather(index=actions.unsqueeze(dim=2), dim=2).squeeze(dim=2)
        ent = - F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        return - 0.5 * ent

    def train(self, num_iter, steps, learning_rate, lr_decay=1.0, clip_grad=10.0, reg=0.0,
              entropy_reg=0.0, log_every=1, test_every=100, stdout=sys.stdout):
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

        # self.env.set_random_states(copy=True)
        # initial_batch = self.env.states
        # np.savetxt(os.path.join(self.log_dir, "initial_batch.txt"), initial_batch.reshape(self.env.batch_size, -1))

        # Start the training loop.
        for i in range(num_iter):
            tic = time.time()

            # Set the initial state and perform policy rollout.
            # self.env.states = initial_batch
            self.env.set_random_states(copy=True)
            states, actions, rewards, masks = self.rollout(steps)

            # Compute the loss.
            logits = self.policy(states)
            exploration_rewards = entropy_reg * self.entropy_term(logits, actions)
            modified_rewards = rewards + exploration_rewards
            q_values = self.reward_to_go(modified_rewards)
            q_values -= self.reward_baseline(modified_rewards, masks)
            nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
            weighted_nll = torch.mul(masks * nll, q_values) # multiplying by masks is important
                                                            # as the q_values are now shifted
            loss = torch.mean(torch.sum(weighted_nll, dim=1))

            # Perform backward pass.
            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.policy.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()

            # Book-keeping.
            self.train_history[i] = {
                "entropy"       : self.env.entropy(),
                "rewards"       : rewards.cpu().numpy(),
                "exploration"   : exploration_rewards.detach().cpu().numpy(),
                "policy_output" : F.softmax(logits[0], dim=-1).detach().cpu().numpy(),
                "loss"          : loss.item(),
                "total_norm"    : total_norm.cpu().numpy(),
                "nsolved"       : sum(self.env.disentangled()),
                "nsteps"        : ((~masks[:,-1])*torch.sum(masks, axis=1)).cpu().numpy(),
            }

            toc = time.time()

            # Log results to file.
            if i % log_every == 0:
                print(f"Iteration ({i}/{num_iter}) took {toc-tic:.3f} seconds.", file=stdout)
                log_train_stats(self.train_history[i], stdout)

            # Test the agent.
            if i % test_every == 0:
                tic = time.time()
                entropies, returns, nsolved = self.test_accuracy(10, steps)#, initial_batch)
                self.test_history[i] = {
                    "entropy" : entropies,
                    "returns" : returns,
                    "nsolved" : nsolved,
                }
                toc = time.time()
                print(f"Iteration {i}\nTesting agent accuracy for {steps} steps...", file=stdout)
                print(f"Testing took {toc-tic:.3f} seconds.", file=stdout)
                log_test_stats((entropies, returns, nsolved), stdout)

#