from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.utils.logger import Logger


class PGAgent(BaseAgent):
    """ Policy-gradient agent implementation of a reinforcement learning agent.
    The agent uses vanilla policy gradient update to improve its policy.
    """

    def __init__(self, env, policy, log_dir="logs"):
        """ Initialize policy gradient agent.

        @param env (QubitsEnvironment object): Environment object.
        @param policy (Policy object): Policy object.
        """
        self.env = env
        self.policy = policy
        self.train_history = {}
        self.test_history = defaultdict(lambda : [])
        self.logger = Logger(log_dir)


    def sum_to_go(self, t):
        """ Sum-to-go returns the sum of the values starting from the current
        index. Given an array `arr = {a_0, a_1, ..., a_(T-1)}` the sum-to-go is
        an array `s` such that:
            `s[0] = a_0 + a_1 + ... + a_(T-1)`
            `s[1] = a_1 + ... + a_(T-1)`
            ...
            `s[i] = a_i + a_(i+1) + ... + a_(T-1)`

        @param t (torch.Tensor): Tensor of shape (N1, N2, ..., Nk, steps),
                where the values to be summed are along the last dimension.
        @return sum_to_go (torch.Tensor): Tensor of shape (N1, N2, ..., Nk, steps)
        """
        return t + torch.sum(t, keepdims=True, dim=-1) - torch.cumsum(t, dim=-1)


    def reward_to_go(self, rewards):
        """ Compute the reward-to-go at every timestep t.
        "Don't let the past destract you"
        Taking a step with the gradient pushes up the log-probabilities of each
        action in proportion to the sum of all rewards ever obtained. However,
        agents should only reinforce actions based on rewards obtained after
        they are taken.
        Check out https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof1.html

        @param rewards (torch.Tensor): Tensor of shape (batch_size, steps),
                containing the rewards obtained at every step.
        @return reward_to_go (torch.Tensor): Tensor of shape (batch_size, steps).
        """
        return self.sum_to_go(rewards)


    def reward_baseline(self, rewards, mask=None):
        """ Compute the baseline as the average return at timestep t.

        The baseline is usually computed as the mean total return.
            `b = E[sum(r_1, r_2, ..., r_t)]`
        Subtracting the baseline from the total return has the effect of
        centering the return, giving positive values to good trajectories and
        negative values to bad trajectories.
        However, when using reward-to-go, subtracting the mean total return
        won't have the same effect. The most common choice of baseline is the
        value function V(s_t). An approximation of V(s_t) is computed as the
        mean reward-to-go.
            `b[i] = E[sum(r_i, r_(i+1), ..., r_T)]`

        @param rewards (torch.Tensor): Tensor of shape (batch_size, steps),
                containing the rewards obtained at every step.
        @param mask (torch.Tensor): Boolean tensor of shape (batch_size, steps),
                that masks out the part of the trajectory after it has finished.
        @return baseline (torch.Tensor): Tensor of shape (batch_size, steps),
                giving the baseline term for every timestep.
        """
        if mask is None:
            result = torch.mean(self.reward_to_go(rewards), dim=0, keepdim=True)
        else:
            result = torch.sum(self.reward_to_go(rewards), dim=0) / torch.maximum(torch.sum(mask, dim=0), torch.Tensor([1]).to(self.policy.device))
        return result
        # if mask is None:
        #     mask = torch.ones_like(rewards)
        # return torch.sum(self.reward_to_go(rewards), dim=0) / torch.sum(mask, dim=0)


    def entropy_term(self, logits, actions):
        """ Compute the entropy regularization term.
        Check out https://arxiv.org/pdf/1805.00909.pdf

        @param logits (torch.Tensor): Tensor of shape (batch_size, steps, num_act),
                giving the logits for every action at every time step.
        @param actions (torch.Tensor): Tensor of shape (b, t), giving the actions
                selected by the policy during rollout.
        @return entropy_term (torch.Tensor): Tensor of shape (b, t), giving the
                entropy regularization term for every timestep.
        """
        # log_probs = F.log_softmax(logits, dim=-1)
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # ent = log_probs.gather(index=actions.unsqueeze(dim=2), dim=2).squeeze(dim=2)
        ent = - F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        return - 0.5 * self.sum_to_go(ent)


    def entropy_term_bukov(self, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        return 0.5 * torch.mean(torch.sum(log_probs**2, dim=(1,2)))


    def train(self, num_episodes, steps, learning_rate, lr_decay=1.0, clip_grad=10.0,
              reg=0.0, entropy_reg=0.0, log_every=1, test_every=100, verbose=False):
        """ Train the agent using vanilla policy-gradient algorithm.

        @param num_episodes (int): Number of episodes to train the agent for.
        @param steps (int): Number of steps to rollout the policy for.
        @param learning_rate (float): Learning rate for gradient decent.
        @param lr_decay (float): Multiplicative factor of learning rate decay.
        @param clip_grad (float): Threshold for gradient norm during backpropagation.
        @param reg (float): L2 regularization strength.
        @param entropy_reg (float): Entropy regularization strength.
        @param log_every (int): Every @log_every iterations write the results
                to the log file.
        @param verbose (bool): If true, printout logging information.
        @param initial_states (np.array): Numpy array giving the initial state of the
                environment. If None, start at random initial states. (exploring starts)
        @param batch_mode (bool): When in `batch_mode`, all states in the batch are the same.
        """
        self.log_every = log_every
        self.steps = steps

        self.logger.setLogTxtFilename("test_history.txt")
        self.logger.setLogTxtFilename("train_history.txt")
        self.logger.verboseTxtLogging(verbose)

        # Log hyperparameters information.
        self.logger.logTxt("##############################")
        self.logger.logTxt("Training parameters:")
        self.logger.logTxt("  Number of trajectories:   {}".format(self.env.batch_size))
        self.logger.logTxt("  Number of episodes:       {}".format(num_episodes))
        self.logger.logTxt("  Learning rate:            {}".format(learning_rate))
        self.logger.logTxt("  Final learning rate:      {}".format(
            round(learning_rate * (lr_decay ** num_episodes), 7)))
        self.logger.logTxt("  Weight regularization:    {}".format(reg))
        self.logger.logTxt("  Entropy regularization:   {}".format(entropy_reg))
        self.logger.logTxt("  Grad clipping threshold:  {}".format(clip_grad))
        self.logger.logTxt("  Policy hidden dimensions: {}".format(self.policy.hidden_sizes))
        self.logger.logTxt("  Policy dropout rate:      {}".format(self.policy.dropout_rate))

        # Move the neural network to device and prepare for training.
        self.policy.train()
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        self.logger.logTxt("\nUsing device: {}\n".format(device))
        self.policy = self.policy.to(device)

        # Initialize the optimizer and the scheduler.
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
        self.logger.logTxt("Using optimizer:\n{}\n".format(str(optimizer)))

        self.env.set_random_state(copy=False)
        initial_state = self.env._state
        # Start the training loop.
        for i in range(num_episodes):
            tic = time.time()

            # Set the initial state and perform policy roll-out.
            # self.env.set_random_state()
            self.env._state = initial_state
            states, actions, rewards, done = self.rollout(steps)
            mask = ~torch.cat((done[:, 0:1], done[:, 1:] & done[:, :-1]), dim=1)

            # Compute the loss.
            logits = self.policy(states)
            nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
            q_values = self.reward_to_go(rewards) + entropy_reg * self.entropy_term(logits, actions)
            q_values = q_values - self.reward_baseline(q_values, mask)
            weighted_nll = torch.mul(mask * nll, q_values)
            loss = torch.mean(torch.sum(weighted_nll, dim=1))

            # Perform backward pass.
            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.policy.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()

            # Book-keeping.
            d_mask = np.any(done.cpu().numpy(), axis=1)
            d_steps = np.argmax(done.cpu().numpy()[d_mask], axis=1) + 1
            self.train_history[i] = {
                "entropy"       : self.env.entropy().tolist(),
                # "states"        : states.cpu().numpy().tolist(),
                "rewards"       : rewards.cpu().numpy().tolist(),
                "loss"          : loss.item(),
                "total_norm"    : total_norm.cpu().numpy().tolist(),
                "nsolved"       : sum(done[:, -1]).cpu().numpy().tolist(),
                # "nsteps"        : (np.argmax(done.cpu().numpy(), axis=1) + 1).tolist(),
                "nsteps"        : d_steps.tolist(),
            }

            toc = time.time()

            # Log results to file.
            if i == 0 or (i+1) % log_every == 0:
                self.logger.logTxt("Episode ({}/{}) took {:.3f} seconds.".format(
                    i + 1, num_episodes, (toc-tic)))
                self.log_train_statistics(i)
                probs = F.softmax(logits, dim=-1)
                print("Timestep {}\nprobs: {}\n{}\n{}\n{}\n{}\n".format(
                    (i+1), probs[0][0], probs[0][1], probs[0][2], probs[0][3], probs[0][4]))

            # Test the agent.
            if i == 0 or (i+1) % test_every == 0:
                self.logger.setLogTxtFilename("test_history.txt", append=True)
                self.logger.logTxt("\n\niteration {}".format(i+1))
                self.log_test_accuracy(num_test=1, steps=steps, initial_state=initial_state)
                self.logger.setLogTxtFilename("train_history.txt", append=True)

        self.plot_training_curves()
        self.plot_distribution()
        self.save_policy()
        # self.save_history()

#