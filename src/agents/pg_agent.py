import numpy as np
import time
import torch
import torch.nn.functional as F


class PGAgent:
    """ Policy-gradient agent. """

    def __init__(self, env, policy):
        """ Initialize policy gradient agent.

        @param env (QubitsEnvironment object): Environment object.
        @param policy (Policy object): Policy object.
        """
        self.env = env
        self.policy = policy
        self.history = []


    def rollout(self, steps, greedy=False):
        """ Starting from the current environment state perform a rollout using
        the current policy.

        @param steps (int): Number of steps to rollout the policy.
        @param greedy (bool): If true, select the next action deterministically.
                If false, select the next action probabilistically.
        @return states (Tensor): Tensor of shape (b, t, q), giving the states
                produced during policy rollout, where
                b = batch size, t = number of time steps,
                q = size of the quantum system (2 ** num_qubits).
        @return actions (Tensor): Tensor of shape (b, t), giving the actions
                selected by the policy during rollout.
        @return rewards (Tensor): Tensor of shape (b, t), giving the rewards
                obtained during policy rollout.
        @return done (Tensor): Tensor of shape (b, t),
        """
        device = self.policy.device
        num_actions = self.env.num_actions
        batch_size = self.env.batch_size  # number of trajectories

        # Allocate numpy arrays to store the data from the rollout.
        states = np.ndarray(shape=(steps,) + self.env.state.shape, dtype=np.float32)
        actions = np.ndarray(shape=(steps, batch_size), dtype=np.int64)
        rewards = np.ndarray(shape=(steps, batch_size), dtype=np.float32)
        done = np.ndarray(shape=(steps, batch_size), dtype=np.bool_)

        # Perform parallel rollout along all trajectories.
        for i in range(steps):
            acts = self.policy.get_action(self.env.state, greedy)
            states[i] = self.env.state
            actions[i] = acts
            s, r, d = self.env.step(acts)
            rewards[i] = r
            done[i] = d

        # Convert numpy arrays to torch tensors and send them to device.
        return (torch.from_numpy(states.transpose(1,0,2)).to(device),
                torch.from_numpy(actions.transpose(1,0)).to(device),
                torch.from_numpy(rewards.transpose(1,0)).to(device),
                torch.from_numpy(done.transpose(1,0)).to(device))


    @torch.no_grad()
    def reward_to_go(self, rewards):
        return rewards + torch.sum(rewards, keepdims=True, axis=1) - torch.cumsum(rewards,axis=1)


    @torch.no_grad()
    def reward_baseline(self, rewards, done):
        """ Compute the baseline as the average return at time-step t.

        @param rewards (torch.Tensor): Tensor of shape (batch_size, steps),
                containing the rewards obtained at every step.
        @param done (torch.Tensor): Tensor of shape (batch_size, steps).
        """
        return torch.mean(self.reward_to_go(rewards), dim=0, keepdim=True)


    def train(self, num_episodes, steps, learning_rate, reg, verbose=True):
        """ Train the agent using vanilla policy-gradient algorithm.

        @param num_episodes (int): Number of episodes to train the agent for.
        @param steps (int): Number of steps to rollout the policy for.
        @param learning_rate (float): Learning rate for gradient decent.
        @param reg (float): L2 regularization strength.
        @param verbose (bool): If true, prinout logging information.
        """
        self.policy.train()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device: %s" % device)
        self.policy = self.policy.to(device)

        # Initialize the optimizer.
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=reg)

        num_trajectories = self.env.batch_size
        for i in range(num_episodes):
            tic = time.time()

            self.env.set_random_state()
            states, actions, rewards, done = self.rollout(steps)
            q_values = self.reward_to_go(rewards) - self.reward_baseline(rewards, done)

            # Compute the loss.
            logits = self.policy(states)
            nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
            weighted_nll = torch.mul(nll, q_values)
            loss = torch.mean(torch.sum(weighted_nll, axis=1))

            # Perform backward pass.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            optimizer.step()

            # Book-keeping.
            entropies = self.env.entropy()
            self.history.append((entropies.min(), entropies.max(),
                                 entropies.mean(), entropies.std()))

            toc = time.time()

            # Printout results.
            if verbose:
                print("Episode (%d/%d) took %.3f seconds." % (i + 1, num_episodes, (toc-tic)))
                print("    Mean final reward: %.4f" % torch.mean(rewards[:,-1]))
                print("    Mean final entropy: %.4f" % entropies.mean())
                print("    Pseudo loss: %.5f" % loss.item())

#