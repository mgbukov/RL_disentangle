def train(self, num_episodes, replays, repeat_initial, steps,
            learning_rate, reg, verbose=True):
    """ Train the agent using vanilla policy-gradient algorithm.

    @param num_episodes (int): Number of episodes to train the agent for.
    @param replays (int): Number of times the trajectories are replayed.
    @param repeat_initial (int):
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
        self.env.set_random_state()
        initial_state = self.env._state.copy()
        for initial in range(repeat_initial):
            self.env.state = initial_state
            states, actions, rewards, done = self.rollout(steps)
            q_values = self.reward_to_go(rewards) - self.reward_baseline(rewards, done)

            for j in range(replays):
                logits = self.policy(states)
                nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
                weighted_nll = torch.mul(nll, q_values)
                loss = torch.mean(torch.sum(weighted_nll, axis=1))

                # Zero the gradients, perform backward pass, clip the gradients, and update the gradients.
                optimizer.zero_grad()
                loss.backward()
                # total_norm = torch.norm(torch.stack([torch.norm(p.grad)for p in self.policy.parameters()]))
                # print("total_norm: %.4f" % total_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
                optimizer.step()