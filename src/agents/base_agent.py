import torch


class BaseAgent:
    """ An abstract class implementation of a reinforcement learning agent.
    The agent is initialized with a policy and an environment, and uses the
    policy to act in the environment.
    Concrete classes must implement their own training strategies.
    """

    def __init__(self):
        raise NotImplementedError("This method must be implemented by the subclass")


    def train(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")


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
        batch_size = self.env.batch_size  # number of trajectories

        # Allocate torch tensors to store the data from the rollout.
        states = torch.zeros(size=(steps,) + self.env.state.shape, dtype=torch.float32, device=device)
        actions = torch.zeros(size=(steps, batch_size), dtype=torch.int64, device=device)
        rewards = torch.zeros(size=(steps, batch_size), dtype=torch.float32, device=device)
        done = torch.zeros(size=(steps, batch_size), dtype=torch.bool, device=device)

        # Perform parallel rollout along all trajectories.
        acts = None
        for i in range(steps):
            states[i] = torch.from_numpy(self.env.state)    # env.state is (B, 2**q)
            acts = self.policy.get_action(states[i], acts, greedy)
            actions[i] = acts
            s, r, d = self.env.step(acts)
            rewards[i] = torch.from_numpy(r)
            done[i] = torch.from_numpy(d)

        # Permute `step` and `batch_size` dimensions.
        return (states.permute(1, 0, 2), actions.permute(1, 0),
                rewards.permute(1, 0), done.permute(1, 0))


    def test_accuracy(self, num_test, steps):
        """ Test the accuracy of the agent using @num_test simulation rollouts.

        @param num_test (int): Number of simulations to test the agent.
        @param steps (int): Number of steps to rollout the policy during simulation.
        """
        solved = 0
        entropies = torch.zeros(size=(num_test, self.env.batch_size))

        for i in range(num_test):
            self.env.set_random_state()
            states, actions, rewards, done = self.rollout(steps, greedy=True)
            solved += sum(done[:,-1])
            entropies[i] = - rewards[:, -1] * torch.log(torch.FloatTensor([2.]))

        self.logger.logTxt("##############################")
        self.logger.logTxt("Testing agent accuracy...")
        self.logger.logTxt("Solved states: {} / {} = {:.3f}%".format(
            solved, num_test*self.env.batch_size, solved/(num_test*self.env.batch_size)*100))
        self.logger.logTxt("95 percentile entropy: {:.5f}".format(
            torch.quantile(entropies.reshape(-1), 0.95)))
        self.logger.logTxt("##############################")

#