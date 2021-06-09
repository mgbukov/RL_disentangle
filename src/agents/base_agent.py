import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F


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


    @torch.no_grad()
    def rollout(self, steps, greedy=False, beta=1.0):
        """ Starting from the current environment state perform a rollout using
        the current policy.

        @param steps (int): Number of steps to rollout the policy.
        @param greedy (bool): If true, select the next action deterministically.
                If false, select the next action probabilistically.
        @param beta (float): Inverse value of the temperature for the boltzmann
                distribution.
        @return states (torch.Tensor): Tensor of shape (b, t, q), giving the states
                produced during policy rollout, where
                b = batch size, t = number of time steps,
                q = size of the quantum system (2 ** num_qubits).
        @return actions (torch.Tensor): Tensor of shape (b, t), giving the actions
                selected by the policy during rollout.
        @return rewards (torch.Tensor): Tensor of shape (b, t), giving the rewards
                obtained during policy rollout.
        @return done (torch.Tensor): Tensor of shape (b, t),
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
            states[i] = torch.from_numpy(self.env.state)    # states is (T, B, 2**q)
            acts = self.policy.get_action(states[i], acts, greedy, beta)
            actions[i] = acts
            s, r, d = self.env.step(acts)
            rewards[i] = torch.from_numpy(r)
            done[i] = torch.from_numpy(d)

        # Permute `step` and `batch_size` dimensions.
        return (states.permute(1, 0, 2), actions.permute(1, 0),
                rewards.permute(1, 0), done.permute(1, 0))


    def log_train_statistics(self, i):
        """ Log training statistics for episode @i. """
        self.logger.logTxt("  Mean final reward:        {:.4f}".format(
            np.mean(np.array(self.train_history[i]["rewards"])[:,-1])))
        self.logger.logTxt("  Mean return:              {:.4f}".format(
            np.mean(np.sum(self.train_history[i]["rewards"], axis=1))))
        self.logger.logTxt("  Mean return policy ent:   {:.4f}".format(
            self.train_history[i]["entropy_term"]))
        self.logger.logTxt("  Mean final entropy:       {:.4f}".format(
            np.mean(self.train_history[i]["entropy"])))
        self.logger.logTxt("  Max final entropy:        {:.4f}".format(
            np.max(self.train_history[i]["entropy"])))
        self.logger.logTxt("  95 percentile entropy:    {:.5f}".format(
            np.percentile(self.train_history[i]["entropy"], 95.0)))
        self.logger.logTxt("  Pseudo loss:              {:.5f}".format(
            self.train_history[i]["loss"]))
        self.logger.logTxt("  Total gradient norm:      {:.5f}".format(
            self.train_history[i]["total_norm"]))
        self.logger.logTxt("  Solved trajectories:      {} / {}".format(
            self.train_history[i]["nsolved"], self.env.batch_size))
        self.logger.logTxt("  Avg steps to disentangle: {:.3f}".format(
            np.mean(self.train_history[i]["nsteps"])))


    def log_test_accuracy(self, num_test, steps):
        """ Test the accuracy of the agent using @num_test simulation rollouts.

        @param num_test (int): Number of simulations to test the agent.
        @param steps (int): Number of steps to rollout the policy during simulation.
        """
        batch_size = self.env.batch_size  # number of trajectories
        self.logger.logTxt("##############################")
        self.logger.logTxt("Testing agent accuracy for {:d} steps...".format(steps))

        # Begin testing.
        tic = time.time()
        solved = 0
        entropies = torch.zeros(size=(num_test, batch_size))
        returns = torch.zeros(size=(num_test, batch_size))
        nsolved = torch.zeros(size=(num_test, batch_size))
        for i in range(num_test):
            disentangled = True
            while disentangled:
                self.env.set_random_state()
                disentangled = self.env.disentangled().any()
            # self.env.set_random_state()
            states, actions, rewards, done = self.rollout(steps, greedy=True)
            solved += sum(done[:,-1])
            entropies[i] = torch.from_numpy(self.env.entropy())
            returns[i] = torch.sum(rewards, axis=1)
            nsolved[i] = done[:, -1]
        toc = time.time()

        # Log results.
        self.logger.logTxt("  Solved states:         {} / {} = {:.3f}%".format(
            solved, num_test*batch_size, solved/(num_test*batch_size)*100))
        self.logger.logTxt("  95 percentile entropy: {:.5f}".format(
            torch.quantile(entropies.reshape(-1), 0.95)))
        self.logger.logTxt("  Max entropy:           {:.5f}".format(entropies.max()))
        self.logger.logTxt("  Mean return:           {:.4f}".format(torch.mean(returns)))
        self.logger.logTxt("  Testing took:          {:.3f} seconds".format(toc-tic))
        self.logger.logTxt("##############################")

        # Bookkeeping.
        self.test_history[steps].append({
            "returns": torch.mean(returns).item(),
            "nsolved": torch.mean(nsolved).item(),
        })


    def plot_training_curves(self):
        """ Use the history saved during training the agent to plot relevant curves. """
        steps = self.steps
        batch_size = self.env.batch_size
        num_train = len(self.train_history)
        num_test = len(self.test_history[steps])
        test_every = (num_train - 1) // (num_test - 1)
        log_every = self.log_every

        # Entropies curve.
        ent_min = np.array([np.min(self.train_history[i]["entropy"]) for i in range(num_train)])
        ent_max = np.array([np.max(self.train_history[i]["entropy"]) for i in range(num_train)])
        ent_mean = np.array([np.mean(self.train_history[i]["entropy"]) for i in range(num_train)])
        ent_std = np.array([np.std(self.train_history[i]["entropy"]) for i in range(num_train)])
        ent_mean_minus_std = ent_mean - 0.5 * ent_std
        ent_mean_plus_std = ent_mean + 0.5 * ent_std
        ent_quantile = np.array([np.percentile(self.train_history[i]["entropy"], 95) for i in range(num_train)])
        self.logger.logPlot(funcs=[ent_min, ent_max, ent_mean, ent_quantile],
                            legends=["min", "max", "mean", "95%quantile"],
                            labels={"x":"Episode", "y":"Entropy"},
                            fmt=["--r", "--b", "-k", ":m"],
                            fills=[(ent_mean_minus_std, ent_mean_plus_std)],
                            figtitle="System entropy after {} steps".format(steps),
                            figname="final_entropy.png")

        # Loss curve.
        loss = [self.train_history[i]["loss"] for i in range(num_train)]
        self.logger.logPlot(funcs=[loss], legends=["loss"],
                            labels={"x":"episode", "y":"loss"}, fmt=["-b"], figname="loss.png")

        # Rewards & Returns curve.
        returns = [np.sum(self.train_history[i]["rewards"], axis=1).mean() for i in range(num_train)]
        avg_returns = np.insert(np.mean(np.array(returns[1:]).reshape(-1, log_every), axis=1), 0, returns[0])
        test_returns = [self.test_history[steps][i]["returns"] for i in range(num_test)]
        self.logger.logPlot(xs=[np.arange(num_train),
                                np.arange(0, num_train, log_every),
                                np.arange(0, num_train, test_every)],
                            funcs=[returns, avg_returns, test_returns],
                            legends=["batch_returns", "avg_returns", "test_returns"],
                            labels={"x":"Episode", "y":"Return"},
                            fmt=["--r", "-k", "-b"],
                            lw=[1.0, 4.0, 4.0],
                            figtitle="Agent achieved return",
                            figname="returns.png")
        self.logger.logPlot(funcs=[returns], legends=["batch_returns"], fmt=["--r"],
                            figtitle="Training Loss", figname="returns_logX.png", logscaleX=True)
        # MUST HAVE POSITIVE VALUES ALONG Y-AXIS!!!
        # self.logger.logPlot(funcs=[returns], legends=["batch_returns"], fmt=["--r"],
        #                     figtitle="Training Loss", figname="returns_logY.png", logscaleY=True)

        # Solved trajectories curve.
        nsolved = [self.train_history[i]["nsolved"] / batch_size for i in range(num_train)]
        avg_nsolved = np.insert(np.mean(np.array(nsolved[1:]).reshape(-1, log_every), axis=1), 0, nsolved[0])
        tst_nsolved = [self.test_history[steps][i]["nsolved"] for i in range(num_test)]
        tst_nsolved1 = [self.test_history[steps+1][i]["nsolved"] for i in range(num_test)]
        tst_nsolved2 = [self.test_history[steps+2][i]["nsolved"] for i in range(num_test)]
        self.logger.logPlot(xs=[np.arange(num_train),
                                np.arange(0, num_train, log_every),
                                np.arange(0, num_train, test_every),
                                np.arange(0, num_train, test_every),
                                np.arange(0, num_train, test_every)],
                            funcs=[nsolved, avg_nsolved, tst_nsolved, tst_nsolved1, tst_nsolved2],
                            legends=["nsolved", "avg_nsolved",
                                     "test_nsolved_{}".format(steps),
                                     "test_nsolved_{}".format(steps+1),
                                     "test_nsolved_{}".format(steps+2)],
                            labels={"x":"Episode", "y":"nsolved"},
                            fmt=["--r", "-k", "-b", "-g", "-y"],
                            lw=[1.0, 4.0, 4.0, 4.0, 4.0],
                            figtitle="Agent accuracy of solved states",
                            figname="nsolved.png")


    def plot_distribution(self):
        """ Plot a heat map of the probability distribution over the actions returned by
        the policy.
        """
        num_test = 100
        probs = np.ndarray(shape=(num_test, self.env.batch_size, self.env.num_actions))
        for i in range(num_test):
            disentangled = True
            while disentangled:
                self.env.set_random_state()
                disentangled = self.env.disentangled().any()
            st = torch.from_numpy(self.env.state).type(torch.float32)
            logits = self.policy(st)
            probs[i] = F.softmax(logits, dim=-1).detach().numpy()
        probs = probs.reshape(num_test * self.env.batch_size, self.env.num_actions)
        self.logger.logPcolor(probs.T, "policy_distribution.png",
                              figtitle="Average distribution of actions given by the policy",
                              labels={"x":"Test No", "y":"Actions"})


    def save_policy(self):
        """ Save the policy as .bin file to disk. """
        self.policy.save(self.logger.log_dir + "/policy.bin")


    def save_history(self):
        """ Save the training history and the testing history as pickle dumps. """
        log_dir = self.logger.log_dir
        with open(os.path.join(log_dir, "train_history.pickle"), "wb") as f:
            pickle.dump(self.train_history, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(log_dir, "test_history.pickle"), "wb") as f:
            pickle.dump(dict(self.test_history), f, protocol=pickle.HIGHEST_PROTOCOL)   # can't pickle defaultdict

#