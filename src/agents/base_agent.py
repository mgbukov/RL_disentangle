import os
import pickle

import numpy as np
import torch


class BaseAgent:
    """
    An abstract class implementation of a reinforcement learning agent.
    The agent is initialized with a policy and an environment, and uses the
    policy to act in the environment.
    Concrete classes must implement their own training strategies.
    """

    def __init__(self, env, policy, kind='vec'):
        self.env = env
        self.policy = policy
<<<<<<< HEAD
        if kind not in ('vec', 'rdm', 'cvec', 'ctrdm'):
            raise ValueError('`kind` must be one of ("vec", "rdm", "cvec", "ctrdm")')
=======
        if kind not in ('vec', 'rdm'):
            raise ValueError('`kind` must be one of ("vec", "rdm")')
>>>>>>> 2a69c6d (Added RDM observation in BaseAgent. Other changes:)
        self.observation_kind = kind
        # Bind `self.observe()` method dynamically
        if kind == 'vec':
            self.observe = self._observe_vec
<<<<<<< HEAD
            self._shape = (2 ** (self.env.L + 1),)
        elif kind == 'cvec':
            self.observe = self._observe_cvec
            self._shape = (2 ** self.env.L,)
        elif kind == 'rdm':
            self.observe = self._observe_rdm
            self._shape = (self.env.num_actions * 16 * 2,)
        elif kind == 'ctrdm':
            self.observe = self._observe_rdm
            self._shape = (self.env.num_actions, 16)
=======
            self._obslen = 2 ** (self.env.L + 1)
        else:
            self.observe = self._observe_rdm
            self._obslen = (self.env.num_actions // 2) * 16 * 2
>>>>>>> 2a69c6d (Added RDM observation in BaseAgent. Other changes:)

    def train(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")

    @torch.no_grad()
    def get_action_from_env(self, greedy=False, beta=1.0):
        b = self.env.batch_size  # number of trajectories
        batch = self.env.states.reshape(b, -1)
        batch =  np.hstack([batch.real, batch.imag])
        acts = self.policy.get_action(torch.from_numpy(batch), greedy=greedy, beta=beta)
        return acts

    # def observe(self):
    # ------------------
    #   DYNAMICALLY BINDED IN `self.__init__()` to either `self._observe_vec()`
    #   or `self._observe_rdm()`

    @torch.no_grad()
    def _observe_vec(self):
        psi = self.env.states.reshape(self.env.batch_size, -1)
        return np.hstack([psi.real, psi.imag])
<<<<<<< HEAD
    
    @torch.no_grad()
    def _observe_cvec(self):
        return self.env.states.reshape(self.env.batch_size, -1)
=======
>>>>>>> 2a69c6d (Added RDM observation in BaseAgent. Other changes:)

    @torch.no_grad()
    def _observe_rdm(self):
        states = self.env.states
        rdms = []
<<<<<<< HEAD
        # qubit_pairs = [a for a in self.env.actions.values() if a[0] < a[1]]
        qubit_pairs = list(self.env.actions.values())
=======
        qubit_pairs = [a for a in self.env.actions.values() if a[0] < a[1]]
        # qubit_pairs = list(self.env.actions.values())
>>>>>>> 2a69c6d (Added RDM observation in BaseAgent. Other changes:)
        for qubits in qubit_pairs:
            sysA = tuple(q+1 for q in qubits)
            sysB = tuple(q+1 for q in range(self.env.L) if q not in qubits)
            permutation = (0,) + sysA + sysB
            states = np.transpose(states, permutation)
            psi = states.reshape(self.env.batch_size, 4, -1)
            rdm = psi @ np.transpose(psi, (0, 2, 1)).conj()
            rdms.append(rdm)
            # Reset permutation
            states = np.transpose(states, np.argsort(permutation))
            assert np.all(states == self.env.states)
<<<<<<< HEAD
        rdms = np.array(rdms)                   # rdms.shape == (Q, B, 4, 4)
        rdms = rdms.transpose((1, 0, 2, 3))     # rdms.shape == (B, Q, 4, 4)
        if self.observation_kind == 'ctrdm':
            return rdms.reshape((-1,) + self._shape)
        elif self.observation_kind == 'rdm':
            result = rdms.reshape(self.env.batch_size, -1, 16)
            return np.dstack([result.real, result.imag]).reshape((-1,) + self._shape)
=======
        rdms = np.array(rdms)           # rdms.shape == (Q, B, 4, 4)
        rdms.transpose((1, 0, 2, 3))    # rdms.shape == (B, Q, 4, 4)
        result = rdms.reshape(self.env.batch_size, -1, 16)
        result = np.dstack([result.real, result.imag]).reshape(-1, self._obslen)
        return result

>>>>>>> 2a69c6d (Added RDM observation in BaseAgent. Other changes:)

    @torch.no_grad()
    def rollout(self, steps, plan=None, greedy=False, beta=1.0):
        """Starting from the current environment state perform a rollout using the
        current policy.

        Args:
            steps (int): Number of steps to rollout the policy.
            plan (torch.Tensor): Tensor of shape (b, t), giving a plan of the actions that
                the agent should take. If None, the agent uses the policy to pick actions.
            greedy (bool): If true, select the next action deterministically. If false,
                select the next action probabilistically.
            beta (float): Inverse value of the temperature for the boltzmann distribution.

        Returns:
            states (torch.Tensor): Tensor of shape (b, t+1, q), giving the states produced
                during policy rollout, where b = batch size, t = number of time steps,
                q = size of the quantum system (2 ** num_qubits).
            actions (torch.Tensor): Tensor of shape (b, t), giving the actions selected by
                the policy during rollout.
            rewards (torch.Tensor): Tensor of shape (b, t), giving the rewards obtained
                during policy rollout.
            done (torch.Tensor): Tensor of shape (b, t) of boolean values, indicating
                for which states of the batch the environment loop is done.
        """
        device = self.policy.device
        b = self.env.batch_size  # number of trajectories
        L = self.env.L

        # Allocate torch tensors to store the data from the rollout.
<<<<<<< HEAD
        if self.observation_kind.startswith('c'):
            states = torch.zeros(size=(steps+1,b)+self._shape, dtype=torch.complex64)
        else:
            states = torch.zeros(size=(steps+1,b)+self._shape, dtype=torch.float32)
        actions = torch.zeros(size=(steps, b), dtype=torch.int64)
        rewards = torch.zeros(size=(steps, b), dtype=torch.float32)
        done = torch.zeros(size=(steps, b), dtype=torch.bool)
=======
        states = torch.zeros(size=(steps+1, b, self._obslen), dtype=torch.float32, device=device)
        actions = torch.zeros(size=(steps, b), dtype=torch.int64, device=device)
        rewards = torch.zeros(size=(steps, b), dtype=torch.float32, device=device)
        done = torch.zeros(size=(steps, b), dtype=torch.bool, device=device)
>>>>>>> 2a69c6d (Added RDM observation in BaseAgent. Other changes:)

        # Perform parallel rollout along all trajectories.
        for i in range(steps):
            batch = self.observe()
            states[i] = torch.from_numpy(batch)
            if plan is None:
                acts = self.policy.get_action(states[i], greedy=greedy, beta=beta)
            else:
                acts = plan[:, i]
            actions[i] = acts
            s, r, d = self.env.step(acts.cpu().numpy())
            rewards[i] = torch.from_numpy(r)
            done[i] = torch.from_numpy(d)

        # Add the last state to the trajectories.
        batch = self.observe()
        states[steps] = torch.from_numpy(batch)

        # Permute `step` and `batch_size` dimensions
        states = states.permute(1, 0, 2)
        done = done.permute(1, 0)
        actions = actions.permute(1, 0)
        rewards = rewards.permute(1, 0)

        # Mask out the rewards after a trajectory is done.
        mask = self.generate_mask(done)
        rewards = mask * rewards

        # Move to device and return
        return (states.to(device), actions.to(device),
                rewards.to(device), done.to(device))

    @torch.no_grad()
    def generate_mask(self, done):
        """Using the `done` tensor generate a mask for the batch of trajectories, that
        masks out the part of any trajectory that has finished.

        If done[i] is False and done[i+1] is True, then the trajectory should be masked
        out at and after step i+2.

        Args:
            done (torch.Tensor): Tensor of shape (b, t) of boolean values, indicating
                for which states of the batch the environment loop is done.

        Returns:
            mask (torch.Tensor): Tensor of shape (b, t), of boolean values, that masks
                out the part of the trajectory after it has finished.
        """
        m = torch.roll(done, shifts=1, dims=1)
        m[:, 0] = False
        return ~m

    @torch.no_grad()
    def test_accuracy(self, num_test, steps, initial_states=None, greedy=True):
        """Test the accuracy of the agent using @num_test simulation rollouts.

        Args:
            num_test (int): Number of simulations to test the agent.
            steps (int): Number of steps to rollout the policy during simulation.
            initial_states (np.Array, optional): Numpy array giving the initial states for
                the environment. Default value is None.
            greedy (bool, optional): If true, select the next action deterministically.
                If false, select the next action probabilistically. Default value is False.

        Returns:
            entropies (np.Array): A numpy array of shape (num_episodes, L), giving the
                final entropies for each trajectory during testing,
            returns (np.Array): A numpy array of shape (num_episodes,), giving the
                obtained return during each trajectory.
            nsolved (np.Array): A numpy array of shape (num_episodes,), of boolean values,
                indicating which trajectories are disentangled.
            nsteps (np.Array): A numpy array of shape (num_episodes,), giving the number
                of steps for each episode.
        """
        batch_size = self.env.batch_size  # number of trajectories
        entropies = np.zeros((num_test,) + self.env.entropy().shape)
        returns = np.zeros((num_test, batch_size))
        nsolved = np.zeros((num_test, batch_size))
        nsteps  = np.zeros((num_test, batch_size))

        # Begin testing.
        for i in range(num_test):
            if initial_states is None:
                self.env.set_random_states(copy=False)
            else:
                self.env.state = initial_states
            states, actions, rewards, done = self.rollout(steps, greedy=greedy)
            mask = self.generate_mask(done)
            entropies[i] = self.env.entropy()
            returns[i] = torch.sum(mask*rewards, axis=1).cpu().numpy()
            nsolved[i] = self.env.disentangled()
            nsteps[i] = torch.sum(mask, axis=1).cpu().numpy()

        num_episodes = num_test * batch_size
        return (entropies.reshape(num_episodes, -1), returns.reshape(num_episodes),
                nsolved.reshape(num_episodes), nsteps.reshape(num_episodes))

    def save_policy(self, filepath, filename="policy.bin"):
        """Save the policy as .bin file to disk."""
        self.policy.save(os.path.join(filepath, filename))

    def save_history(self, filepath):
        """Save the training history and the testing history as pickle dumps."""
        with open(os.path.join(filepath, "train_history.pickle"), "wb") as f:
            train_hist = {k : dict(v) for k, v in self.train_history.items()}
            pickle.dump(train_hist, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(filepath, "test_history.pickle"), "wb") as f:
            pickle.dump(self.test_history, f, protocol=pickle.HIGHEST_PROTOCOL)

#