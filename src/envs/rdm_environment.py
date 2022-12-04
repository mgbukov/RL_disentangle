from itertools import permutations
import numpy as np
from scipy.linalg import expm

from src.envs import util
# from src.envs.batch_transpose import cy_transpose_batch


class QubitsEnvironment:
    """Representation of a multi-qubit quantum system.
    The states of the system are represented as a numpy array of shape (b, 2,2,...,2),
    giving for each state in the batch its decomposition in the computational basis.

    Only two-qubit gates are applied to the system. Taking an action consists of selecting
    a pair of qubits and applying a unitary gate on them. The action space consists of all
    possible pairs of qubits.

    The entropy of a multi-qubit state is computed by considering every single qubit as a
    subsystem and computing the entanglement entropy between that single qubit system and
    the rest of the system.
    The reward upon transition is defined as a function of the mean of the entropies of
    the new multi-qubit state.

    Attributes:
        L (int): Number of qubits in the quantum state.
        epsi (float): Threshold for disentangling.
        batch_size (int): Number of quantum states in the environment.
        num_actions (int): Number of allowable actions.
        actions (dict): A mapping from action index to action name.
        actToKey (dict): A mapping from action name to action index.
        shape (Tuple[Int]): A tuple of int giving the shape of the tensor
            representing the environment.
        stochastic (bool): Flag, indicating if transitions are stochastic.
        stochastic_eps (float): Magnitude of the added noise - 0

    """

    def __init__(self, num_qubits=2, epsi=1e-4, batch_size=1, stochastic=False,
                 stochastic_eps=1e-3):
        """Initialize an environment with a batch of multi-qubit states.

        Args:
            num_qubits (int, optional): Number of qubits in a single state.
                Default value is 2.
            epsi (float, optional): Threshold below which the system is considered disentangled.
                Default value is 1e-4.
            batch_size (int, optional): Number of states in the environment.
                Default value is 1.
            stochastic (bool, optional): If `True`, transitions are stochastic.
                Default value is False.
            stochastic_eps (float, optional): Controls the noise magnitude.
                Default value is 1e-3.
        """
        assert num_qubits >= 2
        self.L = num_qubits
        self.epsi = epsi
        self.batch_size = batch_size
        self.stochastic = bool(stochastic)
        self.stochastic_eps = float(stochastic_eps)

        # The action space consists of all possible pairs of qubits.
        self.num_actions = self.L * (self.L - 1)
        self.actions = dict(enumerate(permutations(range(num_qubits), 2)))
        self.actToKey = {v:k for k, v in self.actions.items()}

        # The states of the system are represented as a numpy array of shape (b, 2,2,...,2).
        self.shape = (batch_size,) + (2,) * num_qubits
        self._states = np.zeros(self.shape, dtype=np.complex64)
        # self._states_swap_buffer = np.zeros(self.shape, dtype=np.complex64)
        self._entropies_cache = np.zeros((self.batch_size, self.L), dtype=np.float32)
        self.reset()

    def reset(self):
        """Prepare all states in the batch as | 000..00 >"""
        psi = np.zeros((self.batch_size, 2 ** self.L), dtype=np.complex64)
        psi[:, 0] = 1.0
        self._states = psi.reshape(self.shape)
        self._entropies_cache = np.zeros((self.batch_size, self.L), dtype=np.float32)

    @property
    def states(self):
        """np.Array: Numpy array of shape (b,2,2,...,2) giving the quantum states."""
        return self._states.copy()

    @states.setter
    def states(self, newstates):
        if self.shape[1:] != newstates.shape[1:]:
            raise ValueError(f"Shape missmatch!\nshape: {self.shape}\nnew: {newstates.shape}")
        self.batch_size = newstates.shape[0]
        self.shape = (self.batch_size,) + (2,) * self.L
        self._states = util.phase_norm(newstates)
        self._entropies_cache = util.entropy(self._states)

    def set_random_states(self, copy=False):
        """Set all states of the environment to random pure states. If copy is True,
        all states in the batch are copies of a single state.
        Compute the entropy of the states and cache them for later use.
        """
        if copy:
            psi = util.random_pure_state(self.L)
            psi = np.tile(psi, (self.batch_size, 1))
        else:
            psi = util.random_batch(self.L, self.batch_size)
        self._states = util.phase_norm(psi.reshape(self.shape))
        self._entropies_cache = util.entropy(self._states)

    def step(self, actions):
        """Applies a batch of actions to the states batch and transitions the environment
        to the next batch of states. This function modifies the internal representation
        of the environment inplace.

        Args:
            actions (np.Array): A numpy array of shape (b,), giving the action selected
                for each state of the environment states.

        Returns:
            states (np.Array): A numpy array of shape (b, 2,2,...,2), giving the next
                batch of states after taking the actions.
            rewards (np.Array): A numpy array of shape (b,), giving the rewards after
                transitioning into the new states.
            done (np.Array): A numpy array of shape (b,) of boolean values, indicating
                which states of the batch are disentangled.
        """
        actions = np.atleast_1d(actions)
        if len(actions) != self.batch_size:
            raise ValueError('Expected array with shape=({},)'.format(self.batch_size))
        L = self.L
        B = self.batch_size
        batch = self._states
        qubit_indices = np.array([self.actions[a] for a in actions], dtype=np.int32)
        # ----
        # Move qubits which are modified by ``actions`` at indices (0, 1)
        # batch = np.ascontiguousarray(batch)
        util.permute_qubits(batch, qubit_indices, L, inverse=False)
        # batch = cy_transpose_batch(batch, qubit_indices, _QSYSTEMS_P[self.L])
        # ----
        # Compute 2x2 reduced density matrices
        batch = batch.reshape(B, 4, 2 ** (L - 2))
        rdms = batch @ np.transpose(batch.conj(), [0, 2, 1])
        # ----
        # TODO Add noise
        # U = expm(-1j  eps  H)
        # H = (A + A^\dagger)/2
        # A ~ random matrix
        # 
        # Compute single qubit entropies
        rdms[np.abs(rdms) < 1e-7] = 0.0
        rhos, Us = np.linalg.eigh(rdms)
        phase = np.exp(-1j * np.angle(np.diagonal(Us, axis1=1, axis2=2)))
        np.einsum('kij,kj->kij', Us, phase, out=Us)
        Us = np.swapaxes(Us.conj(), 1, 2)
        Sent_q0, Sent_q1 = util.calculate_q0_q1_entropy_from_rhos(rhos)
        # ----
        # Apply unitary gates
        batch = (Us @ batch)
        self.unitary = Us
        # ----
        # Add noise
        if self.stochastic:
            A = np.random.uniform(size=Us.shape)
            H = 0.5 * (A + np.swapaxes(A.conj(), 1, 2))
            R = expm(-1j * self.stochastic_eps * H)
            batch = (R @ batch)
        batch = batch.reshape(self.shape)
        # ----
        # Undo qubit permutations
        # batch = np.ascontiguousarray(batch)
        util.permute_qubits(batch, qubit_indices, L, inverse=True)
        # batch = cy_transpose_batch(batch, qubit_indices, _QSYSTEMS_INV_P[self.L])
        # ----
        # The new entropies
        self._entropies_cache[np.arange(B), qubit_indices[:, 0]] = Sent_q0
        self._entropies_cache[np.arange(B), qubit_indices[:, 1]] = Sent_q1
        # ----
        self._states = util.phase_norm(batch)
        return self.states, self.reward(), self.disentangled()

    def peek(self, actions, state_only=False):
        """Applies a batch of actions to the states batch and peeks at the next states of
        the environment. This function has the same functionality as the `step` function,
        but it does not modify the internal representation of the environment.
        """
        # Cache the current environment representation.
        batch = self._states.copy()
        entropies = self._entropies_cache.copy()
        # Transition the environment into the new states after taking `actions`.
        res = self.step(actions)
        # Reverse environment state using the cached values.
        self._states = batch
        self._entropies_cache = entropies
        return res

    def entropy(self):
        """Compute the entanglement entropy for the current states.

        Returns:
            entropies (np.array): A numpy array of shape (b, L), giving single-qubit entropies.
        """
        return self._entropies_cache.copy()

    def reward(self):
        """Returns the rewards on transitioning into the current states of the batch.

        Returns:
            rewards (np.Array): A numpy array of shape (b,).
        """
        return self.Reward(self._states, self.epsi, self._entropies_cache)

    def disentangled(self):
        """ Returns an array indicating which states of the batch are disentangled.

        Returns:
            done (np.Array): A numpy array of shape (b,).
        """
        return self.Disentangled(self._states, self.epsi, self._entropies_cache)

    @staticmethod
    def Entropy(states):
        """For each state in the batch compute the entanglement entropies by considering
        each qubit as a subsystem.

        Args:
            states (np.array): A numpy array of shape (b, 2,2,...,2), giving the states in
                the batch.

        Returns:
            entropies (np.array): A numpy array of shape (b, L), giving single-qubit entropies.
        """
        return util.entropy(states)

    @classmethod
    def Reward(cls, states, epsi=1e-4, entropies=None):
        """Returns the immediate rewards on transition to `states`.
        The rewards are calculated as the negative logarithm of the entropies.
        The choice depends on the fact that the reward increases exponentially, when the
        entropy approaches 0, and thus encouraging the agent to disentangle the state.

        Returns:
            rewards (np.Array): A numpy array of shape (b,).
        """
        return (-1. + 101. * cls.Disentangled(states, epsi, entropies)).astype(np.float32)
        # # Compute the entropy of a system by considering each individual qubit
        # # as a sub-system. Evaluate the reward as the log of the mean sub-system entropy.
        # if entropies is None:
        #     entropies = util._entropy(states)
        # entropies = np.maximum(entropies.mean(axis=1), epsi)
        # rewards = np.log(epsi / entropies)
        # return rewards

    @staticmethod
    def Disentangled(states, epsi=1e-3, entropies=None):
        """ Returns an array of booleans, yielding True for every state in the batch whose
        mean "multi-qubit entropy" is smaller than `epsi`.

        Returns:
            done (np.Array): A numpy array of shape (b,).
        """
        if entropies is None:
            entropies = util.entropy(states)
        return np.all(entropies <= epsi, axis=1)

    @staticmethod
    def EntropyV2(states, subsys_A=None):
        """If `subsys_A` is given, then return the entanglement entropy for every state
        in the batch w.r.t. `subsys_A`. Otherwise, for every state in the batch the mean
        entanglement entropy of all one-qubit subsystems is calculated and returned.

        Args:
            states (np.array): A numpy array of shape (b, 2,2,...,2), giving the states in
                the batch.
            subsys_A (list[int], optional): A list of ints specifying the indices of the
                qubits to be considered as a subsystem. The subsystem is the same for
                every state in the batch. If None, defaults to half of the system.
                Default value is None.

        Returns:
            entropies (np.array): A numpy array of shape (b,), giving the entropies of the
                states in the batch.
        """
        if subsys_A is None:
            return util.entropy(states).mean()
        return util._ent_entropy(states, subsys_A)

#