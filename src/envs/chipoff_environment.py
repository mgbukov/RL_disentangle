from itertools import permutations

import numpy as np

from src.envs import util
# from src.envs.batch_transpose import cy_transpose_batch
from src.envs.util import _QSYSTEMS_P, _QSYSTEMS_INV_P


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

    The terminal state of the environment is reached when the leftmost qubit
    is disentangled from the rest of the system.

    Attributes:
        L (int): Number of qubits in the quantum state.
        epsi (float): Threshold for disentangling.
        batch_size (int): Number of quantum states in the environment.
        num_actions (int): Number of allowable actions.
        actions (dict): A mapping from action index to action name.
        actToKey (dict): A mapping from action name to action index.
        shape (Tuple[Int]): A tuple of int giving the shape of the tensor representing the
            environment.
    """

    def __init__(self, num_qubits=2, epsi=1e-4, batch_size=1):
        """Initialize an environment with a batch of multi-qubit states.

        Args:
            num_qubits (int, optional): Number of qubits in a single state.
                Default value is 2.
            epsi (float, optional): Threshold below which the system is considered disentangled.
                Default value is 1e-4.
            batch_size (int, optional): Number of states in the environment.
                Default value is 1.
        """
        assert num_qubits >= 2
        self.L = num_qubits
        self.epsi = epsi
        self.batch_size = batch_size

        # The action space consists of all possible pairs of qubits.
        self.num_actions = self.L - 1
        self.actions = dict(enumerate((0, i) for i in range(1, num_qubits)))
        self.actToKey = {v:k for k, v in self.actions.items()}

        # The states of the system are represented as a numpy array of shape (b, 2,2,...,2).
        self.shape = (batch_size,) + (2,) * num_qubits
        self._states = np.zeros(self.shape, dtype=np.complex64)
        self.reset()

    def reset(self):
        """Prepare all states in the batch as | 000..00 >"""
        psi = np.zeros((self.batch_size, 2 ** self.L), dtype=np.complex64)
        psi[:, 0] = 1.0
        self._states = psi.reshape(self.shape)

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
        self._states = util._phase_norm(newstates)

    def set_random_states(self, copy=False):
        """Set all states of the environment to random pure states. If copy is True,
        all states in the batch are copies of a single state.
        Compute the entropy of the states and cache them for later use.
        """
        if copy:
            psi = util._random_pure_state(self.L)
            psi = np.tile(psi, (self.batch_size, 1))
        else:
            psi = util._random_batch(self.L, self.batch_size)
        self._states = util._phase_norm(psi.reshape(self.shape))

    def step(self, actions):
        """Applies a batch of actions to the states batch and transitions the environment
        to the next batch of states. This function modifies the internal representation
        of the environment inplace.

        Args:
            actions (np.Array): A numpy array of shape (b, 1), giving the action selected
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
        util._transpose_batch_inplace(batch, qubit_indices, self.L)
        batch = np.ascontiguousarray(batch)
        # ----
        # Compute 2x2 reduced density matrices
        batch = batch.reshape(B, 4, 2 ** (L - 2))
        rdms = batch @ np.transpose(batch.conj(), [0, 2, 1])
        # ----
        # Compute single qubit entropies
        _, Us = np.linalg.eigh(rdms)
        Us = np.swapaxes(Us.conj(), 1, 2)
        # ----
        # Apply unitary gates
        batch = (Us @ batch).reshape(self.shape)
        # ----
        # Undo qubit permutations
        util._transpose_batch_inplace(batch, qubit_indices, L, inverse=True)
        batch = np.ascontiguousarray(batch)
        self._states = util._phase_norm(batch)
        return self._states, self.reward(), self.disentangled()

    def peek(self, actions):
        """Applies a batch of actions to the states batch and peeks at the next states of
        the environment. This function has the same functionality as the `step` function,
        but it does not modify the internal representation of the environment.
        """
        # Cache the current environment representation.
        batch = self._states.copy()
        # Transition the environment into the new states after taking `actions`.
        res = self.step(actions)
        # Reverse environment state using the cached values.
        self._states = batch
        return res

    def entropy(self):
        """Compute the entanglement entropy for the current states.

        Returns:
            entropies (np.array): A numpy array of shape (b, L), giving single-qubit entropies.
        """
        return util._ent_entropy(self.states, [0]).reshape(-1, 1)

    def reward(self):
        """Returns the rewards on transitioning into the current states of the batch.

        Returns:
            rewards (np.Array): A numpy array of shape (b,).
        """
        return self.Reward(self._states, self.epsi)

    def disentangled(self):
        """ Returns an array indicating which states of the batch are disentangled.

        Returns:
            done (np.Array): A numpy array of shape (b,).
        """
        return self.Disentangled(self._states, self.epsi)

    @staticmethod
    def Entropy(states):
        """
        Args:
            states (np.array): A numpy array of shape (b, 2,2,...,2), giving the states in
                the batch.

        Returns:
            entropies (np.array): A numpy array of shape (b,), giving leftmost qubits entropy.
        """
        return util._ent_entropy(states, [0])

    @classmethod
    def Reward(cls, states, epsi=1e-6, entropies=None):
        """
        Returns:
            rewards (np.Array): A numpy array of shape (b,).
        """
        return (-1. + 101. * cls.Disentangled(states, epsi, entropies)).astype(np.float32)

    @staticmethod
    def Disentangled(states, epsi=1e-6, entropies=None):
        """ Returns an array of booleans, yielding True for every state in the
        batch whose leftmost qubit entropy is smaller than `epsi`.

        Returns:
            done (np.Array): A numpy array of shape (b,).
        """
        if entropies is None:
            entropies = util._ent_entropy(states, [0])
        return entropies <= epsi
