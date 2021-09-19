import numpy as np
from math import log2
from itertools import combinations


class QubitsEnvironment:
    """
    Representation of a multi-qubit quantum system.
    The state of the system is represented as a numpy array giving the state
    decomposition in the computational basis.
    Action space consists of unitary gates acting on a pair of qubits.
    The reward upon transition is defined as the negative entanglement entropy
    of the new state.
    """

    @classmethod
    def Entropy(cls, states, subsys=None):
        """
        Compute the entanglement entropy of the sub system for a given state.
        If `subsys` is None the mean entanglement entropy of every
        1-qubit subsystem is returned.

        Parameters
        ----------
        states : numpy.ndarray
        sub_sys : List[int]

        Returns
        -------
        numpy.ndarray
        """
        # The entropy is computed separately for every state of the batch.
        if subsys is None:
            L = int(np.log2(states.shape[1]))
            entropies = [_ent_entropy(states, [i]) for i in range(L)]
            return np.stack(entropies).T
        else:
            return _ent_entropy(states, subsys)

    @classmethod
    def Reward(cls, states, epsi=1e-5, entropies=None):
        """
        Return the immediate reward on transition to state ``state``.
        The reward is calculated as the negative logarithm of the entropy.
        The choice depends on the fact that the reward increases exponentialy,
        when the entropy approaches 0, and thus encouraging the agent to
        disentangle the state.

        Parameters
        ----------
        states : numpy.ndarray
            Batch of states, dimension = (B, Ns)
        epsi : float
            Tolerance below which state is considered disentangled
        entropies : numpy.ndarray, default=None
            Precomputed entropies

        Returns
        -------
        numpy.ndarray
            Shape = (B,)
        """
        # Compute the entropy of a system by considering each individual qubit
        # as a sub-system. Evaluate the reward as the maximum sub-system entropy.
        if entropies is None:
            entropies = cls.Entropy(states)
        entropies = entropies.mean(axis=1)
        # rewards = -entropies / 0.6931471805599453  # log(2)
        entropies = np.maximum(entropies, epsi)
        rewards = np.log(epsi / entropies)
        return rewards

    @classmethod
    def Disentangled(cls, states, epsi, entropies=None):
        """
        Returns True if the entanglement entropy of all state in `states`
        is smaller than `epsi`.

        Parameters
        ----------
        states : numpy.ndarray
        epsi : float
        entopies : numpy.ndarray
            Precomputed entropies for each subsystem

        Returns
        -------
        np.array[bool]
        """
        if entropies is None:
            entropies = cls.Entropy(states)
        return np.all(entropies <= epsi, axis=1)

    def __init__(self, num_qubits=2, epsi=1e-4, batch_size=1):
        """
        Initialize a multi-Qubit system RL environment.

        Parameters
        ----------
        num_qubits : int
            Number of qubits
        epsi : float
            Threshold bellow which the system is considered disentangled
        batch_size : int
            Number of states simultaneously represented by the environment.
        """
        assert num_qubits >= 2
        self.L = int(num_qubits)
        self.Ns = 2 ** self.L
        self.num_actions = (self.L * (self.L - 1)) // 2
        self.batch_size = batch_size
        self.epsi = epsi
        self.actions = dict(enumerate(combinations(range(num_qubits), 2)))
        self.reset()  # bounds `_state` and `_entropies_cache` attributes

    @property
    def state(self):
        """ Return the current state of the environment. """
        # TODO
        # What are possible/optimal choices for state_space, reward_space, action_space?
        # Choose representation of the state of the system here.

        # The state of the system is represented as a concatenation of the real
        # and the imaginary parts.
        states = np.hstack([self._state.real, self._state.imag])
        return states


    def set_random_state(self, copy=False):
        """
        Sets the state of the environment to a random pure state.

        Parameters
        ----------
        copy : bool
            If True, all states in the batch are copies of a single state.
        """
        if copy:
            psi = _random_pure_state(self.L)
            self._state = np.tile(psi, (self.batch_size, 1))
        else:
            self._state = _random_batch(self.L, self.batch_size)
        self._state = _phase_norm(self._state)
        self._entropies_cache = self.Entropy(self._state)

    def reset(self):
        """ Set the state of the environment to a disentangled state. """
        psi = np.zeros((self.batch_size, self.Ns), dtype=np.complex64)
        psi[:, 0] = 1.0
        self._state = psi
        self._entropies_cache = self.Entropy(self._state)

    def entropy(self):
        """ Compute the entanglement entropy for the current state. """
        # return self._entropies_cache
        entropies = self.Entropy(self._state)
        entropies = entropies.mean(axis=1)
        return entropies

    def disentangled(self):
        """ Returns True if the current state is disentangled. """
        # return self.Disentangled(self._state, self.epsi, self._entropies_cache)
        return self.Disentangled(self._state, self.epsi)

    def reward(self):
        """ Returns the immediate reward on transitioning to the current state. """
        # return self.Reward(self._state, self.epsi, self._entropies_cache)
        return self.Reward(self._state, self.epsi)

    def _next_state(self, actions):
        """
        Applies `actions` to a copy of `self._state` and returns the next
        environment state. `self._state` is not modified.

        Parameters
        ----------
        actions : array_like
            Indices in `self.actions`
        """
        actions = np.atleast_1d(actions)
        if len(actions) != self.batch_size:
            raise ValueError('Expected actions of shape ({},)'.format(self.batch_size))

        nextstates = []
        for i, a in enumerate(actions):
            q0, q1 = self.actions[a]
            psi = self._state[i]
            U = _optimal_U(psi, q0, q1)
            phi = _apply_unitary_gate(psi, U, q0, q1)
            nextstates.append(phi)
        return np.stack(nextstates)
    
    def next_state(self, actions):
        return _phase_norm(self._next_state(actions))

    def step(self, actions):
        """
        Applies `actions` to the current state and transitions the
        environment to the next state.

        Parameters
        ----------
        actions : array_like
            Indices in `self.actions`
        """
        actions = np.atleast_1d(actions)
        self._state = self.next_state(actions)
        # Update entropies
        for i, a in enumerate(actions):
            q0, q1 = self.actions[a]
            self._entropies_cache[i, q0] = _ent_entropy(self._state[i], [q0])
            self._entropies_cache[i, q1] = _ent_entropy(self._state[i], [q1])
        return self._state, self.reward(), self.disentangled()

    @classmethod
    def compute_best_path_single_state(cls, num_qubits, epsi, state, steps=None):
        env = cls(num_qubits, epsi=epsi, batch_size=1)
        env._state = state
        frontier = [{"path":[], "state":env._state.copy()}]
        result = []
        done = False

        it = 0
        while True:
            new_frontier = []
            for d in frontier:
                path = d["path"]
                st = d["state"]
                env._state = st
                if env.disentangled():
                    # path.extend([-1] * (steps-i)) #????
                    result.append(path)
                    done = True
                    continue
                if not done:
                    for a in range(env.num_actions):
                        next_st = env.next_state([a])
                        new_path = path.copy()
                        new_path.append(a)
                        new_frontier.extend([{"path":new_path, "state":next_st}])
            frontier = new_frontier
            if done:
                break

            it += 1
            if steps is not None and it > steps:
                break
            # else:
            #     print("searching paths of length: ", it)
        
        return result, done

    def compute_best_path(self, steps=None):
        cache = self._state.copy()
        res = []
        for st in self._state:
            result, done = self.compute_best_path_single_state(self.L, self.epsi, st[np.newaxis], steps)
            res.append({"paths":result, "success":done})
        
        self._state = cache
        return res




def _random_pure_state(L):
    # Sample from Gaussian distribution, as it gives uniformly distibuted
    # points in L dimensional unit sphere
    psi = np.random.randn(2 ** L) + 1j * np.random.randn(2 ** L)
    # Normalize the state vector
    psi /= np.linalg.norm(psi)
    return psi.astype(np.complex64)


def _random_batch(L, batch_size=1):
    states = np.random.randn(batch_size, 2 ** L) + 1j * np.random.randn(batch_size, 2 ** L)
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return states.astype(np.complex64)


def _apply_unitary_gate(state, U, qubit1, qubit2):
    """
    Applies ``U`` on ``(qubit1, qubit2)`` subsystem of ``state``.

    Parameters
    ----------
    state : numpy.ndarray
        Complex vector in $R^{2^L}$
    U : numpy.ndarray
        Unitary gate acting on (qubit1, qubit2) subsystem
    qubit1 : int
    qubit2 : int

    Returns
    -------
    numpy.ndarray
    Complex vector in $R^{2^L}$
    """
    L = int(log2(len(state)))
    psi = state.reshape((2,) * L)
    # Swap qubits
    psi = np.swapaxes(psi, 0, qubit1)
    psi = np.swapaxes(psi, 1, qubit2)
    # Apply U
    psi = psi.reshape((4, -1)) if L > 2 else psi.reshape((4,))
    phi = (U @ psi).reshape((2,) * L)
    # Swap qubits back
    phi = np.swapaxes(phi, 1, qubit2)
    phi = np.swapaxes(phi, 0, qubit1)
    return phi.reshape(-1)


def _rdm_entropy(rdm):
    """ Returns the entanglement entropy of 4x4 density matrix."""
    rdm_A = np.trace(rdm.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    lmbda = np.linalg.svd(rdm_A, full_matrices=False, compute_uv=False)
    lmbda += np.finfo(lmbda.dtypr).eps
    return -np.sum(lmbda * np.log(lmbda), axis=1)


def _rdm(state, qubit1, qubit2):
    """ Returns the reduced density matrix for `qubit1` and `qubit2` of `state`. """
    L = int(log2(len(state)))
    system = [qubit1, qubit2] + [q for q in range(L) if q not in (qubit1, qubit2)]
    psi = state.reshape((2,) * L).transpose(system)
    psi = psi.reshape(4, 2 ** (L - 2))
    rdm = psi @ psi.T.conj()
    return rdm


def _optimal_U(state, qubit1, qubit2):
    rdm = _rdm(state, qubit1, qubit2)
    _, U = np.linalg.eigh(rdm)
    return U.conj().T


def _ent_entropy(states, subsys_A=None):
    """
    Returns the entanglement entropy for every state in the batch w.r.t
    `subsys_A`.

    Parameters
    ----------
    states : npumpy.ndarray
        Batch of states or single state, shape = (batch_size, state_size).
    subsys_A : array_like
        Same for every state in the batch. If None, defaults to half subsystem.
    """
    states = np.atleast_2d(states)
    L = int(log2(states.shape[1]))
    # Default initialize ``sub_sys_A`` and ``sub_sys_B``
    if subsys_A is None:
        subsys_A = list(range(L // 2))
    subsys_B = [i for i in range(L) if i not in subsys_A]
    system = subsys_A + subsys_B
    subsys_A_size = len(subsys_A)
    subsys_B_size = L - subsys_A_size
    states = states.reshape((-1,) + (2,) * L)
    states = np.transpose(states, (0,) + tuple(t + 1 for t in system))
    states = states.reshape((-1, 2 ** subsys_A_size, 2 ** subsys_B_size))
    lmbda = np.linalg.svd(states, full_matrices=False, compute_uv=False)
    # shift lmbda to be positive within machine precision
    lmbda += np.finfo(lmbda.dtype).eps
    return -2.0 / subsys_A_size * np.einsum('ai, ai->a', lmbda ** 2, np.log(lmbda))


def _phase_norm(state):
    st = state.copy().astype(np.complex128)
    phi = np.angle(st[:, 0])
    z = np.expand_dims(np.cos(phi) - 1j * np.sin(phi), axis=1)
    return st * z



if __name__ == '__main__':
    np.random.seed(44)
    pass
