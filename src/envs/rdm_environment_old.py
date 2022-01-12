import numpy as np
from itertools import combinations

_QSYSTEMS_P = {}
_QSYSTEMS_INV_P = {}
_SINGLE_ENTROPY_P = {}
for L in range(2, 15):
    _range = combinations(range(L), 2)
    r = []
    for x, y in _range:
        r.append((x,y))
        r.append((y,x))
    for q0, q1 in r:
        sysA = [q0, q1]
        sysB = [q for q in range(L) if q not in sysA]
        P = sysA + sysB
        _QSYSTEMS_P[(L, q0, q1)] = np.array(P, dtype=np.int8)
        _QSYSTEMS_INV_P[(L, q0, q1)] = np.argsort(P).astype(np.int8)
    for q0 in range(L):
        p = [0] + [q0 + 1] + [q + 1 for q in range(L) if q != q0]
        _SINGLE_ENTROPY_P[(L, q0)] = np.array(p, dtype=np.int8)


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
        states : numpy.ndarray, shape=(B, Ns)
        sub_sys : List[int]
        Returns
        -------
        numpy.ndarray, shape=(B, L)
        """
        # The entropy is computed separately for every state of the batch.
        L = int(np.log2(states.shape[1]))
        if subsys is None:
            entropies = [_ent_entropy_single(states, i, L) for i in range(L)]
            return np.stack(entropies).T
        else:
            return _ent_entropy(states, subsys, L)

    @classmethod
    def Reward(cls, states, epsi=1e-5, entropies=None):
        """
        Return the immediate reward on transition to ``states``.
        The reward is calculated as the negative logarithm of the entropy.
        The choice depends on the fact that the reward increases exponentialy,
        when the entropy approaches 0, and thus encouraging the agent to
        disentangle the state.
        Parameters
        ----------
        states : numpy.ndarray, shape=(B, Ns)
        epsi : float
            Tolerance below which state is considered disentangled.
        entropies : numpy.ndarray, shape=(B, L), default=None
            Precomputed entropies for each single qubit subsystem.
        Returns
        -------
        numpy.ndarray, shape=(B,)
        """
        # Compute the entropy of a system by considering each individual qubit
        # as a sub-system. Evaluate the reward as the maximum sub-system entropy.
        if entropies is None:
            entropies = cls.Entropy(states)
        entropies = np.maximum(entropies.mean(axis=1), epsi)
        rewards = np.log(epsi / entropies)

        # rewards = np.maximum(entropies[:, 0], epsi)

        return rewards

    @classmethod
    def Disentangled(cls, states, epsi, entropies=None):
        """
        Returns True if the entanglement entropy of all state in `states`
        is smaller than `epsi`.
        Parameters
        ----------
        states : numpy.ndarray, shape=(B, Ns)
        epsi : float
            Tolerance below which state is considered disentangled.
        entopies : numpy.ndarray, shape=(B, L)
            Precomputed entropies for each single qubit subsystem.
        Returns
        -------
        numpy.array[bool], shape=(B,)
        """
        if entropies is None:
            entropies = cls.Entropy(states)
        # return entropies[:, 0] <= epsi
        return np.all(entropies <= epsi, axis=1)

    def __init__(self, num_qubits=2, epsi=1e-4, batch_size=1):
        """
        Initializes a multi-qubit RL environment.
        Parameters
        ----------
        num_qubits : int
            Number of qubits.
        epsi : float
            Threshold bellow which the system is considered disentangled.
        batch_size : int
            Number of states simultaneously represented by the environment.
        """
        assert num_qubits >= 2
        self.L = int(num_qubits)
        self.Ns = 2 ** self.L
        self.num_actions = (self.L * (self.L - 1)) #// 2
        self.batch_size = batch_size
        self.epsi = epsi
        self.shape = (batch_size, self.Ns)
        
        _acts = list(combinations(range(num_qubits), 2))
        self.keyToAction = {}
        for i, a in enumerate(_acts):
            x,y = a
            self.keyToAction[2*i] = (x,y)
            self.keyToAction[2*i+1] = (y,x)

        self.actionToKey = {v: k for k, v in self.keyToAction.items()}
        self.reset()  # bounds `_state` and `_entropies_cache` attributes

    @property
    def actions(self):
        return list(self.keyToAction.keys())

    # TODO: Unify setter and getter. One returns array of type np.float32,
    # the other takes np.complex64 as argument.
    # Do we need setter and getter ??
    @property
    def state(self):
        # The state of the system is represented as a concatenation of the real
        # and the imaginary parts.
        # states = np.hstack([self._state.real, self._state.imag])
        # return states
        return self._state

    @state.setter
    def state(self, newstate):
        # newstate = np.atleast_2d(newstate.copy())
        # assert newstate.shape == (self.batch_size, 2 * self.Ns)
        # self._state = newstate[:, :self.Ns] + 1j * newstate[:, self.Ns:]
        # self._state = _phase_norm(self._state)
        psi = np.atleast_2d(newstate.copy())
        self._state = _phase_norm(psi)
        self._entropies_cache = self.Entropy(self._state)

    @property
    def states(self):
        return self._state
    @states.setter
    def states(self, newstate):
        psi = np.atleast_2d(newstate.copy())
        self._state = _phase_norm(psi)
        self._entropies_cache = self.Entropy(self._state)

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

    def set_random_states(self, copy=True):
        self.set_random_state(copy)

    def reset(self):
        """ Set the state of the environment to a disentangled state. """
        psi = np.zeros((self.batch_size, self.Ns), dtype=np.complex64)
        psi[:, 0] = 1.0
        self._state = psi
        self._entropies_cache = self.Entropy(self._state)

    def entropy(self):
        """ Compute the entanglement entropy for the current state. """
        return self._entropies_cache.copy()

    def disentangled(self):
        """ Returns True if the current state is disentangled. """
        return self.Disentangled(self._state, self.epsi, self._entropies_cache)

    def reward(self):
        """ Returns the immediate reward on transitioning to the current state. """
        return self.Reward(self._state, self.epsi, self._entropies_cache)

    def next_state(self, actions):
        """
        Applies `actions` to a copy of `self._state` and returns the next
        environment state. `self._state` is not modified.
        Parameters
        ----------
        actions : array_like
            Indices in `self.keyToAction`. Must have length equal to ``self.batch_size``.
        Returns
        -------
        np.ndarray[np.complex64], shape=(B, Ns)
            The next state (with phase norm).
        """
        actions = np.atleast_1d(actions)
        if len(actions) != self.batch_size:
            raise ValueError('Expected actions of shape ({},)'.format(self.batch_size))
        L = self.L
        B = self.batch_size
        states = self._state
        nextstates = np.zeros((B, self.Ns), dtype=np.complex64)
        rdms = np.zeros((B, 4, 4), dtype=np.complex64)
        qubits = [self.keyToAction[a] for a in actions]
        for i in range(B):
            q0, q1 = qubits[i]
            rdms[i] = _rdm(states[i], q0, q1, self.L)
        Us = _optimal_Us_rmds(rdms)
        for i in range(B):
            q0, q1 = qubits[i]
            nextstates[i] = _apply_unitary_gate(states[i], Us[i], q0, q1, L)
        return _phase_norm(nextstates)

    def step(self, actions):
        """
        Applies `actions` to `self._state` and transitions the environment.
        Parameters
        ----------
        actions : array_like
            Indices in `self.keyToAction`. Must have length equal to ``self.batch_size``.
        Returns
        -------
        Tuple[np.ndarray[np.complex64], np.ndarray[np.float32], np.ndarray[bool]]
            (state, reward, done) tuple.
                *state* : A copy of `self._state` after environment
                  transition (with phase norm), shape=(B, Ns).
                *reward* : Current reward, shape=(B,).
                *done* : Boolean mask of disentangled states, shape=(B,).
        """
        actions = np.atleast_1d(actions)
        if len(actions) != self.batch_size:
            raise ValueError('Expected actions of shape ({},)'.format(self.batch_size))
        L = self.L
        B = self.batch_size
        states = self._state
        rdms = np.zeros((B, 4, 4), dtype=np.complex64)
        qubits = [self.keyToAction[a] for a in actions]
        for i in range(B):
            q0, q1 = qubits[i]
            rdms[i] = _rdm(states[i], q0, q1, self.L)
        # Eigh
        rhos, Us = np.linalg.eigh(rdms)
        a = rhos[:, 0] + rhos[:, 1]  # rho_{q0-1}
        b = rhos[:, 2] + rhos[:, 3]  # rho_{q0-2}
        c = rhos[:, 0] + rhos[:, 2]  # rho_{q1-1}
        d = rhos[:, 1] + rhos[:, 3]  # rho_{q1-2}
        Sent_q0 = -a*np.log(a + np.finfo(a.dtype).eps) - b*np.log(b + np.finfo(b.dtype).eps)
        Sent_q1 = -c*np.log(c + np.finfo(c.dtype).eps) - d*np.log(d + np.finfo(d.dtype).eps)
        # Update entropies and environment states
        for i in range(B):
            q0, q1 = qubits[i]
            self._entropies_cache[i, q0] = Sent_q0[i]
            self._entropies_cache[i, q1] = Sent_q1[i]
            self._state[i] = _apply_unitary_gate(states[i], Us[i].conj().T, q0, q1, L)
        self._state = _phase_norm(self._state)
        return self._state, self.reward(), self.disentangled()

    # def step(self, actions):
    #     """
    #     Applies `actions` to the current state and transitions the
    #     environment to the next state.

    #     Parameters
    #     ----------
    #     actions : array_like
    #         Indices in `self.keyToAction`
    #     """
    #     actions = np.atleast_1d(actions)
    #     self._state = self.next_state(actions)
    #     # Update entropies
    #     L = self.L
    #     for i, a in enumerate(actions):
    #         q0, q1 = self.keyToAction[a]
    #         self._entropies_cache[i, q0] = _ent_entropy_single(self._state[i], q0, L)
    #         self._entropies_cache[i, q1] = _ent_entropy_single(self._state[i], q1, L)
    #     return self._state, self.reward(), self.disentangled()

    @classmethod
    def compute_best_path_single_state(cls, num_qubits, epsi, state, steps=None):
        env = cls(num_qubits, epsi=epsi, batch_size=1)
        env.state = state
        frontier = [{"path":[], "state":env.state.copy()}]
        result = []
        done = False

        it = 0
        while True:
            new_frontier = []
            for d in frontier:
                path = d["path"]
                st = d["state"]
                env.state = st
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
                        state_stack = np.hstack([next_st.real, next_st.imag])
                        new_frontier.extend([{"path":new_path, "state":state_stack}])
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
        for st in self.state:
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


def _apply_unitary_gate(state, U, qubit1, qubit2, L):
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
    numpy.ndarray[complex64]
    """
    psi = state.reshape((2,) * L)
    # Swap qubits
    psi = np.transpose(psi, _QSYSTEMS_P[(L, qubit1, qubit2)])
    # Apply U
    psi = psi.reshape((4, -1)) if L > 2 else psi.reshape((4,))
    psi = (U @ psi).reshape((2,) * L)
    # Swap qubits back
    psi = np.transpose(psi, _QSYSTEMS_INV_P[(L, qubit1, qubit2)])
    return psi.reshape(-1)


def _rdm_entropy(rdm):
    """ Returns the entanglement entropy of 4x4 density matrix."""
    rdm_A = np.trace(rdm.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    lmbda = np.linalg.svd(rdm_A, full_matrices=False, compute_uv=False)
    lmbda += np.finfo(lmbda.dtypr).eps
    return -np.sum(lmbda * np.log(lmbda), axis=1)


def _rdm(state, qubit1, qubit2, L):
    """ Returns the reduced density matrix for `qubit1` and `qubit2` of `state`. """
    psi = state.reshape((2,) * L).transpose(_QSYSTEMS_P[(L, qubit1, qubit2)])
    psi = psi.reshape(4, 2 ** (L - 2))
    rdm = psi @ psi.T.conj()
    return rdm


def _optimal_U(state, qubit1, qubit2):
    rdm = _rdm(state, qubit1, qubit2)
    _, U = np.linalg.eigh(rdm)
    return U.conj().T


def _optimal_Us_rmds(rdms):
    _, U = np.linalg.eigh(rdms)
    return np.swapaxes(U.conj(), 1, 2)


def _ent_entropy_single(states, q, L):
    states = np.atleast_2d(states).reshape((-1,) + (2,) * L)
    states = np.transpose(states, _SINGLE_ENTROPY_P[(L, q)])
    states = states.reshape((-1, 2, 2 ** (L - 1)))
    lmbda = np.linalg.svd(states, full_matrices=False, compute_uv=False)
    lmbda += np.finfo(lmbda.dtype).eps
    return -2.0 * np.einsum('ai, ai->a', lmbda ** 2, np.log(lmbda))


def _ent_entropy(states, subsys_A, L):
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
    subsys_B = [i for i in range(L) if i not in subsys_A]
    system = subsys_A + subsys_B
    subsys_A_size = len(subsys_A)
    subsys_B_size = L - subsys_A_size
    print("states.shape before:", states.shape)
    states = states.reshape((-1,) + (2,) * L)
    states = np.transpose(states, (0,) + tuple(t + 1 for t in system))
    states = states.reshape((-1, 2 ** subsys_A_size, 2 ** subsys_B_size))
    print("states.shape after:", states.shape)
    lmbda = np.linalg.svd(states, full_matrices=False, compute_uv=False)
    print("lmbda.shape:", lmbda.shape)
    # shift lmbda to be positive within machine precision
    lmbda += np.finfo(lmbda.dtype).eps
    return -2.0 / subsys_A_size * np.einsum('ai, ai->a', lmbda ** 2, np.log(lmbda))


def _phase_norm(state):
    phi = np.angle(state[:, 0])
    z = np.expand_dims(np.cos(phi) - 1j * np.sin(phi), axis=1)
    return state * z


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    #                           ROLLOUT  TEST
    #
    # E = QubitsEnvironment(8, batch_size=256)
    # actions = np.random.uniform(0, E.num_actions, (100, 256)).astype(np.int32)
    # for a in actions:
    #     states, rewards, done = E.step(a)


    # -------------------------------------------------------------------------
    #                           EQUIVALENCE  TEST
    #
    # np.random.seed(32)
    # E = QubitsEnvironment(6, batch_size=8)
    # E.set_random_state()

    # State = E._state
    # actions = np.random.uniform(0, E.num_actions, (20, 8)).astype(np.int32)
    # prev_states, now_states = [], []
    # prev_ent, now_ent = [], []
    # for a in actions:
    #     states, rewards, done = E.step(a)
    #     prev_states.append(states.copy())
    #     prev_ent.append(E._entropies_cache.copy())

    # E._state = State.copy()
    # E._entropies_cache = E.Entropy(E._state)
    # for a in actions:
    #     states, rewards, done = E.transition(a)
    #     now_states.append(states.copy())
    #     now_ent.append(E._entropies_cache.copy())
    # now_ent = np.array(now_ent)
    # prev_ent = np.array(prev_ent)
    # for i in range(len(actions)):
    #     print('\n', np.isclose(now_states[i].round(7), prev_states[i].round(7), atol=1e-7))
    #     print('\n', np.isclose(now_ent[i], prev_ent[i], atol=1e-7))
    pass
