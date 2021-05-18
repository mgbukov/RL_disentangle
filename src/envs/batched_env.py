from itertools import combinations
from itertools import product

import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from scipy.optimize import minimize
from scipy.sparse.linalg import expm


class QubitsEnvironment:
    """
    Representation of a multi-qubit quantum system.
    The state of the system is represented as a numpy array giving the state
    decomposition in the computational basis.
    Action space consists of unitary gates acting on a single qubit or on a pair
    of qubits.
    The reward upon transition is defined as the negative entanglement entropy
    of the new state.
    """

    @classmethod
    def Entropy(cls, states, basis, sub_sys=[0,]):
        """
        Compute the entanglement entropy of the sub system for a given state.

        Parameters
        ----------
        cls : class
        states : numpy.ndarray
        basis : quspin.basis.spin_basis_1d
        sub_sys : List[int]

        Returns
        -------
        numpy.ndarray
        """
        # The shape of the environment state is (batch_size, system_size, 1).
        # Squeeze the third dimension and transpose in order to compute the entropy.
        states = np.squeeze(states, axis=-1).transpose(1, 0)

        # The entropy is computed separately for every state of the batch.
        ent = basis.ent_entropy(states, sub_sys_A=sub_sys, density=True, enforce_pure=True)["Sent_A"]
        ent = np.array(ent).astype(np.float32).reshape(-1)
        return ent


    @classmethod
    def Reward(cls, states, basis, epsi=1e-5):
        """
        Return the immediate reward on transition to state ``state``.
        The reward is calculated as the negative logarithm of the entropy.
        The choice depends on the fact that the reward increases exponentialy,
        when the entropy approaches 0, and thus encouraging the agent to
        disentangle the state.

        Parameters
        ----------
        cls : class
        states : numpy.ndarray
        basis : quspin.basis.spin_basis_1d

        Returns
        -------
        numpy.ndarray
        """
        # TODO :
        # What are possible/optimal choices for state_space, reward_space,action_space?
        # Choose the reward function here.

        # Compute the entropy of a system by considering each individual qubit as a
        # sub-system. Evaluate the reward as the negative maximum sub-system entropy.
        N = basis.N
        entropies = [cls.Entropy(states, basis, sub_sys=[i,]) for i in range(N)]
        entropies = np.vstack(entropies).max(axis=0)
        entropies = np.maximum(entropies, epsi)
        rewards = -(entropies / np.log(2))
        return rewards

    @classmethod
    def RewardDifferences(cls, states, prev_states, basis, epsi=1e-5):
        N = basis.N
        entropies = [cls.Entropy(states, basis, sub_sys=[i,]) for i in range(N)]
        entropies = np.vstack(entropies).max(axis=0)
        entropies = np.maximum(entropies, epsi)

        old_entropies = [cls.Entropy(prev_states, basis, sub_sys=[i,]) for i in range(N)]
        old_entropies = np.vstack(old_entropies).max(axis=0)
        old_entropies = np.maximum(old_entropies, epsi)

        rewards = (entropies - old_entropies) / np.log(2)
        return rewards


    @classmethod
    def Disentangled(cls, states, basis, epsi):
        """
        Returns `True` if the entanglement entropy of ``state`` is smaller
        than  ``epsi``.

        Parameters
        ----------
        cls : class
        states : numpy.ndarray
        basis : quspin.basis.spin_basis_1d
        epsi : float

        Returns
        -------
        np.array[bool]
        """
        return cls.Entropy(states, basis) <= epsi


    def __init__(self, num_qubits=2, epsi=1e-4, batch_size=1, pack_size=1):
        """
        Initialize a multi-Qubit system RL environment.

        Parameters
        ----------
        N : int
            Number of qubits
        epsi : float
            Threshold bellow which the system is considered disentangled
        batch_size : int
            Number of states simultaneously represented by the environment.
        pack_size : int
            Number of state copies in the batch.
        """
        assert num_qubits >= 2
        assert batch_size % pack_size == 0
        self.N = int(num_qubits)
        self.batch_size = batch_size
        self.pack_size = pack_size
        self.epsi = epsi
        self.basis = spin_basis_1d(num_qubits)

        # The shape of the environment state is (batch_size, system_size, 1)
        # The shape of the stack of gates used to transition the environment
        # to the next state is (batch_size, system_size, system_size).
        # When performing batch-matrix-multiplication with np.matmul both arguments
        # must be 3D tensors in order to perform the operation correctly.
        # If one argument is 3D and the second one is 2D, then the first argument
        # is treated as a stack of matrices and the second one is treated as
        # a conventional matrix and is broadcast accordingly.
        self._state = np.ndarray((self.batch_size, self.basis.Ns, 1), dtype=np.complex64)

        # Create hermitian operators
        self._construct_operators()
        self.num_actions = len(self.operators)

        # Reset the environment to a disentangled state.
        self.reset()


    @property
    def state(self):
        """ Return the current state of the environment. """
        # TODO
        # What are possible/optimal choices for state_space, reward_space, action_space?
        # Choose representation of the state of the system here.

        # The state of the system is represented as a concatenation of the real
        # and the imaginary parts.
        states = np.hstack([self._state.real, self._state.imag])
        states = np.squeeze(states, axis=-1)
        return states


    @state.setter
    def state(self, new_state):
        """ Set the current state of the environment. """
        assert self._state.shape == new_state.shape
        self._state = new_state.copy()


    def reset(self):
        """ Set the current state of the environment to a disentangled state.
        Set the batch of previous actions taken in the environment to None.
        Set the discount factor to its initial value.
        """
        zero_state = np.zeros(shape=(self.batch_size, self.basis.Ns, 1), dtype=np.complex64)
        zero_state[:, 0] = 1.0
        self.state = zero_state


    def set_random_state(self):
        """ Set the current state to a batch of random pure states. """
        self.reset()
        self.state = self._construct_random_pure_states(self.batch_size)


    def set_random_pack_state(self):
        """ Set the current state of the environment to a batch of random pure
        states, where every state is repeated @pack_size times.
        """
        self.reset()
        pack = self._construct_random_pure_states(self.batch_size // self.pack_size)
        self.state = np.tile(pack, (self.pack_size, 1, 1))


    def entropy(self):
        """ Compute the entanglement entropy for the current state. """
        return self.Entropy(self._state, self.basis)


    def disentangled(self):
        """ Returns True if the current state is disentangled. """
        return self.Disentangled(self._state, self.basis, self.epsi)


    def reward(self):
        """ Returns the immediate reward on transitioning to the current state. """
        return self.Reward(self._state, self.basis, self.epsi)


    def next_state(self, actions, angle=None):
        """
        Return the state resulting from applying action ``action`` to the
        current state.

        Parameters
        ----------
        action : np.array[int]
            Index in ``self.operators``
        angle : Union[float, None]
            If the angle is None, an optimizing procedure is run to
            compute the angle minimizing the entropy after applying
            the unitary gate from ``self.operators[i]``. If not none, the
            gate is applied with the specified angle.

        Returns
        -------
        numpy.ndarray
            The next state
        """
        assert len(actions) == self.batch_size
        gates = []
        angles = []
        for i in range(self.batch_size):
            U = self._unitary_gate_factory(actions[i])
            if angle is None:
                # Compute the optimal angle of rotation for the selected quantum gate.
                alpha_0 = np.pi / np.exp(1)
                F = lambda angle, U: self.Entropy(np.matmul(U(angle), self._state[np.newaxis, i]), self.basis)[0]
                res = minimize(F, alpha_0, args=(U,), method="Nelder-Mead", tol=self.epsi)
                if res.success:
                    angle = res.x[0]
                else:
                    raise Exception(
                        'Optimization procedure exited with '
                        'an error.\n %s' % res.message)
            gates.append(U(angle))
            angles.append(angle)
            angle = None
        gates = np.vstack(gates)
        return np.matmul(gates, self._state)


    # def next_state(self, actions, angle=None):
    #     assert len(actions) == self.batch_size
    #     U = [self._unitary_gate_factory(actions[i]) for i in range(self.batch_size)]

    #     def F(angle, U):
    #         g = [U[i](angle[i]) for i in range(self.batch_size)]
    #         g = np.vstack(g)
            
    #         entropies = self.Entropy(np.matmul(g, self._state), self.basis)
    #         return np.sum(entropies)

    #     if angle is None:
    #         alpha_0 = [np.pi / np.exp(1)] * self.batch_size
    #         res = minimize(F, alpha_0, args=(U,), method="Nelder-Mead", tol=1e-2)

    #         if res.success:
    #             angle = res.x
    #         else:
    #             raise Exception("Optimization procedure exited with an error.\n %s" % res.message)

    #     gates = [U[i](angle[i]) for i in range(self.batch_size)]
    #     gates = np.vstack(gates)
    #     return np.matmul(gates, self._state)


    def step(self, actions, angle=None):
        """
        Applies the specified action to the current state and transitions the
        environment to the next state.

        Parameters
        ----------
        actions : np.array[int]
            Index in ``self.operators``
        angle : Union[float, None]
            If the angle is None, an optimizing procedure is run to
            compute the angle minimizing the entropy after applying
            the unitary gate from ``self.operators[i]``. If not none, the
            gate is applied with the specified angle.

        Returns
        -------
        Tuple[numpy.ndarray, float, bool]
        (new_state, reward, done) tuple
        """
        self._state = self.next_state(actions, angle)
        return self.state, self.reward(), self.disentangled()


    def step2(self, actions, angle=None):
        next_state = self.next_state(actions, angle)
        reward = self.RewardDifferences(next_state, self._state, self.basis, self.epsi)
        self._state = next_state
        return next_state, reward, self.disentangled()


    def _construct_operators(self, q=2):
        """
        Construct multi-qubit gate operators.
        Construct dictionaries `operator_to_idx` and `idx_to_operator` mapping
        the index of an operator to its name.

            self.operators[idx] = np.array[...]
            self.idx_to_operator[idx] = "(0, 1)-xx"
            self.operator_to_idx["(0, 1)-xx"] = idx

        Parameters
        ----------
        q : int
            Number of qubits acted upon by a single gate.
            default: 2
        """
        basis = self.basis
        no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

        generators = []

        for qubits in combinations(range(self.N), q):
            for gate in product("xyz", repeat=q):
                gate = ''.join(gate)
                static = [[gate, [(1.0,) + qubits]]]
                H = hamiltonian(static, [], basis=basis, **no_checks)
                generators.append((str(qubits) + '-' + gate, H.toarray()))

        idx_to_name = {}
        name_to_idx = {}
        operators = []
        for i, (name, H) in enumerate(generators):
            idx_to_name[i] = name
            name_to_idx[name] = i
            operators.append(H)
        self.operator_to_idx = name_to_idx
        self.idx_to_operator = idx_to_name
        self.operators = np.array(operators)


    def _unitary_gate_factory(self, i):
        def U(angle):
            gate = expm(-1j * angle * self.operators[i])
            gate = np.expand_dims(gate, axis=0)
            return gate
        return U


    def _construct_random_pure_states(self, batch_size):
        """ Construct a batch of size @batch_size of random pure states. """
        N = self.basis.Ns
        real = np.random.uniform(-1, 1, (batch_size, N, 1)).astype(np.float32)
        imag = np.random.uniform(-1, 1, (batch_size, N, 1)).astype(np.float32)
        s = real + 1j * imag
        norm = np.linalg.norm(s, axis=1, keepdims=True)
        return s / norm


    def set_bellman_state(self):
        self.reset()
        bellman_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        bellman_state = np.tile(bellman_state, (self.batch_size, 1))
        bellman_state = np.expand_dims(bellman_state, axis=-1)
        self.state = bellman_state

#