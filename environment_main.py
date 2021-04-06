import numpy as np
from itertools import combinations
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from scipy.sparse.linalg import expm
from scipy.optimize import minimize


class QubitSystem:
    """
    Representation of a multi-qubit quantum system.
    The state of the system is represented as a numpy array giving the
    state decomposition in the computational basis.
    Action space consists of unitary gates acting on a single qubit or
    on a pair of qubits. The reward upon transition is defined as the
    negative entanglement entropy of the new state.
    """

    @classmethod
    def Entropy(cls, state, basis, sub_sys=[0]):
        """
        Compute the entanglement entropy of the sub system for a given state.

        Parameters
        ----------
        cls : class
        state : numpy.ndarray
        basis : quspin.basis.spin_basis_1d
        sub_sys : List[int]

        Returns
        -------
        np.float32
        """
        ent = basis.ent_entropy(state, sub_sys_A=sub_sys, density=True)["Sent_A"]
        return ent.astype(np.float32).reshape(1)[0]

    @classmethod
    def Reward(cls, state, basis):
        """
        Return the immediate reward on transition to state ``state``.

        Parameters
        ----------
        cls : class
        state : numpy.ndarray
        basis : quspin.basis.spin_basis_1d

        Returns
        -------
        np.float32
        """
        # TODO :
        # What are possible/optimal choices for state_space, reward_space, action_space?
        # Choose the reward function here.
        return -cls.Entropy(state, basis)

    @classmethod
    def Disentangled(cls, state, basis, tol):
        """
        Returns `True` if the entanglement entropy of ``state`` is smaller
        than  ``tol``.

        Parameters
        ----------
        cls : class
        state : numpy.ndarray
        basis : quspin.basis.spin_basis_1d
        tol : float

        Returns
        -------
        bool
        """
        return cls.Entropy(state, basis) <= tol

    def __init__(self, N=2, entanglement_tol=1e-4):
        """
        Initialize a multi-Qubit system RL environment.

        Parameters
        ----------
        N : int
            Number of qubits
        entanglement_tol : float
            Threshold bellow which the system is considered disentangled
        """
        self.N = int(N)
        self.entanglement_tol = entanglement_tol
        self.basis = spin_basis_1d(N)
        self.reset()
        self._construct_operators()

    @property
    def state(self):
        """ Return the current state of the environment. """
        # TODO
        # What are possible/optimal choices for state_space, reward_space, action_space?
        # Choose representation of the state of the system here.
        return self._state

    @state.setter
    def state(self, new_state):
        """ Set the current state of the environment. """
        assert len(new_state) == self.basis.Ns
        self._state = new_state.copy()

    def set_random_state(self):
        """ Set the current state of the environment to a random pure state. """
        self._state = self._construct_random_pure_state()

    def reset(self):
        """ Set the current state of the environment to a disentangled state. """
        self._state = np.zeros(shape=self.basis.Ns)
        self._state[0] = 1.0

    def entropy(self):
        """ Compute the entanglement entropy for the current state. """
        return self.Entropy(self._state, self.basis)

    def disentangled(self):
        """ Returns True if the current state is disentangled. """
        return self.Disentangled(self._state, self.basis, self.entanglement_tol)

    def reward(self):
        """ Returns the immediate reward on transitioning to the current state. """
        return self.Reward(self._state, self.basis)

    def next_state(self, action, angle=None):
        """
        Return the state resulting from applying action ``action`` to the
        current state.

        Parameters
        ----------
        action : int
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
        U = self._unitary_gate_factory(action)
        if angle is None:
            # Compute the optimal angle of rotation for the selected quantum gate.
            alpha_0 = np.pi / np.exp(1)
            F = lambda angle, U: self.Entropy(U(angle).dot(self._state), self.basis),
            res = minimize(F, alpha_0, args=(U,), method="Nelder-Mead", tol=1e-6)
            if res.success:
                angle = res.x[0]
            else:
                raise Exception(
                    'Optimization procedure exited with '
                    'an error.\n %s' % res.message)
        return U(angle).dot(self._state)

    def step(self, action, angle=None):
        """
        Applies the specified action to the current state and transitions the
        environment to the next state.

        Parameters
        ----------
        action : int
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
        self._state = self.next_state(action, angle)
        return self._state, self.reward(), self.disentangled()

    def _construct_operators(self):
        basis = self.basis
        no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

        generators = []
        # Single quibit gate generators
        q = [[1.0, 0]]
        for i in range(self.N):
            q[0][1] = i
            H_x = hamiltonian([['x', q]], [], basis=basis, **no_checks)
            H_y = hamiltonian([['y', q]], [], basis=basis, **no_checks)
            H_z = hamiltonian([['z', q]], [], basis=basis, **no_checks)
            p = '%d ' % i
            generators.append((p + 'x', H_x.toarray()))
            generators.append((p + 'y', H_y.toarray()))
            generators.append((p + 'z', H_z.toarray()))

        if self.N > 1:
            # Two-qubit gate generators
            q = [[1.0, 0, 0]]
            for t in combinations(range(self.N), 2):
                q[0][1:] = t
                H_xx = hamiltonian([['xx', q]], [], basis=basis, **no_checks)
                H_yy = hamiltonian([['yy', q]], [], basis=basis, **no_checks)
                H_zz = hamiltonian([['zz', q]], [], basis=basis, **no_checks)
                p = '%d %d ' % t
                generators.append((p + 'xx', H_xx.toarray()))
                generators.append((p + 'yy', H_yy.toarray()))
                generators.append((p + 'zz', H_zz.toarray()))

        # What about more than two-gate generators?
        #
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
            return expm(-1j * angle * self.operators[i])
        return U

    def _construct_random_pure_state(self):
        N = self.basis.Ns
        real = np.random.uniform(-1, 1, N).astype(np.float32)
        imag = np.random.uniform(-1, 1, N).astype(np.float32)
        s = real + 1j * imag
        norm = np.sqrt(np.sum(s.conj() * s).real)
        return s / norm

#