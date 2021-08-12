import numpy as np
import torch
# TODO Use JAX for matrix exponentiation
# import jax
# import jax.numpy as jnp

from math import log2
from itertools import chain, combinations, product
from scipy.optimize import minimize
from scipy.linalg import expm


_BACKEND_ = np
_DTYPE_ = np.float32
_COMPLEX_DTYPE_ = np.complex64
_LOG2_CONST_ = np.float32(log2(2))


def set_backend(backend):
    global _BACKEND_
    global _DTYPE_
    global _COMPLEX_DTYPE_
    global _LOG2_CONST_

    if backend == 'numpy':
        _BACKEND_ = np
        _DTYPE_ = np.float32
        _COMPLEX_DTYPE_ = np.complex64
        _LOG2_CONST_ = np.float32(log2(2))

    elif backend == 'torch':
        _BACKEND_ = torch
        _DTYPE_ = torch.float32
        _COMPLEX_DTYPE_ = torch.complex64
        _LOG2_CONST_ = torch.Tensor([log2(2)])

    else:
        print('Unknown backend! Left unchanged.')


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
    def Entropy(cls, states, sub_sys=[0,]):
        """
        Compute the entanglement entropy of the sub system for a given state.

        Parameters
        ----------
        cls : class
        states : numpy.ndarray
        sub_sys : List[int]

        Returns
        -------
        numpy.ndarray
        """
        # The entropy is computed separately for every state of the batch.
        return ent_entropy(states, sub_sys_A=sub_sys)

    @classmethod
    def Reward(cls, states, epsi=1e-5):
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

        Returns
        -------
        numpy.ndarray
        """
        # Compute the entropy of a system by considering each individual qubit
        # as a sub-system. Evaluate the reward as the maximum sub-system entropy.
        N = int(log2(states.shape[1]))
        entropies = [cls.Entropy(states, sub_sys=[i,]) for i in range(N)]  # (q1, q2)
        entropies = _BACKEND_.vstack(entropies).max(axis=0)
        if _BACKEND_ is torch:
            entropies = entropies.values
        # Note : torch.maximum() is not supported for tensors with complex dtypes
        # entropies = _BACKEND_.maximum(entropies, epsi)
        # rewards = np.where(entropies > epsi, -1.0, 0.0)

        rewards = -entropies / _LOG2_CONST_
        # rewards = -np.log10(entropies / np.log(2))
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
    def Disentangled(cls, states, epsi):
        """
        Returns `True` if the entanglement entropy of ``state`` is smaller
        than  ``epsi``.

        Parameters
        ----------
        cls : class
        states : numpy.ndarray
        epsi : float

        Returns
        -------
        np.array[bool]
        """
        return cls.Entropy(states) <= epsi

    def __init__(self, num_qubits=2, epsi=1e-4, batch_size=1, device='cpu'):
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
        device : str
            Device to use for PyTorch
        """
        assert num_qubits >= 2
        assert device in ('gpu', 'cpu')
        self.L = int(num_qubits)
        self.Ns = 2 ** self.L
        self.num_actions = 9 * (self.L * (self.L - 1)) // 2
        self.batch_size = batch_size
        self.epsi = epsi
        self.device = device
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
        if _BACKEND_ is torch:
            self._xx = torch.from_numpy(np.kron(sigma_x, sigma_x))#.to(device=device)
            self._xy = torch.from_numpy(np.kron(sigma_x, sigma_y))#.to(device=device)
            self._xz = torch.from_numpy(np.kron(sigma_x, sigma_z))#.to(device=device)
            self._yx = torch.from_numpy(np.kron(sigma_y, sigma_x))#.to(device=device)
            self._yy = torch.from_numpy(np.kron(sigma_y, sigma_y))#.to(device=device)
            self._yz = torch.from_numpy(np.kron(sigma_y, sigma_z))#.to(device=device)
            self._zx = torch.from_numpy(np.kron(sigma_z, sigma_x))#.to(device=device)
            self._zy = torch.from_numpy(np.kron(sigma_z, sigma_y))#.to(device=device)
            self._zz = torch.from_numpy(np.kron(sigma_z, sigma_z))#.to(device=device)
        else:
            self._xx = np.kron(sigma_x, sigma_x)
            self._xy = np.kron(sigma_x, sigma_y)
            self._xz = np.kron(sigma_x, sigma_z)
            self._yx = np.kron(sigma_y, sigma_x)
            self._yy = np.kron(sigma_y, sigma_y)
            self._yz = np.kron(sigma_y, sigma_z)
            self._zx = np.kron(sigma_z, sigma_x)
            self._zy = np.kron(sigma_z, sigma_y)
            self._zz = np.kron(sigma_z, sigma_z)
        # Construct actions mapping
        qubits = combinations(range(num_qubits), 2)
        ops = product('xyz', 'xyz')
        names = []
        for ((q0, q1), (op0, op1)) in product(qubits, ops):
            names.append('{0}-{1}_{2}{3}'.format(q0, q1, op0, op1))
        self.actions = dict(enumerate(names))
        self.operators = {
            'xx': self._xx,
            'xy': self._xy,
            'xz': self._xz,
            'yx': self._yx,
            'yy': self._yy,
            'yz': self._yz,
            'zx': self._zx,
            'zy': self._zy,
            'zz': self._zz
        }
        # Internal attributes
        if _BACKEND_ is torch:
            self.__backend = 'torch'
            self.__optimization_function = QubitsEnvironment._F_torch
            self.__exp_function = torch.matrix_exp
            # self.__exp_function = jax.scipy.linalg.expm
        else:
            self.__backend = 'numpy'
            self.__optimization_function = QubitsEnvironment._F_numpy
            self.__exp_function = expm
        self.reset()

    @property
    def state(self):
        """ Returns real vector representation of the environment's state. """
        # The state of the system is represented as a concatenation of the real
        # and the imaginary parts.
        states = _BACKEND_.hstack([self._state.real, self._state.imag])
        return states

    # TODO : Check use of this method
    def set_state(self, new_state):
        """ Set the current state of the environment. """
        if self.__backend == 'torch':
            if isinstance(new_state, np.ndarray):
                new_state = torch.from_numpy(new_state.astype(np.complex64))
            elif isinstance(new_state, torch.Tensor):
                new_state = new_state.type(torch.cfloat)
        else:
            if isinstance(new_state, torch.Tensor):
                new_state = new_state.numpy().astype(np.complex64)
            elif isinstance(new_state, np.ndarray):
                new_state = new_state.astype(np.complex64)

        if self._state.shape != new_state.shape:
            raise ValueError(
                'Expected shape: {}, got {}'.format(self._state.shape, new_state.shape))

        norms = np.linalg.norm(new_state, axis=-1)
        if not np.all(np.isclose(norms, 1.0)):
            raise ValueError('State must be a unit vector!')
        self._state = new_state

    def set_random_state(self, copy=True):
        """
        Set the state of the environment to a random pure state.

        Parameters
        ----------
        copy : bool
            If True, all states in the batch are copies of a single state.
        """
        while True:
            real = np.random.uniform(-1, 1, (self.batch_size, self.Ns)).astype(np.float32)
            imag = np.random.uniform(-1, 1, (self.batch_size, self.Ns)).astype(np.float32)
            s = real + 1j * imag
            if np.all(np.linalg.norm(s, axis=1, keepdims=True) != 0.0):
                break
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        if copy:
            s[1:] = s[0]
        if self.__backend == 'torch':
            s = torch.from_numpy(s)
        self._state = s

    def reset(self):
        """ Set the state of the environment to a disentangled state. """
        zero_state = np.zeros((self.batch_size, self.Ns), dtype=np.complex64)
        zero_state[:, 0] = 1.0
        if self.__backend == 'torch':
            self._state = torch.from_numpy(zero_state)
        else:
            self._state = zero_state

    def entropy(self):
        """ Compute the entanglement entropy for the current state. """
        return self.Entropy(self._state)

    def disentangled(self):
        """ Returns True if the current state is disentangled. """
        return self.Disentangled(self._state, self.epsi)

    def reward(self):
        """ Returns the immediate reward on transitioning to the current state. """
        return self.Reward(self._state, self.epsi)

    def next_state(self, actions, angle=None):
        state, _ = self._next_state(actions, angle)
        return state

    def _next_state(self, actions, angle=None):
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
        Union[numpy.ndarray, torch.Tensor]
            The next state
        """
        if len(actions) != self.batch_size:
            raise ValueError(
                'Expected actions of shape ({},)'.format(self.batch_size))

        backend = self.__backend
        F = self.__optimization_function
        exp_function = self.__exp_function

        if backend == 'torch':
            states = self._state.detach()  # `self._state` is batch of states
        else:
            states = self._state

        nextstates = []


        if angle is None:
            # Run optimization procedure to compute the angles of rotation
            # that minimize the qubit systems' entropies
            angles = []
            for i in range(self.batch_size):
                a = actions[i]
                q0, q1, op = extract_qubits_and_op(self.actions[a])
                op = self.operators[op]
                # Compute the optimal angle of rotation for the
                # selected quantum gate.
                hist = []  # only for debuging
                res = minimize(
                    F,
                    np.pi / np.exp(1),
                    args=(states[i], op, q0, q1, hist),
                    method="Nelder-Mead",
                    tol=1e-2
                )
                if res.success:
                    angle = res.x[0]
                    print('\n\nE iter :', res.nit)
                    print('E angle:', angle)
                    print('F called with angles:', [np.round(f, 5) for f in map(float, hist)])
                else:
                    raise Exception(
                        'Optimization procedure exited with '
                        'an error.\n %s' % res.message)
                s = apply_gate_fast(
                    states[i],
                    exp_function(-1j * angle * op),
                    q0,
                    q1
                )
                nextstates.append(s)
                angles.append(angle)
        else:
            angles = [angle] * self.batch_size
            for i in range(self.batch_size):
                a = actions[i]
                q0, q1, op = extract_qubits_and_op(self.actions[a])
                op = self.operators[op]
                nextstates.append(
                    apply_gate_fast(
                        states[i],
                        exp_function(-1j * np.float32(angle) * op),
                        q0,
                        q1
                ))
        if backend == 'torch':
            return torch.stack(nextstates), torch.Tensor(angles)
        else:
            return np.stack(nextstates), np.array(angles)

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

    def step_ext(self, actions, angle=None):
        states, angles = self._next_state(actions, angle)
        self._state = states
        return self.state, self.reward(), self.disentangled(), angles

    def set_bellman_state(self):
        bellman = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex64)
        bellman = np.tile(bellman, (self.batch_size, 1))
        if self.__backend == 'torch':
            self._state = torch.from_numpy(bellman)
        else:
            self._state = bellman

    @staticmethod
    def _F_torch(angle, state, op, q0, q1):
        """ Optimization function for the angle. """
        gate = torch.matrix_exp(-1j * float(angle) * op)
        state = apply_gate_fast(state, gate, q0, q1)
        L = int(log2(len(state)))
        entropies = torch.Tensor([ent_entropy(state, [j]) for j in range(L)])
        return (entropies * entropies).sum()

    @staticmethod
    def _F_numpy(angle, state, gate, q0, q1, hist):
        """ Optimization function for the angle. """
        hist.append(angle)  # only for debuging
        gate = expm(-1j * angle * gate)
        #print(angle)
        state = apply_gate_fast(state, gate, q0, q1)
        L = int(log2(len(state)))
        entropies = np.array([ent_entropy(state, [j]) for j in range(L)])
        # entropies = ent_entropy(state)
        # print('E Entropies:', entropies.ravel())
        return np.sum(entropies)


def extract_qubits_and_op(name):
    qubits, op = name.split('_')
    q0, q1 = qubits.split('-')
    return int(q0), int(q1), op


def ent_entropy(states, sub_sys_A=None):
    # If we have a single state
    if len(states.shape) == 1:
        states = states[np.newaxis, :]
    # Number of qubits
    L = int(log2(states.shape[1]))
    # Default initialize ``sub_sys_A`` and ``sub_sys_B``
    if sub_sys_A is None:
        sub_sys_A = list(range(L // 2))
    sub_sys_B = [i for i in range(L) if i not in sub_sys_A]
    system = sub_sys_A + sub_sys_B
    sub_sys_A_size = len(sub_sys_A)
    sub_sys_B_size = L - sub_sys_A_size
    states = states.reshape((-1,) + (2,) * L)

    # Dispatch on Numpy or Torch (or JAX ?)
    if isinstance(states, torch.Tensor):
        axes = (0,) + tuple(t + 1 for t in chain(sub_sys_A, sub_sys_B))
        # TODO Check torch.permute
        for i, j in enumerate(axes):
            states = torch.transpose(states, i, j)
        states = states.reshape((-1, 2 ** sub_sys_A_size, 2 ** sub_sys_B_size))
        u, lmbda, v = torch.linalg.torch.svd(states, some=True, compute_uv=False)
        # shift lmbda to be positive within machine precision
        lmbda += torch.finfo(lmbda.dtype).eps
        return -2.0 / sub_sys_A_size * torch.einsum('ai, ai->a', lmbda ** 2, torch.log(lmbda) )

    else:
        states = np.transpose(states, (0,) + tuple(t + 1 for t in system))
        states = states.reshape((-1, 2 ** sub_sys_A_size, 2 ** sub_sys_B_size))
        lmbda = np.linalg.svd(states, full_matrices=False, compute_uv=False)
        # shift lmbda to be positive within machine precision
        lmbda += np.finfo(lmbda.dtype).eps
        return -2.0 / sub_sys_A_size * np.einsum('ai, ai->a', lmbda ** 2, np.log(lmbda) )


# @jit
def ent_entropies(states, sub_sys_A=None):
    # If we have a single state
    if len(states.shape) == 1:
        states = states[np.newaxis, :]
    # Number of qubits
    L = int(log2(states.shape[1]))
    # Default initialize ``sub_sys_A`` and ``sub_sys_B``
    if sub_sys_A is None:
        sub_sys_A = list(range(L // 2))
    sub_sys_B = [i for i in range(L) if i not in sub_sys_A]
    system = sub_sys_A + sub_sys_B
    sub_sys_A_size = len(sub_sys_A)
    sub_sys_B_size = L - sub_sys_A_size
    states = states.reshape((-1,) + (2,) * L)

    states = jnp.transpose(states, (0,) + tuple(t + 1 for t in system))
    lmbda = jnp.linalg.svd(states, full_matrices=False, compute_uv=False)
    lmbda += jnp.finfo(lmbda.dtype).eps
    Sent = -2.0 / sub_sys_A_size * jnp.einsum('ai,ai->a', lmbda**2, jnp.log(lmbda) )
    print(Sent)
    return Sent

def apply_gate_fast(state, gate, qubit1, qubit2):
    """
    Fast way to apply ``gate`` on ``(qubit1, qubit2)``
    subsystem of ``state``.

    Parameters
    ----------
    state : Union[numpy.ndarray, torch.Tensor]
        Complex vector in $R^{2^L}$
    gate : Union[numpy.ndarray, torch.Tensor]
        Gate acting on (qubit1, qubit2) subsystem
    qubit1 : int
    qubit2 : int

    Returns
    -------
    Union[numpy.ndarray, torch.Tensor]
    Complex vector in $R^{2^L}$
    """

    #print('ENV', gate.shape, state.shape)

    assert len(state) & (len(state) - 1) == 0 # Unnecessary
    L = int(log2(len(state)))
    assert L <= 10  # Temporary constraint
    psi = state.reshape((2,) * L)

    if isinstance(state, torch.Tensor):
        psi = torch.transpose(psi, 0, qubit1)
        psi = torch.transpose(psi, 1, qubit2)
    else:
        psi = np.swapaxes(psi, 0, qubit1)
        psi = np.swapaxes(psi, 1, qubit2)

    psi = psi.reshape((4, -1)) if L > 2 else psi.reshape((4,))
    phi = (gate @ psi).reshape((2,) * L)

    if isinstance(state, torch.Tensor):
        phi = torch.transpose(phi, 1, qubit2)
        phi = torch.transpose(phi, 0, qubit1)
    else:
        phi = np.swapaxes(phi, 1, qubit2)
        phi = np.swapaxes(phi, 0, qubit1)

    return phi.reshape(-1)



# ########################################################################### #
#                       D E P R E C A T E D     C O D E                       #


# def apply_gate_fast(state, gate, qubit1=0, qubit2=1):
#     """
#     Fast way to apply gates ``(gate1, gate2)`` on ``(qubit1, qubit2)``
#     subsystem of ``state``.

#     Parameters
#     ----------
#     state : Union[numpy.ndarray, torch.Tensor]
#         Complex vector in $R^{2^L}$
#     gate1 : Union[numpy.ndarray, torch.Tensor]
#         Gate acting on ``qubit1`` subsystem
#     gate2 : Union[numpy.ndarray, torch.Tensor]
#         Gate acting on ``qubit2`` subsystem
#     qubit1 : int
#     qubit2 : int

#     Returns
#     -------
#     Union[numpy.ndarray, torch.Tensor]
#     Complex vector in $R^{2^L}$
#     """
#     L = int(log2(len(state)))

#     psi = state.reshape((2,) * L)
#     gate = gate.reshape((2,2,2,2))

#     einstring = 'ij kl, kl -> ij'
#     if isinstance(state, torch.Tensor):
#         return torch.einsum(einstring, gate, psi).reshape(-1)
#     return np.einsum(einstring, gate, psi).reshape(-1)


# def _ent_entropy_numpy(states, sub_sys_A=None):
#     # If we have a single state
#     if len(states.shape) == 1:
#         states = states[np.newaxis, :]
#     # Number of qubits
#     L = int(log2(states.shape[1]))
#     # Default initialize ``sub_sys_A`` and ``sub_sys_B``
#     if sub_sys_A is None:
#         sub_sys_A = list(range(L // 2))
#     sub_sys_B = [i for i in range(L) if i not in sub_sys_A]
#     system = sub_sys_A + sub_sys_B

#     sub_sys_A_size = len(sub_sys_A)
#     sub_sys_B_size = L - sub_sys_A_size
#     states = states.reshape((-1,) + (2,) * L)
#     states = np.transpose(states, (0,) + tuple(t + 1 for t in system))
#     states = states.reshape((-1, 2 ** sub_sys_A_size, 2 ** sub_sys_B_size))

#     lmbda = np.linalg.svd(states, full_matrices=False, compute_uv=False)
#     # shift lmbda to be positive within machine precision
#     lmbda += np.finfo(lmbda.dtype).eps
#     # return -2.0/torch.log2(sub_sys_A_size) * ( lmbda ** 2 @ torch.log(lmbda) )
#     return -2.0 / sub_sys_A_size * np.einsum('ai, ai->a', lmbda ** 2, np.log(lmbda) )


# def _ent_entropy_torch(states, sub_sys_A=None):
#     # If we have a single state
#     if len(states.shape) == 1:
#         states = states[np.newaxis, :]
#     states = torch.Tensor(states)
#     # Number of qubits
#     L = int(log2(states.shape[1]))
#     # Default initialize ``sub_sys_A`` and ``sub_sys_B``
#     if sub_sys_A is None:
#         sub_sys_A = list(range(L // 2))
#     sub_sys_B = [i for i in range(L) if i not in sub_sys_A]
#     axes = (0,) + tuple(t + 1 for t in chain(sub_sys_A, sub_sys_B))
#     sub_sys_A_size = len(sub_sys_A)
#     sub_sys_B_size = L - sub_sys_A_size

#     states = states.reshape((-1,) + (2,) * L)
#     for i, j in enumerate(axes):
#         states = torch.transpose(states, i, j)
#     states = states.reshape((-1, 2 ** sub_sys_A_size, 2 ** sub_sys_B_size))

#     _, lmbda, _ = torch.linalg.torch.svd(states, compute_uv=False)
#     # shift lmbda to be positive within machine precision
#     lmbda += torch.finfo(lmbda.dtype).eps
#     return -2.0 / sub_sys_A_size * torch.einsum('ai, ai->a', lmbda ** 2, torch.log(lmbda) )



# def _random_pure_state(L, batch=1):
#     real = np.random.uniform(-1, 1, (batch, L)).astype(np.float32)
#     imag = np.random.uniform(-1, 1, (batch, L)).astype(np.float32)
#     s = real + 1j * imag
#     s /= np.linalg.norm(s, axis=1, keepdims=True)
#     if _BACKEND_ is torch:
#         s = torch.from_numpy(s).astype(_DTYPE_)
#     return s


# e = QubitsEnvironment(batch_size=32)
# e.set_random_state()
# e.entropy()
# print(e.operators)
# print(e._QubitsEnvironment__backend)
# print(e.actions)
# e.step(np.random.uniform(0, 8, size=e.batch_size).astype(np.int))