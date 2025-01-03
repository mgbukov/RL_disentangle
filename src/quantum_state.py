from itertools import permutations, combinations
import numpy as np

from .mpslib import generate_random_MPS, MPS_to_state


class VectorQuantumState:
    """VectorQuantumState is a vectorized quantum state for parallel simulation.
    Each quantum state simulates the evolution of a system of qubits separately.
    All quantum states in the Vector system have the same number of qubits.
    """

    def __init__(self, num_qubits, num_envs, act_space="reduced",
                 state_generator="haar_full", **generator_kwargs):
        """Init a vector quantum system.

        Args:
            num_qubits: int
                Number of qubits in each of the quantum states.
            num_envs: int
                Number of quantum states in the vectorized environment.
            act_space: str, optional
                Whether to use the full or the reduced action space.
                One of ["full", "reduced"]. Default is "reduced".
            state_generator: str, optional
                Choice of state distribution from which states are drawn on
                reset() call. Currently supported:
                    haar_full:  States are drawn from `num_qubits` dimensional
                                Hilbert space.
                    haar_geom:  States are drawn from 2 to `num_qubits`
                                dimensional Hilbert space. The dimension is
                                chosen from geometric distribution.
                    haar_unif:  States are drawn from 2 to `num_qubits`
                                dimensional Hilbert space and each dimension
                                has equal probability of being sampled.
                    mps:        Matrix product states, parametarized by maximum
                                bond dimension `chi_max`.
            generator_kwargs:
                Keyword arguments passed to state generator function.
        """
        assert num_qubits >= 2
        self.num_qubits = num_qubits
        self.num_envs = num_envs

        # The action space consists of all possible pairs of qubits.
        if act_space == "full":
            self.actions = dict(enumerate(permutations(range(num_qubits), 2)))
        elif act_space == "reduced":
            self.actions = dict(enumerate(combinations(range(num_qubits), 2)))
        else:
            raise ValueError
        self.actToKey = {v:k for k, v in self.actions.items()}
        self.num_actions = len(self.actions)

        # Every system in the vectorized environment is represented as a numpy
        # array of complex numbers with shape (b, 2, 2, ..., 2).
        self.shape = (num_envs,) + (2,) * num_qubits
        self._states = np.zeros(self.shape, dtype=np.complex64)

        # Every time `self.reset()` is called a new state must be generatod.
        # The generation of new states is handled by a function,
        # which is selected using `state_generator` and `generator_kwargs`
        # paratemers.
        self.state_generator_name = state_generator
        if state_generator == "haar_full":
            self.state_generator = sample_haar_full
        elif state_generator == "haar_geom":
            self.state_generator = sample_haar_geom
        elif state_generator == "haar_unif":
            self.state_generator = sample_haar_unif
        elif state_generator == "mps":
            self.state_generator = sample_mps
        else:
            raise ValueError("`state_generator` must be one of ('haar_full', " \
                             "'haar_geom', 'haar_unif', 'mps')")
        self.state_generator_kwargs = generator_kwargs["generator_kwargs"]

        # Store the entanglement of every system for faster retrieval.
        self.entanglements = np.zeros((num_envs, num_qubits), dtype=np.float32)

        # Dynamic attributes - assinged only after call to `self.apply()`:
        #   Unitary gates applied in last action
        self.Us_ = None
        #   Boolean indicators of (i,j)->(j,i) swaps before U*|psi>
        self.preswaps_ = None
        #   Boolean indicators of (j,i)->(i,j) swaps after U*|psi>
        self.postswaps_ = None

    @property
    def states(self): return self._states

    @states.setter
    def states(self, newstates):
        if newstates.shape[1:] != self.shape[1:]:
            raise ValueError
        self.num_envs = newstates.shape[0]
        self.shape = newstates.shape
        self._states = phase_norm(newstates)
        self.entanglements = entropy(self._states)

    def set_random_states_(self):
        """Set all states of the vectorized environment to full dimensional =
           Haar random states."""
        N, Q = self.num_envs, self.num_qubits
        states = np.random.randn(N, 2 ** Q) + 1j * np.random.randn(N, 2 ** Q)
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        states = states.astype(np.complex64)
        self._states = phase_norm(states.reshape(self.shape))
        self.entanglements = entropy(self._states)

    def apply(self, acts):
        acts = np.atleast_1d(acts)
        if len(acts) != self.num_envs:
            raise ValueError(f"Expected array with shape=({self.num_envs},)")
        N, Q = self.num_envs, self.num_qubits
        EPS = np.finfo(self.entanglements.dtype).eps
        batch = self._states
        ax0 = np.arange(N)

        # Move qubits which are modified by `acts` at indices (0, 1).
        # We will apply the action so that the leading quibt is the one that
        # has more entanglement entropy.
        _qubit_indices = np.array([self.actions[a] for a in acts])
        _ent_relation = (self.entanglements[ax0, _qubit_indices[:, 0]] >= \
                         self.entanglements[ax0, _qubit_indices[:, 1]] + EPS)
        qubit_indices = np.where(
            _ent_relation[:, None], _qubit_indices, _qubit_indices[:,::-1])
        self.preswaps_ = ~_ent_relation
        permute_qubits(batch, qubit_indices, Q, inverse=False)

        # Compute 4x4 reduced density matrices
        batch = batch.reshape(N, 4, 2 ** (Q - 2))
        rdms = batch @ np.transpose(batch.conj(), [0, 2, 1])
        rdms += np.finfo(rdms.dtype).eps * np.diag([0.0, 1.0, 2.0, 4.0])

        # Compute single qubit entanglements.
        rhos, Us = np.linalg.eigh(rdms)
        for n in range(N):
            max_col = np.abs(Us[n]).argmax(axis=0)
            for k in range(4):
                Us[n, :, k] *= np.exp(-1j * np.angle(Us[n, max_col[k], k]))
        Us = np.swapaxes(Us.conj(), 1, 2)
        self.Us_ = Us

        # Apply unitary gates.
        batch = (Us @ batch)
        batch = batch.reshape(self.shape)
        # Undo qubit permutations
        permute_qubits(batch, qubit_indices, Q, inverse=True)
        self._states = phase_norm(batch)

        # Recalculate entanglements only for q0 and q1.
        Sent_q0, Sent_q1 = calculate_q0_q1_entropy_from_rhos(rhos)
        q0_entanglement = np.where(self.preswaps_, Sent_q1, Sent_q0)
        q1_entanglement = np.where(self.preswaps_, Sent_q0, Sent_q1)
        self.entanglements[ax0, qubit_indices[:, 0]] = q0_entanglement
        self.entanglements[ax0, qubit_indices[:, 1]] = q1_entanglement
        self.entanglements = entropy(self._states)

        # Do postswaps
        _post_ent_relation = (self.entanglements[ax0, _qubit_indices[:, 0]] >= \
                              self.entanglements[ax0, _qubit_indices[:, 1]] + EPS)
        postswaps_ = []
        for n in range(N):
            if _post_ent_relation[n] ^ _ent_relation[n]:
                i, j = _qubit_indices[n]
                self._states[n] = np.swapaxes(self._states[n], i, j)
                self.entanglements[n,i], self.entanglements[n,j] = \
                self.entanglements[n,j], self.entanglements[n,i]
                postswaps_.append(True)
            else:
                postswaps_.append(False)
        self.postswaps_ = np.array(postswaps_)

    def reset_sub_environment_(self, k):
        x = self.state_generator(self.num_qubits, **self.state_generator_kwargs)
        self._states[k] = phase_norm(x)
        self.entanglements[k] = entropy(np.expand_dims(self._states[k], axis=0))


#----------------------------   State Generators   ----------------------------#

def sample_haar_full(num_qubits, **kwargs):
    """Draw sample Haar state from `num_qubits` dimensional Hilbert space."""
    if num_qubits < 1:
        raise ValueError("`num_qubits` must be > 1.")
    x = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
    x /= np.linalg.norm(x)
    return x.reshape((2,) * num_qubits)


def sample_haar_geom(num_qubits, p_gen=0.95, **kwargs):
    """Draw sample Haar state. The dimension of the Hilbert space is chosen
        using geometric distribution with parameter `p_gen`."""
    psi = random_quantum_state(num_qubits, p_gen)
    return np.transpose(psi, np.random.permutation(num_qubits))


def sample_haar_unif(num_qubits, min_entangled=1, **kwargs):
    """Draw sample Haar state. The dimension of the Hilbert space is chosen
       using uniform distribuiton in range [`min_entangled`, `num_qubits`]."""
    if num_qubits == 0:
        return np.array([1.])
    m = np.random.randint(min_entangled, num_qubits+1)
    psi = sample_haar_full(m).ravel()
    r = num_qubits - m
    while r > min_entangled:
        q = np.random.randint(min_entangled, min(m + 1, r + 1))
        psi = np.kron(psi, sample_haar_full(q).ravel())
        r -= q
    # Draw the last one
    if r > 0:
        psi = np.kron(psi, sample_haar_full(r).ravel())
    psi /= np.linalg.norm(psi.ravel())
    return psi.reshape((2,) * num_qubits)


def sample_mps(num_qubits, chi_max=None, **kwargs):
    if chi_max is None:
        chi_max = 2
    As, _, Lambdas = generate_random_MPS(num_qubits, 2, chi_max)
    psi = MPS_to_state(As, Lambdas, canonical=-1)
    return psi.reshape((2,) * num_qubits)



#------------------------------ Utility functions -----------------------------#
def random_quantum_state(q, prob=0.95):
    """Generate a quantum state as a Haar random state drawn from a subspace of
    the full Hilbert space.

    Args:
        q: int
            Number of qubits in the quantum state. Must be non-negative.
        prob: float, optional
            Probability for drawing the state from the full Hilbert space, i.e.
            all the qubits are entangled. (prob \in (0, 1]). Default 0.95.

    Returns:
        psi: np.Array
            Numpy array of shape (2, 2, ..., 2) representing the generated
            quantum state in the Hilbert space.
    """
    # Base case.
    if q == 0: return np.array([1.])
    if q == 1 or q == 2:
        psi = np.random.randn(2 ** q) + 1j * np.random.randn(2 ** q)
        psi = psi.reshape((2,) * q).astype(np.complex64)
        psi /= np.linalg.norm(psi, keepdims=True)
        return psi

    assert q > 2
    assert prob > 0 and prob <= 1.

    # Generate a geometric probability distribution for the subspace size.
    distr = np.array([prob * (1-prob) ** (q-i) for i in range(1, q+1)])
    distr /= distr.sum()

    # Generate Haar randoms state from a subspace of the full Hilbert space.
    # The final state is a product between the generated Haar states.
    k = np.random.choice(range(1, q+1), p=distr)
    A = np.random.randn(2 ** k) + 1j * np.random.randn(2 ** k)
    B = random_quantum_state(q-k, prob)
    psi = np.kron(A, B)

    # Reshape and normalize.
    psi = psi.reshape((2,) * q).astype(np.complex64)
    psi /= np.linalg.norm(psi, keepdims=True)
    return psi

def phase_norm(states):
    """Normalizes the relative phase shift between different qubits in one system."""
    B = states.shape[0]
    L = states.ndim - 1
    first = states.reshape(B, -1)[:, 0]
    phi = np.angle(first)
    z = np.cos(phi) - 1j * np.sin(phi)
    result = states * z.reshape((B,) + (1,) * L)
    # Set explicitly the imaginary part of first component to 0. This operation
    # is mandatory, because above multiplication can leave the imaginary part
    # nonzero, which breaks batch & solo rollout equivalence tests.
    for i in range(B):
        result[i].flat[0] = result[i].flat[0].real
    return result

def _ent_entropy(states, subsys_A):
    """Returns the entanglement entropy for every state in the batch w.r.t. `subsys_A`.

    Args:
        states (np.array): A numpy array of shape (b, 2,2,...,2), giving the states in the
            batch.
        subsys_A (list[int]): A list of ints specifying the indices of the qubits to be
            considered as a subsystem. The subsystem is the same for every state in the
            batch. If None, defaults to half of the system.

    Returns:
        entropies (np.Array): A numpy array of shape (b,), giving the entropy of each
            state in the batch.
    """
    L = states.ndim - 1
    subsys_B = [i for i in range(L) if i not in subsys_A]
    system = subsys_A + subsys_B
    subsys_A_size = len(subsys_A)
    subsys_B_size = L - subsys_A_size
    states = np.transpose(states, (0,) + tuple(t + 1 for t in system))
    states = states.reshape((-1, 2 ** subsys_A_size, 2 ** subsys_B_size))
    lmbda = np.linalg.svd(states, full_matrices=False, compute_uv=False)
    lmbda += np.finfo(lmbda.dtype).eps # shift lmbda to be positive within machine precision
    return -2.0 / subsys_A_size * np.einsum('ai, ai->a', lmbda ** 2, np.log(lmbda))

def entropy(states):
    """ For each state in the batch compute the entanglement entropies by
    considering each qubit as a subsystem.

    Args:
        states (np.array): A numpy array of shape (b, 2,2,...,2), giving the
        states in the batch.

    Returns:
        entropies (np.array): A numpy array of shape (b, L), giving single-qubit
        entropies.
    """
    L = states.ndim - 1
    entropies = [_ent_entropy(states, [i]) for i in range(L)]
    return np.stack(entropies).T

def calculate_q0_q1_entropy_from_rhos(rhos):
    a = rhos[:, 0] + rhos[:, 1] # rho_{q0-1}
    b = rhos[:, 2] + rhos[:, 3] # rho_{q0-2}
    c = rhos[:, 0] + rhos[:, 2] # rho_{q1-1}
    d = rhos[:, 1] + rhos[:, 3] # rho_{q1-2}
    Sent_q0 = -a*np.log(np.maximum(a, np.finfo(a.dtype).eps)) - \
               b*np.log(np.maximum(b, np.finfo(b.dtype).eps))
    Sent_q1 = -c*np.log(np.maximum(c, np.finfo(c.dtype).eps)) - \
               d*np.log(np.maximum(d, np.finfo(d.dtype).eps))
    return Sent_q0, Sent_q1

#------------------------------ System variables ------------------------------#
def _generate_permutation_maps(L):
    qubits_permutations = np.zeros((L, L, L), dtype=np.int32)
    qubits_inverse_permutations = np.zeros_like(qubits_permutations)
    for q0, q1 in permutations(range(L), 2):
        sysA = [q0, q1]
        sysB = [q for q in range(L) if q not in sysA]
        P = sysA + sysB
        qubits_permutations[q0, q1] = np.array(P, dtype=np.int32)
        qubits_inverse_permutations[q0, q1] = np.argsort(P).astype(np.int32)
    return qubits_permutations, qubits_inverse_permutations

_QUBITS = {}
_QSYSTEMS_P = {}
_QSYSTEMS_INV_P = {}
for L in range(2, 15):
    _QUBITS[L] = (2,)*L
    _QSYSTEMS_P[L], _QSYSTEMS_INV_P[L] = _generate_permutation_maps(L)

def permute_qubits(batch, qubits_indices, L, inverse=False, inplace=True):
    P = _QSYSTEMS_INV_P[L] if inverse else _QSYSTEMS_P[L]
    if inplace:
        for i, (q0, q1) in enumerate(qubits_indices):
            batch[i] = np.transpose(batch[i], P[q0, q1])
        return
    result = np.zeros_like(batch)
    for i, (q0, q1) in enumerate(qubits_indices):
        result[i] = np.transpose(batch[i], P[q0, q1])
    return result

#