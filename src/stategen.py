"""Quantum State Generators"""
import numpy as np

from .mpslib import MPS_to_state, generate_random_MPS, state_to_MPS


# Choice of state distribution from which states are drawn on
# reset() call. Currently supported:
#     haar_full:  States are drawn from `num_qubits` dimensional
#                 Hilbert space.
#     haar_geom:  States are drawn from 2 to `num_qubits`
#                 dimensional Hilbert space. The dimension is
#                 chosen from geometric distribution.
#     haar_unif:  States are drawn from 2 to `num_qubits`
#                 dimensional Hilbert space and each dimension
#                 has equal probability of being sampled.
#     mps:        Matrix product states, parametarized by maximum
#                 bond dimension `chi_max`.

class StateGenerator:

    def __init__(self, sample_fn, num_qubits, sample_params={}):
        self.sample_fn = sample_fn
        self.num_qubits = num_qubits
        self.sample_params = sample_params

    def __call__(self):
        return self.sample_fn(self.num_qubits, **self.sample_params)

    def update(self, **new_sample_params):
        self.sample_params.update(new_sample_params)


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


def sample_haar_unif(num_qubits, min_entangled=1, max_entangled=None, **kwargs):
    """Draw sample Haar state. The dimension of the Hilbert space is chosen
       using uniform distribuiton in range [`min_entangled`, `max_entangled`]."""
    if num_qubits == 0:
        return np.array([1.])
    if max_entangled is None:
        max_entangled = num_qubits
    m = np.random.randint(min_entangled, max_entangled+1)
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


def sample_haar_product(num_qubits, min_subsystem_size, max_subsystem_size):
    subsystems = sample_subsystem_sizes(num_qubits, min_subsystem_size, max_subsystem_size)

    psi = sample_haar_full(subsystems[0])
    for L in subsystems[1:]:
        psi = np.kron(psi, sample_haar_full(L).ravel())

    psi /= np.linalg.norm(psi.ravel())
    return psi.reshape((2,) * num_qubits)


def sample_mps(num_qubits, chi_max=None, **kwargs):
    if chi_max is None:
        chi_max = 2
    As, _, Lambdas = generate_random_MPS(num_qubits, 2, chi_max)
    psi = MPS_to_state(As, Lambdas, canonical=-1)
    return psi.reshape((2,) * num_qubits)


def sample_subsystem_sizes(num_qubits, min_size, max_size):
    """
    Returns a tuple `(L_0, L_1, ...., L_k)`, where each `L_i` is the size of a
    quantum subsystem and `min_size <= L_i <= max_size`.
    The elements `L_i` in the returned tuple sum to `num_qubits`.
    Each `L_i` is sampled from uniform distribution `U[min_size, max_size]`.
    """
    result = []
    n = 0
    while n < num_qubits:
        x = np.random.randint(min_size, max_size+1)
        # Clip to `num_qubits`
        x = np.clip(x, 0, num_qubits - n)
        result.append(x)
        n += x
    return tuple(result)


def sample_generalized_haar(num_qubits: int, min_subsystem_size: int,
                            max_subsystem_size: int, eta: float):
    """
    Generates Haar random state, in which entanglement is concentrated in
    subsystems with size between `[min_system_size, max_system_size]`.
    Entanglement between subsystems is controlled via the `eta` parameter.
    This function allows the user to "blend" between product states and fully
    entangled states.

    Args:
        num_qubits: int
            Number of qubits
        min_subsystem_size: int
            Minimum number of qubits in subsystem
        max_subsystem_size: int
            Maximum number of qubits in subsystem
        eta: float
            Controls entanglement: `eta` > 1.0E2 corresponds to no entanglement
            across the bonds, `eta` < 1.0E-2 to roughly maximum entanglement.

    Returns:
        psi: np.ndarray
            Numpy array with shape (2, 2, ..., 2) representing the generated state
    """

    # Sample subsystem sizes:
    #   Example: (2, 3, 5, 2)
    subsystem_sizes = sample_subsystem_sizes(num_qubits, min_subsystem_size, max_subsystem_size)

    # Initialize bond positions array. Bond positions include `0` and `num_qubits`
    #   Example: [0, 2, 5, 10, 12]
    bonds = np.cumsum((0,) + subsystem_sizes)

    # Initialize bond dimensions
    #   Example: [1, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2, 1]
    chivec = [2 ** min(j, num_qubits - j) for j in range(num_qubits)]

    # Initialize partitions
    chipar = chivec[bonds]

    # Create a Haar random state and decompose it as MPS
    psi = sample_haar_full(num_qubits)
    gammas, lambdas = state_to_MPS(psi, chivec, num_qubits)

    # Now modify entanglement structure only at partitions / cuts
    for b, chi in zip(bonds[1:-1], chipar[1:-1]):

        # Define new `$\lambda$` matrix. Since `$\lambda` is a diagonal matrix,
        # we generate only the diagonal vector from uniform distribution and
        # then exponentiate it component-wise. Lambda values define a probability
        # distribution, so they must sum to 1
        lambdaM = np.exp( -eta * np.sort(np.random.uniform(size=chi)) )
        lambdaM /= np.linalg.norm(lambdaM)

        # Update adjacent `$\gamma$` matrices with the following equation:
        #
        #   `$\gamma_{new} = \gamma * \lambda / \lambda_{new}$`
        #
        # `atol` and `rtol` are constant values added to avoid division of 0
        atol = 0.0
        rtol = np.finfo(lambdaM.dtype).eps
        max_lambda = np.max(lambdaM, initial=0.)
        val = atol + max_lambda * rtol

        divres = np.zeros_like(lambdaM)
        #                                          where=(np.abs(lambdaM) > tol) & (np.abs(lambdas[b]) > tol)
        np.divide(lambdas[b], lambdaM, out=divres, where=np.abs(lambdas[b]/lambdaM) >= val)
        gammas[b] = np.einsum('a,aib->aib', divres, gammas[b] )

        # Replace corresponding Lambda matrix
        lambdas[b] = lambdaM

    state = MPS_to_state(gammas, lambdas, canonical=0)
    return state.reshape((2,) * num_qubits)


