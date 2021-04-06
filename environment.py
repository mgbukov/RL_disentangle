import numpy as np

np.random.seed(45)
from itertools import combinations
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from scipy.sparse.linalg import expm
from scipy.optimize import minimize


## 1. Initialize with random state
## 2. Step
## 3. Actions
## 4. Examine current state
## 5. Calculate entropy
# 6. Calculate reward
## 7. Take action
## 8. Reset with random state

class Environment:

    def __init__(self, N=2, entanglement_tol=1e-4):
        self.N = int(N)
        self.basis = spin_basis_general(N)
        self.entanglement_tol = entanglement_tol
        self.reset()
        self._bind_action_set()

    def reset(self):
        self.state = self._construct_random_pure_state()

    def _construct_random_pure_state(self):
        N = self.basis.Ns
        real = np.random.uniform(-1, 1, N).astype(np.float32)
        imag = np.random.uniform(-1, 1, N).astype(np.float32)
        s = real + 1j * imag
        norm = np.sum(s.conj() * s).real
        return s / np.sqrt(norm)

    def _construct_gate_generators(self):
        N = self.N
        basis = self.basis
        no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

        generators = []
        # Single quibit gate generators
        q = [[1.0, 0]]
        for i in range(N):
            q[0][1] = i
            H_xI = hamiltonian([['x', q]], [], basis=basis, **no_checks)
            H_yI = hamiltonian([['y', q]], [], basis=basis, **no_checks)
            H_zI = hamiltonian([['z', q]], [], basis=basis, **no_checks)
            p = str(i)
            generators.append((p + 'x', H_xI))
            generators.append((p + 'y', H_yI))
            generators.append((p + 'z', H_zI))

        # Two-qubit gate generators
        q = [[1.0, 0, 0]]
        for t in combinations(range(N), 2):
            q[0][1:] = t
            H_xI = hamiltonian([['xx', q]], [], basis=basis, **no_checks)
            H_yI = hamiltonian([['yy', q]], [], basis=basis, **no_checks)
            H_zI = hamiltonian([['zz', q]], [], basis=basis, **no_checks)
            # TODO This will break if N > 9
            p = '%d%d' % t
            generators.append((p + 'xx', H_xI))
            generators.append((p + 'yy', H_yI))
            generators.append((p + 'zz', H_zI))

        # What about more than two-gate generators?
        #
        return generators

    def _make_unitary_gate(self, G):
        """Create and return unit gate from generator ``G``"""
        optstate = self._compute_optimal_angle(G)
        angle = optstate.x
        return expm(-1j * angle * G)

    def entropy(self):
        return self.basis.ent_entropy(self.state)['Sent_A']

    @staticmethod
    def _compute_state_entropy_after_gate(angle, H, state):
        """Compute and return ``state``'s S_entropy"""
        U = expm(-1j * angle * H.toarray())
        psi = U.dot(state)
        return self.basis.ent_entropy(psi)['Sent_A']

    def _compute_optimal_angle(self, G):
        """Compute the optimal angle for unitary gate with generator ``G``"""
        angle = np.pi
        state = self.state.copy()
        return minimize(
            Environment._compute_state_entropy_after_gate,
            angle,
            args=(G, state),
            method='Nelder-Mead',
            tol=1e-5
        )

    def _bind_action_set(self):
        G = self._construct_gate_generators()
        id_to_name = {}
        generators = []
        for i, (name, H) in enumerate(G):
            id_to_name[i] = name
            generators.append(H)
        self.actions = id_to_name
        self.generators = np.array(generators)

    def step_(self, action):
        H = self.generators[action]
        U = self._make_unitary_gate(H)
        return U.dot(self.state)

    def step(self, action):
        """Modifyes ``self.state`` and returns (newstate, reward, done) tuple"""
        s = self.step_(action)
        self.state = s
        done = Environment._is_terminal(s)
        reward = 1 if done, else 0
        return s, reward, done

    @staticmethod
    def _is_terminal(state):
        ent = self.basis.ent_entropy(self.state)
        return np.isclose(0.0, ent, atol=self.entanglement_tol)


