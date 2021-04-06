import numpy as np

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
## 6. Calculate reward
## 7. Take action
## 8. Reset with random state

class Environment:

    def __init__(self, N=2, entanglement_tol=1e-4):
        self.N = int(N)
        self.basis = spin_basis_general(int(N))
        self.entanglement_tol = entanglement_tol
        self.reset()
        self._bind_action_set()

    def seed(self, seed):
        """Set the NumPy random seed."""
        np.random.seed(seed)

    def reset(self):
        self.state = self._construct_random_pure_state()

    def entropy(self):
        ent = self.basis.ent_entropy(self.state)['Sent_A']
        return ent.astype(np.float32).reshape(1)[0]

    def step_(self, action):
        H = self.operators[action]
        U = self._make_unitary_gate(H)
        return U.dot(self.state)

    def step(self, action):
        """Modifyes ``self.state`` and returns (newstate, reward, done) tuple"""
        s = self.step_(action)
        self.state = s
        done = self._is_terminal(s)
        reward = 1 if done else 0
        return s, reward, done

    def terminal(self):
        return self._is_terminal(self.state)

    def unitary_gate_from_operator(self, i, angle=np.pi/2):
        op = self.operators[i]
        return expm(-1j * angle * op)

    def _construct_random_pure_state(self):
        N = self.basis.Ns
        real = np.random.uniform(-1, 1, N).astype(np.float32)
        imag = np.random.uniform(-1, 1, N).astype(np.float32)
        s = real + 1j * imag
        norm = np.sqrt(np.sum(s.conj() * s).real)
        return s / norm

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
            generators.append((p + 'x', H_xI.toarray()))
            generators.append((p + 'y', H_yI.toarray()))
            generators.append((p + 'z', H_zI.toarray()))

        if self.N > 1:
            # Two-qubit gate generators
            q = [[1.0, 0, 0]]
            for t in combinations(range(N), 2):
                q[0][1:] = t
                H_xI = hamiltonian([['xx', q]], [], basis=basis, **no_checks)
                H_yI = hamiltonian([['yy', q]], [], basis=basis, **no_checks)
                H_zI = hamiltonian([['zz', q]], [], basis=basis, **no_checks)
                # TODO This will break if N > 9
                p = '%d%d' % t
                generators.append((p + 'xx', H_xI.toarray()))
                generators.append((p + 'yy', H_yI.toarray()))
                generators.append((p + 'zz', H_zI.toarray()))

        # What about more than two-gate generators?
        #
        return generators

    def _make_unitary_gate(self, G, angle=None):
        """Create and return unit gate from generator ``G``"""
        if angle is None:
            optstate = self._compute_optimal_angle(G)
            angle = optstate.x
        return expm(-1j * float(angle) * G)

    def _compute_entropy_after_gate(self, angle, H, state):
        """Compute and return ``state``'s S_entropy"""
        U = expm(-1j * angle * H)
        psi = U.dot(state)
        return self.basis.ent_entropy(psi)['Sent_A']

    def _compute_optimal_angle(self, G):
        """Compute the optimal angle for unitary gate with generator ``G``"""
        angle = np.pi
        state = self.state.copy()
        return minimize(
            self._compute_entropy_after_gate,
            angle,
            args=(G, state),
            method='Nelder-Mead',
            tol=1e-5
        )

    def _bind_action_set(self):
        G = self._construct_gate_generators()
        idx_to_name = {}
        name_to_idx = {}
        generators = []
        for i, (name, H) in enumerate(G):
            idx_to_name[i] = name
            name_to_idx[name] = i
            generators.append(H)
        self.operator_to_idx = name_to_idx
        self.idx_to_operator = idx_to_name
        self.operators = np.array(generators)

    def _is_terminal(self, state):
        ent = self.basis.ent_entropy(state)['Sent_A']
        return np.isclose(0.0, ent, atol=self.entanglement_tol)
