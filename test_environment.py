import numpy as np
import unittest
from itertools import combinations, chain
from quspin.operators import hamiltonian

from environment import Environment


class TestEnvironment(unittest.TestCase):

    def test_initialization(self):
        """Test default initialization and initialization with N qubits"""
        E = Environment()
        self.assertEqual(E.N, 2)
        self.assertEqual(len(E.state), 4)
        E = Environment(4)
        self.assertEqual(len(E.state), 16)
        E = Environment(10)
        self.assertEqual(len(E.state), 1024)

    def test_state(self):
        """Test if the state is a unit vector."""
        E = Environment()
        for _ in range(100):
            norm = np.sum(np.abs(E.state) ** 2)
            self.assertTrue(np.isclose(norm.imag, 0.0))
            self.assertTrue(np.isclose(norm.real, 1.0))
            E.reset()

    def test_action_set(self):
        """Test the construction of actions."""
        envs = (Environment(), Environment(4), Environment(8))
        operators1 = 'x y z'.split()
        operators2 = 'xx yy zz'.split()
        for E in envs:
            actions = E.action_to_idx.keys()
            ops1 = (str(i) + o for i in range(E.N) for o in operators1)
            ops2 = ('%d%d' % t + o
                        for t in combinations(range(E.N), 2)
                            for o in operators2)
            O = list(chain(ops1, ops2))
            for op in O:
                self.assertIn(op, actions)
            self.assertEqual(len(O), len(E.action_to_idx))

    def test_gate_generators(self):
        """Test the gate generators."""
        envs = [Environment(), Environment(4), Environment(8)]
        for E in envs:
            for i in range(E.N):
                p = str(i)
                H = hamiltonian([['x', [[1.0, i]]]], [], basis=E.basis).toarray()
                idx = E.action_to_idx[p + 'x']
                self.assertTrue(np.all(H == E.generators[idx]))
                H = hamiltonian([['y', [[1.0, i]]]], [], basis=E.basis).toarray()
                idx = E.action_to_idx[p + 'y']
                self.assertTrue(np.all(H == E.generators[idx]))
                H = hamiltonian([['z', [[1.0, i]]]], [], basis=E.basis).toarray()
                idx = E.action_to_idx[p + 'z']
                self.assertTrue(np.all(H == E.generators[idx]))

            for i, j in combinations(range(E.N), 2):
                n = '%d%d' % (i, j)
                H = hamiltonian([['xx', [[1.0, i, j]]]], [], basis=E.basis).toarray()
                idx = E.action_to_idx[n + 'xx']
                self.assertTrue(np.all(H == E.generators[idx]))
                H = hamiltonian([['yy', [[1.0, i, j]]]], [], basis=E.basis).toarray()
                idx = E.action_to_idx[n + 'yy']
                self.assertTrue(np.all(H == E.generators[idx]))
                H = hamiltonian([['zz', [[1.0, i, j]]]], [], basis=E.basis).toarray()
                idx = E.action_to_idx[n + 'zz']
                self.assertTrue(np.all(H == E.generators[idx]))

    def test_entropy(self):
        """Test the entanglement entropy calculation."""
        E = Environment(2)
        C = 1 / np.sqrt(2)

        # Bell states
        phi_plus = np.zeros(4)
        phi_plus[E.basis.state_to_int('00')] = C
        phi_plus[E.basis.state_to_int('11')] = C

        phi_minus = np.zeros(4)
        phi_minus[E.basis.state_to_int('00')] = C
        phi_minus[E.basis.state_to_int('11')] = -C

        psi_plus = np.zeros(4)
        psi_plus[E.basis.state_to_int('01')] = C
        psi_plus[E.basis.state_to_int('10')] = C

        psi_minus = np.zeros(4)
        psi_minus[E.basis.state_to_int('01')] = C
        psi_minus[E.basis.state_to_int('10')] = -C

        expected = np.log(2)
        self.assertIsInstance(E.entropy(), np.float32)
        E.state = phi_plus
        self.assertTrue(np.isclose(E.entropy(), expected))
        E.state = phi_minus
        self.assertTrue(np.isclose(E.entropy(), expected))
        E.state = psi_plus
        self.assertTrue(np.isclose(E.entropy(), expected))
        E.state = psi_minus
        self.assertTrue(np.isclose(E.entropy(), expected))

if __name__ == '__main__':
    unittest.main()