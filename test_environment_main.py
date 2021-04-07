import numpy as np
import unittest
from itertools import combinations, chain, product
from quspin.operators import hamiltonian
from scipy.linalg import expm

from environment_main import QubitsEnvironment


class TestQubitsEnvironment(unittest.TestCase):

    def test_initialization(self):
        """Test default initialization and initialization with N qubits"""
        E = QubitsEnvironment()
        self.assertEqual(E.N, 2)
        self.assertEqual(len(E.state), 4)
        E = QubitsEnvironment(4)
        self.assertEqual(len(E.state), 16)
        E = QubitsEnvironment(10)
        self.assertEqual(len(E.state), 1024)

    def test_state(self):
        """Test if the state is a unit vector."""
        E = QubitsEnvironment()
        for _ in range(100):
            norm = np.sum(np.abs(E.state) ** 2)
            self.assertTrue(np.isclose(norm.imag, 0.0))
            self.assertTrue(np.isclose(norm.real, 1.0))
            E.reset()

    def test_action_space(self):
        envs = (QubitsEnvironment(), QubitsEnvironment(4), QubitsEnvironment(8))
        operators1 = [' x', ' y', ' z']
        operators2 = [' xx', ' yy', ' zz']
        for E in envs:
            actions = E.operator_to_idx.keys()
            ops1 = (str(i) + o for i in range(E.N) for o in operators1)
            ops2 = ('%d %d' % t + o
                        for t in combinations(range(E.N), 2)
                            for o in operators2)
            O = list(chain(ops1, ops2))
            for op in O:
                self.assertIn(op, actions)
            self.assertEqual(len(O), len(E.operator_to_idx))

    def test_operators(self):
        envs = [QubitsEnvironment(), QubitsEnvironment(4), QubitsEnvironment(8)]
        no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)
        for E in envs:
            for i in range(E.N):
                p = str(i)
                # Pauli X
                H = hamiltonian(
                    static_list=[['x', [[1.0, i]]]],
                    dynamic_list=[],
                    basis=E.basis,
                    **no_checks).toarray()
                idx = E.operator_to_idx[p + ' x']
                self.assertTrue(np.all(H == E.operators[idx]))
                # Pauli Y
                H = hamiltonian(
                    static_list=[['y', [[1.0, i]]]],
                    dynamic_list=[],
                    basis=E.basis,
                    **no_checks).toarray()
                idx = E.operator_to_idx[p + ' y']
                self.assertTrue(np.all(H == E.operators[idx]))
                # Pauli Z
                H = hamiltonian(
                    static_list=[['z', [[1.0, i]]]],
                    dynamic_list=[],
                    basis=E.basis,
                    **no_checks).toarray()
                idx = E.operator_to_idx[p + ' z']
                self.assertTrue(np.all(H == E.operators[idx]))

            for i, j in combinations(range(E.N), 2):
                n = '%d %d' % (i, j)
                # Pauly XX
                H = hamiltonian(
                    static_list=[['xx', [[1.0, i, j]]]],
                    dynamic_list=[],
                    basis=E.basis,
                    **no_checks).toarray()
                idx = E.operator_to_idx[n + ' xx']
                self.assertTrue(np.all(H == E.operators[idx]))
                # Pauly YY
                H = hamiltonian(
                    static_list=[['yy', [[1.0, i, j]]]],
                    dynamic_list=[],
                    basis=E.basis,
                    **no_checks).toarray()
                idx = E.operator_to_idx[n + ' yy']
                self.assertTrue(np.all(H == E.operators[idx]))
                # Pauly ZZ
                H = hamiltonian(
                    static_list=[['zz', [[1.0, i, j]]]],
                    dynamic_list=[],
                    basis=E.basis,
                    **no_checks).toarray()
                idx = E.operator_to_idx[n + ' zz']
                self.assertTrue(np.all(H == E.operators[idx]))

    def test_entropy(self):
        """Test the entanglement entropy calculation."""
        E = QubitsEnvironment(2)
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

    def test_disentangled(self):
        E = QubitsEnvironment()
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

        E.state = phi_plus
        self.assertFalse(E.disentangled())
        E.state = phi_minus
        self.assertFalse(E.disentangled())
        E.state = psi_plus
        self.assertFalse(E.disentangled())
        E.state = psi_minus
        self.assertFalse(E.disentangled())

        states = [
            np.array([1.0, 0.0]),
            np.array([C, C]),
            np.array([C, C * 1j]),
            np.array([np.sqrt(3/4), 0.5])
        ]
        for sA, sB in product(states, states):
            E.state = np.kron(sA, sB)
            self.assertTrue(E.disentangled())
            E.state = np.kron(sB, sA)
            self.assertTrue(E.disentangled())
            E.state = np.kron(sA[::-1], sB)
            self.assertTrue(E.disentangled())
            E.state = np.kron(sA, sB[::-1])
            self.assertTrue(E.disentangled())
            E.state = np.kron(sA[::-1], sB[::-1])
            self.assertTrue(E.disentangled())

        stateA = np.array([1.0, 0.0])
        stateB = np.array([1.0, 0.0])
        stateAB = np.kron(stateA, stateB)
        E.state = stateAB
        self.assertTrue(E.disentangled())

    def test_unitary_gates(self):
        envs = [QubitsEnvironment(), QubitsEnvironment(4)]
        angles = 2 * np.pi / np.arange(1, 7)
        single = ('x', 'y', 'z')
        two = ('xx', 'yy', 'zz')
        no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)
        for E in envs:
            for i in range(E.N):
                for op, angle in product(single, angles):
                    H = hamiltonian(
                        static_list=[[op, [[1.0, i]]]],
                        dynamic_list=[],
                        basis=E.basis,
                        **no_checks).toarray()
                    U = expm(-1j * angle * H)
                    idx = E.operator_to_idx['%d %s' % (i, op)]
                    eq = np.isclose(U, E._unitary_gate_factory(idx)(angle))
                    self.assertTrue(np.all(eq))

            for i, j in combinations(range(E.N), 2):
                for op, angle in product(two, angles):
                    H = hamiltonian(
                        static_list=[[op, [[1.0, i, j]]]],
                        dynamic_list=[],
                        basis=E.basis,
                        **no_checks).toarray()
                    U = expm(-1j * angle * H)
                    idx = E.operator_to_idx['%d %d %s' % (i, j, op)]
                    eq = np.isclose(U, E._unitary_gate_factory(idx)(angle))
                    self.assertTrue(np.all(eq))

    def test_reward(self):
        pass



if __name__ == '__main__':
    unittest.main()
