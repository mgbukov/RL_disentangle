import numpy as np
import matplotlib.pyplot as plt
import torch
import unittest
from itertools import product
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian

import src.envs.noquspin_env as env
import src.envs.batched_env as qenv

# from src.envs.noquspin_env import ent_entropy, QubitsEnvironment


class TestNoQuspin(unittest.TestCase):

    def test_initialization(self):
        """Test default initialization and initialization with N qubits"""
        env.set_backend('numpy')
        E = env.QubitsEnvironment()
        self.assertEqual(E.L, 2)
        self.assertEqual(E.Ns, 4)
        self.assertEqual(len(E.actions), 9)
        self.assertEqual(E._state.shape, (1, E.Ns))
        self.assertTrue(E._state.dtype == np.complex64)
        E = env.QubitsEnvironment(4, batch_size=32)
        self.assertEqual(E.state.shape, (32, 2 * E.Ns))
        self.assertTrue(E.state.dtype == np.float32)

        env.set_backend('torch')
        E = env.QubitsEnvironment()
        self.assertEqual(E.L, 2)
        self.assertEqual(E.Ns, 4)
        self.assertEqual(len(E.actions), 9)
        self.assertEqual(E._state.shape, (1, E.Ns))
        self.assertTrue(E._state.dtype == torch.complex64)
        E = env.QubitsEnvironment(4, batch_size=32)
        self.assertEqual(E.state.shape, (32, 2 * E.Ns))
        self.assertTrue(E.state.dtype == torch.float32)

        for L in (2, 4, 5, 6, 7):
            E = env.QubitsEnvironment(L)
            self.assertEqual(len(E.actions), 9 * L * (L - 1) // 2)

    def test_state(self):
        """Test if the state is a unit vector."""

        # Assert that the norm is a unit vector
        env.set_backend('numpy')
        E = env.QubitsEnvironment(batch_size=1)
        for _ in range(100):
            norm = np.linalg.norm(E._state, axis=1)
            self.assertTrue(np.all(np.isclose(norm, 1.0)))
            E.set_random_state(copy=False)

        env.set_backend('torch')
        E = env.QubitsEnvironment(batch_size=1)
        for _ in range(100):
            norm = np.linalg.norm(E._state, axis=1)
            self.assertTrue(np.all(np.isclose(norm, 1.0)))
            E.set_random_state()

        # Assert that we can manually set the state
        env.set_backend('numpy')
        E = env.QubitsEnvironment(4, batch_size=64)
        np.random.seed(8)
        # Try with non-unit vectors
        s = np.random.rand(64, 16) + 1j * np.random.rand(64, 16)
        self.assertRaises(ValueError, E.set_state, s)
        # Try with different shape
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        s = s.astype(np.complex64)
        self.assertRaises(ValueError, E.set_state, s[:10])
        # Try with everything correct
        E.set_state(s)
        self.assertTrue(np.all(E._state == s))
        E.reset()
        E.set_state(torch.from_numpy(s))
        self.assertTrue(np.all(E._state == s))

    def test_entropy(self):
        """Test the entanglement entropy calculation."""
        env.set_backend('numpy')
        E = env.QubitsEnvironment(2, batch_size=1)
        C = 1 / np.sqrt(2)

        # Bell states
        phi_plus = np.zeros(4)
        phi_plus[[0, 3]] = C

        phi_minus = np.zeros(4)
        phi_minus[0] = C
        phi_minus[3] = -C

        psi_plus = np.zeros(4)
        psi_plus[[1,2]] = C

        psi_minus = np.zeros(4)
        psi_minus[0] = C
        psi_minus[3] = -C

        expected = np.log(2)
        E.set_state(phi_plus[np.newaxis, :])
        self.assertTrue(np.isclose(E.entropy()[0], expected))
        E.set_state(phi_minus[np.newaxis, :])
        self.assertTrue(np.isclose(E.entropy()[0], expected))
        E.set_state(psi_plus[np.newaxis, :])
        self.assertTrue(np.isclose(E.entropy()[0], expected))
        E.set_state(psi_minus[np.newaxis, :])
        self.assertTrue(np.isclose(E.entropy()[0], expected))

    def test_disentangled(self):
        env.set_backend('torch')
        E = env.QubitsEnvironment(2)
        C = 1 / np.sqrt(2)

        # Bell states
        phi_plus = np.zeros(4)
        phi_plus[[0, 3]] = C

        phi_minus = np.zeros(4)
        phi_minus[0] = C
        phi_minus[3] = -C

        psi_plus = np.zeros(4)
        psi_plus[[1,2]] = C

        psi_minus = np.zeros(4)
        psi_minus[0] = C
        psi_minus[3] = -C

        E.set_state(phi_plus[np.newaxis, :])
        self.assertFalse(E.disentangled())
        E.set_state(phi_minus[np.newaxis, :])
        self.assertFalse(E.disentangled())
        E.set_state(psi_plus[np.newaxis, :])
        self.assertFalse(E.disentangled())
        E.set_state(psi_minus[np.newaxis, :])
        self.assertFalse(E.disentangled())

        stateA = np.array([1.0, 0.0])
        stateB = np.array([1.0, 0.0])
        stateAB = np.kron(stateA, stateB)
        E.set_state(stateAB[np.newaxis, :])
        self.assertTrue(E.disentangled())


    def test_state_equivalence_fixed_angles(self):
        """ Asserts that the results of the No Quspin Environment are
        the same as the Quspin one. """
        env.set_backend('numpy')
        # Test No Quspin state transitions against the Quspin Envirnment for
        # different batch sizes and number of qubits
        batches = (1, 4, 32)
        qubits = (2, 4, 5)
        N = 100
        for b, L in product(batches, qubits):
            # print(f'Batch size: {b}\t\tL: {L}')
            Q = qenv.QubitsEnvironment(L, epsi=1e-3, batch_size=b)
            E = env.QubitsEnvironment(L, epsi=1e-3, batch_size=b)
            self.assertTrue(np.all(np.isclose(Q.state, E.state)))

            np.random.seed(46)  # 44, 45, 46
            actions = np.random.uniform(0, E.num_actions, (N, b)).astype(np.int)
            for a in actions:
                Q.step(a, angle=0.546)
                E.step(a, angle=0.546)
                self.assertTrue(np.all(np.isclose(Q.state, E.state, atol=1e-5)))

    def test_state_equivalence(self):
        """ Asserts that the results of the No Quspin Environment are
        the same as the Quspin one. """
        env.set_backend('numpy')
        # Test No Quspin state transitions against the Quspin Envirnment for
        # different batch sizes and number of qubits
        batches = (1, 4, 32)
        qubits = (2, 4, 5)
        N = 100
        for b, L in product(batches, qubits):
            print(f'Batch size: {b}\t\tL: {L}')
            Q = qenv.QubitsEnvironment(L, epsi=1e-3, batch_size=b)
            E = env.QubitsEnvironment(L, epsi=1e-3, batch_size=b)
            np.random.seed(44)  # 44, 45, 46
            Q.set_random_state()
            E.set_state(Q._state.squeeze(-1))

            self.assertTrue(np.all(np.isclose(Q.state, E.state)))

            actions = np.random.uniform(0, E.num_actions, (N, b)).astype(np.int)
            for a in actions:
                Q.step(a)
                E.step(a)
                print(Q.state)
                print(E.state)
                self.assertTrue(np.all(np.isclose(Q.state, E.state, atol=1e-5)))

    def test_state_norm(self):
        env.set_backend('numpy')
        N = 100
        b = 1
        E = env.QubitsEnvironment(4, batch_size=b)
        actions = np.random.uniform(0, E.num_actions, (N, b)).astype(np.int)
        for a in actions:
            E.step(a)
            norm = np.linalg.norm(E._state, axis=1)
            self.assertTrue(np.all(np.isclose(norm, 1.0, atol=1e-6)))


    # def test_nunmpy_torch_gates(self):
    #     env.set_backend('numpy')
    #     E = env.QubitsEnvironment(4, epsi=1e-3)
    #     N = 100
    #     actions = np.random.uniform(0, E.num_actions, (N, 1)).astype(np.int)

    #     numpy_gates


    def test_quspin_numpy_divergence(self):
        """ Asserts that the results of the No Quspin Environment are
        the same as the Quspin one. """
        env.set_backend('numpy')
        Q = qenv.QubitsEnvironment(4, epsi=1e-3)
        E = env.QubitsEnvironment(4, epsi=1e-3)
        self.assertTrue(np.all(np.isclose(Q.state, E.state)))

        np.random.seed(44)
        N = 100
        divergence = []
        actions = np.random.uniform(0, E.num_actions, (N, 1)).astype(np.int)
        for i in range(N):
            a = actions[i]
            Q.step(a, angle=0.546)
            E.step(a, angle=0.546)
            divergence.append(np.linalg.norm(E.state.ravel() - Q.state.ravel()))
            self.assertTrue(np.all(np.isclose(Q.state, E.state, atol=1e-5)))
            self.assertTrue(np.all(np.isclose(np.linalg.norm(E.state.ravel()), 1.0, atol=1e-5)))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('State divergence between QuSpin and Numpy\n'
                     'for 100 random actions, 4 qubits')
        ax.set_xlabel('N')
        ax.set_ylabel('L2 norm')
        ax.plot(divergence)
        plt.savefig('quspin_numpy_divergence.png', dpi=200)
        plt.close(fig)

    def test_quspin_torch_divergence(self):
        """ Asserts that the results of the No Quspin Environment are
        the same as the Quspin one. """
        env.set_backend('torch')
        Q = qenv.QubitsEnvironment(4, epsi=1e-3)
        E = env.QubitsEnvironment(4, epsi=1e-3)
        self.assertTrue(np.all(np.isclose(Q.state, E.state)))

        np.random.seed(44)
        N = 100
        state_divergence = []
        norm_divergence = []
        actions = np.random.uniform(0, E.num_actions, (N, 1)).astype(np.int)
        for i in range(N):
            a = actions[i]
            Q.step(a, angle=0.546)
            E.step(a, angle=0.546)
            # print(Q.state)
            # print(E.state)
            state_diff = np.linalg.norm(E.state.numpy().ravel() - Q.state.ravel())
            state_divergence.append(state_diff)
            self.assertTrue(np.all(np.isclose(Q.state, E.state, atol=1e-3)))
            norm = np.linalg.norm(E.state.numpy().ravel())
            print(norm, 1.0 - norm)
            norm_divergence.append(1.0 - norm)
            self.assertTrue(np.all(np.isclose(norm, 1.0, atol=1e-2)))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('State & norm divergence between QuSpin and PyTorch\n'
                     'for 100 random actions, 4 qubits')
        ax.set_xlabel('N')
        ax.set_ylabel('L2 norm')
        ax.plot(state_divergence, label='state')
        ax.plot(norm_divergence, label='norm')
        ax.legend()
        plt.savefig('quspin_torch_divergence_temp2.png', dpi=200)
        plt.close(fig)

    def test_fast_apply_gates_2qubits(self):

        from quspin.basis import spin_basis_1d
        from quspin.operators import hamiltonian
        from scipy.linalg import expm

        basis = spin_basis_1d(2)
        no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

        xx = hamiltonian([['xx', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        xy = hamiltonian([['xy', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        xz = hamiltonian([['xz', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        yx = hamiltonian([['yx', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        yy = hamiltonian([['yy', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        yz = hamiltonian([['yz', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        zx = hamiltonian([['zx', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        zy = hamiltonian([['zy', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()
        zz = hamiltonian([['zz', [[1.0, 0, 1]]]], [], basis=basis, **no_checks).toarray()

        state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex64)

        # print(expm(xx))
        # print(np.kron(expm(sigma_x), expm(sigma_x)))

        psi = xx @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_x, sigma_x), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = xy @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_x, sigma_y), 0, 1)
        print(psi)
        print(phi)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = xz @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_x, sigma_z), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = yx @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_y, sigma_x), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = yy @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_y, sigma_y), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = yz @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_y, sigma_z), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = zx @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_z, sigma_x), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = zy @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_z, sigma_y), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        psi = zz @ state
        phi = env.apply_gate_fast(state, np.kron(sigma_z, sigma_z), 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))

        # Test Unitary gates
        psi = expm(1j * 0.4 * xx) @ state
        gate = expm(1j * 0.4 * np.kron(sigma_x, sigma_x))

        phi = env.apply_gate_fast(state, gate, 0, 1)
        self.assertTrue(np.all(np.isclose(psi, phi)))


    def test_backends_actions(self):
        L = 4
        env.set_backend('numpy')
        E = env.QubitsEnvironment(L)
        num_actions = E.num_actions
        numpy_final = {}
        for a in range(num_actions):
            E.reset()
            for _ in range(100):
                E.step([a], angle=0.563)
            numpy_final[a] = E.state

        env.set_backend('torch')
        E = env.QubitsEnvironment(L)
        torch_final = {}
        for a in range(num_actions):
            E.reset()
            for _ in range(100):
                E.step([a], angle=0.563)
            torch_final[a] = E.state

        difference = np.zeros(num_actions, dtype=np.float32)
        for a in numpy_final.keys():
            diff = numpy_final[a] - torch_final[a].numpy()
            difference[a] = np.linalg.norm(diff)
        print(difference)

        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_title('State divergence between Numpy and PyTorch\n'
                     'after applying same action 100 times,\n'
                     f'{L} qubits')
        ax.set_xlabel('action')
        ax.set_ylabel('L2 norm of the difference')
        ax.plot(np.arange(num_actions), difference, marker='o', ls='--')
        plt.savefig(f'numpy_torch_action_divergence_{L}qubits.png', dpi=160)


    def test_nans_after_optimizer(self):
        L = 4
        env.set_backend('torch')
        E = env.QubitsEnvironment(L)
        for a in range(E.num_actions):
            E.set_random_state()
            for i in range(10):
                E.step([a])
                if np.any(E.state.numpy() != E.state.numpy()):
                    print('=' * 80)
                    print(f'NaNs at step {i}')
                    print(f'Action: {a}')
                    self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
