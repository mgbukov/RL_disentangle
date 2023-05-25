import numpy as np
import sys
from typing import Literal

sys.path.append('..')
from src.envs.rdm_environment import QubitsEnvironment
from qiskit.helpers import *

np.set_printoptions(precision=6, suppress=True)
np.random.seed(44)

def observe_rdms(env):
    states = env.states
    rdms = []
    qubit_pairs = list((i,j) for i,j in env.actions.values() if i < j)
    for qubits in qubit_pairs:
        sysA = tuple(q+1 for q in qubits)
        sysB = tuple(q+1 for q in range(env.L) if q not in qubits)
        permutation = (0,) + sysA + sysB
        states = np.transpose(states, permutation)
        psi = states.reshape(env.batch_size, 4, -1)
        rdm = psi @ np.transpose(psi, (0, 2, 1)).conj()
        rdms.append(rdm)
        # Reset permutation
        states = np.transpose(states, np.argsort(permutation))
        assert np.all(states == env.states)
    rdms = np.array(rdms)                   # rdms.shape == (Q, B, 4, 4)
    rdms = rdms.transpose((1, 0, 2, 3))     # rdms.shape == (B, Q, 4, 4)
    return rdms[0]


def test_get_action_4q(policy: Literal['universal', 'equivariant', 'transformer']):
    env = QubitsEnvironment(4, epsi=1e-3)

    for _ in range(100):
        env.set_random_states()
        assert not env.disentangled()[0]

        nsteps = 5 if policy == 'universal' else 8
        for _ in range(nsteps):
            rdms = observe_rdms(env)
            U, i, j = get_action_4q(rdms, policy)
            a = env.actToKey[(i,j)]
            env.step([a])
            assert U.dtype == np.complex64
            assert U.shape == (4,4)
            I = U @ U.T.conj()
            assert np.all(np.isclose(I, np.eye(4, dtype=np.complex64), atol=1e-6))
            assert np.all(np.isclose(env.unitary[0], U))
            if env.disentangled()[0]:
                break
        if not env.disentangled()[0]:
            print(f'[INFO]: Failed to disentangle state in {nsteps}!'
                   '\n\tFinal entropies: ', env.entropy()[0]
            )
        # assert env.disentangled()[0]


def fidelity(psi, phi):
    psi = psi.ravel()
    phi = phi.ravel()
    return np.abs(np.dot(psi.conj(), phi)) ** 2


def test_peek_next_4q():
    env = QubitsEnvironment(4, epsi=1e-3)

    for _ in range(100):
        env.set_random_states()
        assert not env.disentangled()[0]

        for _ in range(10):
            psi = env.states[0]
            a = int(np.random.uniform(0, env.num_actions - 1))
            phi, r, done = env.step(a)
            U = env.unitary[0]
            i, j = env.actions[a]
            phi2, ent, r2 = peek_next_4q(psi, U, i, j)
            assert np.isclose(fidelity(phi, phi2.ravel()), 1.0)
            assert r == r2
            assert np.all(np.isclose(env.entropy()[0], ent, atol=1e-5))


if __name__ == '__main__':
    print('Testing Universal policy')
    test_get_action_4q('universal')
    print('Testing PE policy')
    test_get_action_4q('equivariant')
    print('Testing Transformer policy')
    test_get_action_4q('transformer')
    test_peek_next_4q()
    print('ok')
