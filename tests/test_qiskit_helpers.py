import itertools
import numpy as np
import sys
from typing import Literal

sys.path.append('..')
from src.quantum_env import QuantumEnv, rdm_2q_half
from src.quantum_state import phase_norm
from qiskit.helpers import *

np.random.seed(44)
np.set_printoptions(precision=6, suppress=True)


def observe_rdms(state):
    return rdm_2q_half(state.reshape(1,2,2,2,2)).reshape(6, 4, 4)


def fidelity(psi, phi):
    psi = psi.ravel()
    phi = phi.ravel()
    return np.abs(np.dot(psi.conj(), phi)) ** 2


def test_get_preswap_gate(n_tests=100):
    """
    Test if the returned pre-swap gate in qiskit code agrees with the
    `_preswap` flag in the RL environment.
    """
    env = QuantumEnv(4, 1, obs_fn="phase_norm")
    P = np.eye(4,4,dtype=np.complex64)
    P[[2,1]] = P[[1,2]]
    result = True
    failed = 0
    for _ in range(n_tests):
        env.reset()
        env.simulator.set_random_states_()
        # Rollout a trajectory
        for _ in range(6):
            psi = env.simulator.states[0].copy()
            rdms6 = observe_rdms(psi)
            a = int(np.random.randint(0, env.simulator.num_actions))
            i, j = env.simulator.actions[a]
            # ent = get_entanglements(psi)
            env.step([a], reset=False)
            qiskit_swap = np.all(get_preswap_gate(rdms6, i, j) == P)
            env_swap = env.simulator.preswaps_[0]
            res = (qiskit_swap == env_swap) # or (np.abs(ent[i] - ent[j]) < 1e-7)
            print('.' if res else 'F', end='', flush=True)
            result &= res
            failed += int(not res)
    print('\ntest_get_preswap_gate():', f'{failed}/{6*n_tests} failed')
    return result


def test_get_postswap_gate(n_tests=100):
    """
    Test if the returned post-swap gate in qiskit code agrees with the
    `_postswap` flag in the RL environment.
    """
    env = QuantumEnv(4, 1, obs_fn="phase_norm")
    P = np.eye(4,4,dtype=np.complex64)
    P[[1, 2]] = P[[2, 1]]
    result = True
    failed = 0

    for _ in range(n_tests):
        env.reset()
        env.simulator.set_random_states_()
        # Rollout a trajectory
        for _ in range(6):
            psi = env.simulator.states[0].copy()
            rdms6 = observe_rdms(psi)
            a = int(np.random.randint(0, env.simulator.num_actions))
            i, j = env.simulator.actions[a]
            # ent = get_entanglements(psi)
            env.step([a], reset=False)
            qiskit_swap = np.all(get_postswap_gate(rdms6, i, j) == P)
            env_swap = env.simulator.postswaps_[0]
            # This testline may fail if the entanglements are equal
            # That's why they are tested for equality
            res = (qiskit_swap == env_swap) #or (np.abs(ent[i] - ent[j]) < 1e-7)
            result &= res
            failed += int(not res)
            print('.' if res else 'F', end='', flush=True)
    print('\ntest_get_postwap_gate():', f'{failed}/{6*n_tests} failed')
    return result


def test_get_U(n_tests=100):
    """
    Test if the returned action gate U in qiskit code equals the  `_Us`
    attribute in the RL environment.
    """
    env = QuantumEnv(4, 1, obs_fn="phase_norm")
    result = True
    failed = 0
    P = np.eye(4,4,dtype=np.complex64)
    P[[1,2]] = P[[2,1]]

    for _ in range(n_tests):
        env = QuantumEnv(4, 1, obs_fn="phase_norm")
        env.reset()
        env.simulator.set_random_states_()
        # Rollout a trajectory
        for _ in range(6):
            psi = env.simulator.states[0].copy()
            rdms6 = observe_rdms(psi)
            a = int(np.random.randint(0, env.simulator.num_actions))
            i, j = env.simulator.actions[a]
            U = get_U(rdms6, i, j, apply_preswap=True, apply_postswap=False)
            # Since `apply_preswap` is True, the returned U is already multiplied
            # on the right with swap gate (because qubits i,j needs to be swapped).
            # The effect of this matrix multiplication is that columns 2,3
            # in U are swapped.
            if np.all(get_preswap_gate(rdms6, i, j) == P):
                U = U[:, [0,2,1,3]]
            env.step([a], reset=False)
            res = np.all(np.isclose(U, env.simulator.Us_[0], atol=1e-7))
            failed += int(not res)
            print('.' if res else 'F', end='', flush=True)
        result &= res
    print('\ntest_get_U():', f'{failed}/{6*n_tests} failed')
    return result


def test_peek_next_4q(n_tests=100):
    """Test if peek_next_4q() returns the same states as the environment's apply()."""
    env = QuantumEnv(4, 1, obs_fn="phase_norm")
    result = True
    failed = 0

    for _ in range(n_tests):
        env.reset()
        env.simulator.set_random_states_()
        res = True
        # Rollout a trajectory
        for _ in range(6):
            psi = env.simulator.states[0].copy()
            a = int(np.random.randint(0, env.simulator.num_actions))
            i, j = env.simulator.actions[a]
            env.step([a], reset=False)
            U = get_U(observe_rdms(psi), i, j, True, True)
            phi = env.simulator.states[0]
            ent = get_entanglements(phi)
            phi2, ent2, _ = peek_next_4q(psi, U, i, j)
            res = np.isclose(fidelity(phi, phi2), 1.0, atol=1e-2)
            res &= np.all(np.isclose(ent, ent2, atol=1e-5))
            failed += int(not res)
            print('.' if res else 'F', end='', flush=True)
        result &= res
    print('\ntest_peek_next_4q():', f'{failed}/{6*n_tests} failed')
    return result


def test_rdms_noise(
        policy: Literal['universal', 'equivariant', 'transformer', 'ordered'],
        state
    ):
    I = np.eye(4, dtype=np.complex64)
    env = QuantumEnv(4, 1, obs_fn="rdm_2q_half")
    env.reset()

    result = True
    for noise in [1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        env.states = np.expand_dims(state, 0)
        nsteps = 5 if policy == 'universal' else 8
        for _ in range(nsteps):
            rdms = observe_rdms(env.simulator.states[0])
            noisy_rdms = rdms + \
                np.random.normal(scale=noise, size=rdms.shape) + \
                1j * np.random.normal(scale=noise, size=rdms.shape)
            U, i, j = get_action_4q(noisy_rdms, policy)
            a = env.simulator.actToKey[(i,j)]
            _, _, done, _, _ = env.step([a], reset=False)
            assert np.all(np.isclose(I, U @ U.T.conj(), atol=1e-6))
            # Break loop early if policy != 'universal'
            if policy != 'universal' and done[0]:
                break
        print('.' if done else 'F', end='')
        result &= done
    return result


def do_qiskit_rollout(state, policy):
    P = np.eye(4,4, dtype=np.complex64)
    P[[1,2]] = P[[2,1]]
    s = state.ravel()
    actions, states, Us, RDMs = [], [], [], []
    entanglements = []
    preswaps, postswaps = [], []

    n = 0
    done = False
    while not done and n < 10:
        states.append(s.copy())
        rdms = observe_rdms(s)
        RDMs.append(rdms)
        entanglements.append(get_entanglements(s))

        U, i, j = get_action_4q(rdms, policy)
        preswaps.append(np.all(get_preswap_gate(rdms, i, j) == P))
        postswaps.append(np.all(get_postswap_gate(rdms, i, j) == P))
        a = ACTION_SET_REDUCED.index((i,j))
        s_next, ent, _ = peek_next_4q(s, U, i, j)
        done = np.all(ent < 1e-3)
        # states in RL environemnt are phase normed
        s = phase_norm(s_next.reshape(1,2,2,2,2)).ravel()
        actions.append(a)
        Us.append(get_U(rdms, i, j,apply_preswap=True, apply_postswap=False))
        n += 1

    states.append(s)
    RDMs.append(observe_rdms(s))
    entanglements.append(get_entanglements(s))

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "entanglements": np.array(entanglements),
        "Us": np.array(Us),
        "RDMs": np.array(RDMs),
        "preswaps": np.array(preswaps),
        "postswaps": np.array(postswaps)
    }


def do_rlenv_rollout(state, policy):
    env = QuantumEnv(4, 1, obs_fn="rdm_2q_mean_real")
    env.reset()
    env.simulator.states = state.reshape(1,2,2,2,2)

    actions, states, Us = [], [], []
    entanglements = []
    RDMs = []
    preswaps, postswaps = [], []

    policy_net = None
    if policy == "transformer":
        policy_net = TRANSFORMER_POLICY
    elif policy == "ordered":
        policy_net = QS_POLICY
    else:
        raise ValueError("Test is valid only with 'ordered' or 'transformer' "
                         "policies.")

    done = False
    n = 0
    while not done and n < 10:
        s = env.simulator.states.copy()
        states.append(s.ravel())
        entanglements.append(get_entanglements(s.ravel()))
        RDMs.append(observe_rdms(s))
        a = eval_policy(env.obs_fn(s)[0], policy_net)
        _, _, done, _, _ = env.step([a], reset=False)
        actions.append(a)
        Us.append(env.simulator.Us_.copy())
        preswaps.append(env.simulator.preswaps_[0])
        postswaps.append(env.simulator.postswaps_[0])
        n += 1

    s = env.simulator.states.copy().ravel()
    states.append(s)
    RDMs.append(observe_rdms(s))
    entanglements.append(get_entanglements(s))

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "entanglements": np.array(entanglements),
        "Us": np.array(Us),
        "RDMs": np.array(RDMs),
        "preswaps": np.array(preswaps),
        "postswaps": np.array(postswaps)
    }


def test_rollout_equivalence(policy, n_tests=200):
    env = QuantumEnv(4, 1, obs_fn="phase_norm")
    result = True
    failed = 0
    # diverging_states = []

    for _ in range(n_tests):
        env.reset()
        env.simulator.set_random_states_()
        psi = env.simulator.states.ravel().copy()
        qiskit_rollout = do_qiskit_rollout(psi.copy(), policy)
        rl_env_rollout = do_rlenv_rollout(psi.copy(), policy)
        res = True
        # Test action selection
        if len(qiskit_rollout['actions']) != len(rl_env_rollout['actions']):
            res = False
        else:
            res = np.all(qiskit_rollout['actions'] == rl_env_rollout['actions'])
        # Test states overlap / fidelity
        overlaps = []
        for x, y in zip(qiskit_rollout["states"], rl_env_rollout["states"]):
            overlaps.append(fidelity(x, y))
        overlaps = np.array(overlaps)
        res &= np.all(np.isclose(np.abs(overlaps - 1.0), 0.0, atol=1e-2))
        # if not res:
        #     diverging_states.append(psi)
        #     print()
        #     print(qiskit_rollout['entanglements'])
        #     print(rl_env_rollout['entanglements'])
        #     print(qiskit_rollout['actions'], rl_env_rollout['actions'])
        #     print(qiskit_rollout['preswaps'], rl_env_rollout['preswaps'])
        #     print(qiskit_rollout['postswaps'], rl_env_rollout['postswaps'])
        #     print(overlaps)
        #     print()
        print('.' if res else 'F', end='', flush=True)
        failed += int(not res)
        result &= res
    print('\ntest_rollout_equivalence():', f'{failed}/{n_tests} failed')
    # np.save('diverging-states.npy', np.array(diverging_states))
    return result


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    test_get_preswap_gate(100)
    test_get_postswap_gate(100)
    test_get_U(100)
    test_peek_next_4q(100)
    test_rollout_equivalence("ordered", 100)

    # # Test |BB>|BB> and all it's permutations state wih noise added to RDMs
    print('Testing |BB>|BB> state with universal circuit:')
    bell = np.sqrt(1/2) * np.array([1.0, 0.0, 0.0, 1.0])
    psi = np.kron(bell, bell).reshape(2, 2, 2, 2)   # 01-23 entangled
    for P in itertools.permutations(range(4)):
        print(f'\n\tPermutation {P}: ', end='')
        test_rdms_noise('universal', np.transpose(psi, P))

    # # Test |BB>|BB> and all it's permutations state wih noise added to RDMs
    print('\n\nTesting |BB>|BB> state with ordered policy:')
    bell = np.sqrt(1/2) * np.array([1.0, 0.0, 0.0, 1.0])
    psi = np.kron(bell, bell).reshape(2, 2, 2, 2)   # 01-23 entangled
    for P in itertools.permutations(range(4)):
        print(f'\n\tPermutation {P}: ', end='')
        test_rdms_noise('ordered', np.transpose(psi, P))
    print()
