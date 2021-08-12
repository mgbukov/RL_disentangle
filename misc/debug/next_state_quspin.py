import numpy as np
from itertools import combinations, product
from math import log2
from scipy.optimize import minimize
from scipy.linalg import expm
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

from noquspin_env import (QubitsEnvironment,
                          set_backend,
                          ent_entropy,
                          extract_qubits_and_op,
                          apply_gate_fast)
set_backend('numpy')


def construct_random_pure_state(L):
    real = np.random.uniform(-1, 1, 2 ** L).astype(np.float32)
    imag = np.random.uniform(-1, 1, 2 ** L).astype(np.float32)
    s = real + 1j * imag
    return s / np.linalg.norm(s)


def construct_operators(L):
    basis = spin_basis_1d(L)
    no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)

    generators = []
    for qubits in combinations(range(L), 2):
        for gate in product("xyz", repeat=2):
            gate = ''.join(gate)
            static = [[gate, [(1.0,) + qubits]]]
            H = hamiltonian(static, [], basis=basis, **no_checks)
            q0, q1 = qubits
            generators.append((f'{q0}-{q1}_{gate}', H.toarray()))

    idx_to_name = {}
    operators = []
    for i, (name, H) in enumerate(generators):
        idx_to_name[i] = name
        operators.append(H)
    operators = np.array(operators)
    return operators, idx_to_name


# Tests
#   1. Entropy calculated by QuSpin -> different trajectories
#   2. Entropy calculated by ent_entropy() -> different trajectories
#   3. Entropy calculated only for 1 subsystem instread of L (isolating each qubit)
#      Sub_sys_A is [0: L/2 ], Sub_sys_B is [L / 2 :]   -> different trajectories
#
def quspin_entropies(state, L):
    # entropies = [ent_entropy(state, [j]) for j in range(L)]
    # entropies = ent_entropy(state)
    B = spin_basis_1d(L)
    entropies = [B.ent_entropy(state, sub_sys_A=[j])['Sent_A'] for j in range(L)]
    result = np.array(entropies).astype(np.float32).ravel()
    # print('Q Entropies:', result)
    return result


# Tests
#   1. angle is of type float (double precision) in expm() call -> different trajectories
#   2. angle is of type np.float32 (single precision) in expm() call -> different trajectories
#   3. Return the MAX of `entropies` array
#   4. Return the L2 norm of `entropies` array
#
def F_quspin(angle, state, op, history):
    history.append(angle)
    gate = expm(-1j * np.float32(angle) * op)
    L = int(log2(len(state)))
    s = gate @ state
    entropies = quspin_entropies(s, L)
    # return np.linalg.norm(entropies)
    #print(angle, entropies)
    return np.sum(entropies)


def apply_operator(state, op):
    alpha_0 = np.pi / np.exp(1)
    hist = []
    res = minimize(
        F_quspin,
        alpha_0,
        args=(state, op, hist),
        method="Nelder-Mead",
        tol=1e-2
    )
    if res.success:
        angle = res.x[0]
        print('\n\nQuspin iter :', res.nit)
        print('Quspin angle:', angle)
    else:
        raise Exception(
            'Optimization procedure exited with '
            'an error.\n %s' % res.message)
    print('apply_oprator() called with angles:', [np.round(f, 5) for f in map(float, hist)])
    #print('QUSPIN',op.shape, state.shape, angle)
    return expm(-1j * angle * op) @ state


def rollout_quspin(state, actions):
    L = int(log2(len(state)))
    ops, _ = construct_operators(L)
    trajectory = []
    for a in actions:
        trajectory.append(state)
        state = apply_operator(state, ops[a])
    return np.array(trajectory)


def rollout_environment(state, actions):
    L = int(log2(len(state)))
    env = QubitsEnvironment(L, batch_size=1)
    env.set_state(state[np.newaxis, :])
    assert np.all(env._state == state[np.newaxis, :])
    trajectory = []
    for a in actions:
        # Because `env._state` is of shape (batch_size, 2 ** L)
        trajectory.append(np.squeeze(env._state))
        state = env.step([a])
    return np.array(trajectory)


# ########################################################################### #
#                       Test trajectories for 4 qubits                        #
#                                                                             #

L = 10
N = 10
env = QubitsEnvironment(L)
num_actions = env.num_actions

np.random.seed(44)
actions = np.random.randint(0, num_actions, N)
#print(actions)
#exit()

psi = construct_random_pure_state(L)


Qsteps = rollout_quspin(psi, actions)
#exit()

print('\n\n---------------------')
np.random.seed(44)
Esteps = rollout_environment(psi, actions)

print()
print()
print(np.isclose(Qsteps, Esteps, atol=1e-5))

# print(np.linalg.norm(Qsteps))
# print(np.linalg.norm(Esteps))

#exit()

# ########################################################################### #
#                       Test apply_gate_fast() for 4 qubits                   #
#                                                                             #
print('\n\nTest apply_gate_fast() deviation for 4 qubits...\n')
L = 4
N = 10
E = QubitsEnvironment(L)
num_actions = E.num_actions

np.random.seed(44)
angles = np.random.uniform(0, 2 * np.pi, num_actions)
quspin_operators, names = construct_operators(L)
psi = construct_random_pure_state(L)

for i in range(num_actions):
    q0, q1, op = extract_qubits_and_op(names[i])
    ang = angles[i]

    # Quspin
    q_gate = expm(-1j * ang * quspin_operators[i])
    e_gate = expm(-1j * ang * E.operators[op])

    phi_quspin = q_gate @ psi
    phi_fast = apply_gate_fast(psi, e_gate, q0, q1)
    print('\t', names[i], 'total deviation:')
    print(1.0 - np.abs(phi_fast.conj().T @ phi_quspin)**2 )


