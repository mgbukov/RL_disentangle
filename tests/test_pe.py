import itertools
import numpy as np
import torch
import sys
from itertools import permutations
import pdb

sys.path.append("..")
from src.policies.equivariant_policy import (
    PermutationLayer,
    PermutationLayer2,
    ComplexDenseNet,
    PEPolicy
)
from src.envs.rdm_environment import QubitsEnvironment


def calculate_rdms(state):
    L = state.ndim
    result = []
    for q0, q1 in permutations(range(L), 2):
        P = (q0, q1) + tuple(q for q in range(L) if q not in (q0, q1))
        psi = np.transpose(state, P).reshape(4, -1)
        rdm = psi @ psi.T.conj()
        result.append(rdm)
    return np.stack(result)


def derangements(iterable, n=0, ge=False):
    """
    Returns all permutations of `iterable` with exactly `n` fixed
    points if `ge` == False (the returned list may be empty).
    If `ge` == True, returns all permutations having at least `n` fixed points.
    """
    x = np.asarray(iterable)
    x0 = x.copy()
    result = []
    for p in permutations(x):
        fixed_points = np.sum(np.array(p) == x0)
        if (not ge and fixed_points == n) or (ge and fixed_points >= n):
            result.append(p)
    return result


def transpositions(iterable, order=1):
    """
    Returns all transpositions of `iterable` if `order` == 1, including the
    identity.
    If `order` > 1, returns all transpositions of `transpositions(iterable, order-1)`
    """
    if order == 1:
        result = set([tuple(iterable)])
        n = len(list(iterable))
        for i, j in permutations(range(n), 2):
            y = list(iterable)
            y[i], y[j] = y[j], y[i]
            result.add(tuple(y))
        return sorted(result)
    elif order > 1:
        transp = transpositions(iterable, order - 1)
        n = len(list(iterable))
        result = set()
        for x in transp:
            for i, j in permutations(range(n), 2):
                y = list(x)
                y[i], y[j] = y[j], y[i]
                result.add(tuple(y))
        return sorted(result)
    else:
        return [tuple(iterable)]


def qubits_permutation_to_rdm_permutation(qubits_indices):
    rdm_indices = [p for p in permutations(qubits_indices, 2)]
    n = len(qubits_indices)
    rankings = {p: i for i, p in enumerate(permutations(range(n), 2))}
    return np.array([rankings[p] for p in rdm_indices])


def test_pe_layer_init():
    layer = PermutationLayer(20, 16, 64, ())
    x = torch.complex(torch.randn(1, 20, 16), torch.randn(1, 20, 16))
    y = layer(x)
    assert y.shape == (x.shape[0], 20, 64)

    layer = PermutationLayer(20, 16, 64, (512, 256, 128))
    x = torch.complex(torch.randn(1, 20, 16), torch.randn(1, 20, 16))
    y = layer(x)
    assert y.shape == (x.shape[0], 20, 64)

    subnet = ComplexDenseNet(32, [128, 64, 32], 1)
    layer = PermutationLayer2(subnet)
    x = torch.complex(torch.randn(1, 20, 16), torch.randn(1, 20, 16))
    y = layer(x)
    assert y.shape == (x.shape[0], 20, 1)


def test_PElayer_equivariance(n_qubits=3, order=1):
    """
    Tests the permutation equivariance property of `PermutationLayer` over
    `order` repeated traspositions of qubits. If `order` is 1, this reduces to
    swaps of 2 qubits.
    """
    env = QubitsEnvironment(n_qubits)
    env.set_random_states()
    in_features = 16
    out_features = 1
    n_inputs = env.num_actions

    # Initialize layer
    subnet = ComplexDenseNet(2 * in_features, [128, 64, 32], out_features)
    layer = PermutationLayer2(subnet)

    # Initialize inputs
    qubits_permutations = transpositions(range(n_qubits), order)
    permutation_keys = list(qubits_permutations)
    permutation_vals = []
    for p in qubits_permutations:
        psi = np.transpose(env.states[0], p)
        rdms = calculate_rdms(psi).reshape(n_inputs, in_features)
        permutation_vals.append(rdms)

    # pdb.set_trace()

    # Get outputs
    outputs = {}
    for key, val in zip(permutation_keys, permutation_vals):
        x = torch.from_numpy(val).unsqueeze(0)
        assert x.shape == (1, n_inputs, in_features)
        out = layer(x).detach().numpy()
        assert out.shape == (1, n_inputs, out_features)
        outputs[key] = out

    # Test equivariance
    test_result = True
    I = tuple(range(n_qubits))      # identity permutation
    I_out = outputs[I]              # identity output
    for k, v in outputs.items():
        rdm_permutation = qubits_permutation_to_rdm_permutation(k)
        inv_permutation = np.argsort(rdm_permutation)
        # v.shape == (1, n_inputs, out_features)
        v_inverse = v[:, inv_permutation, :]
        try:
            assert np.all(np.isclose(v_inverse, I_out))
        except AssertionError:
           test_result = False
           break
    return test_result



def _test_nn_equivariance(env, neural_network):

    TEST_RESULT = True
    n_qubits = env.L
    n_inputs = env.num_actions
    in_features = 16
    out_features = 1

    # Construct inputs
    permutation_keys = []
    permutation_vals = []
    for P in itertools.permutations(range(n_qubits), 2):
        # Permute state
        p = P + tuple(q for q in range(env.L) if q not in P)
        phi = np.transpose(env.states[0], p)
        rdms = calculate_rdms(phi).reshape(-1, 16)
        assert rdms.shape == (n_inputs, in_features)
        permutation_keys.append(p)
        permutation_vals.append(rdms)

    permutation_vals = np.stack(permutation_vals, axis=0)
    assert permutation_vals.shape == (n_inputs, n_inputs, in_features)

    result = {}
    for i in range(len(permutation_vals)):
        x = torch.from_numpy(permutation_vals)[i:i+1]
        assert x.shape == (1, n_inputs, in_features)
        # We need abs() here, because `neural_network` can be a single
        # `PermutationLayer`
        outputs = neural_network(x).detach().numpy()
        assert outputs.shape == (1, n_inputs,)
        result[permutation_keys[i]] = outputs

    reference_permutation = tuple(range(n_qubits))
    reference_output = result[reference_permutation][0].ravel()
    for k, v in result.items():
        rdm_perm = qubits_permutation_to_rdm_permutation(k)
        inv_perm = np.argsort(rdm_perm)
        v_permuted = v[0].ravel()[inv_perm]
        try:
            assert np.all(np.isclose(v_permuted, reference_output))
        except AssertionError:
            TEST_RESULT = False
            break
    return TEST_RESULT




def test_pe_policy_equivariance(n_qubits=3):
    env = QubitsEnvironment(n_qubits)
    assert _test_nn_equivariance(env, PEPolicy(env.num_actions, (16,) ))
    assert _test_nn_equivariance(env, PEPolicy(env.num_actions, (16, 64,) ))
    assert _test_nn_equivariance(env, PEPolicy(env.num_actions, (16, 64, 32) ))
    return True


if __name__ == '__main__':
    test_pe_layer_init()

    test_PElayer_equivariance(5, 3)
    for order in range(1, 5):
        print(f'\nTesting equivariance of transpositions (order = {order})')
        for n_qubits in range(3, 8):
            if (test_PElayer_equivariance(n_qubits, order)):
                print('.', end='')
            else:
                print('F', end='')
        print()
    # for n in range(3, 7):
    #     res = test_pe_layer_equivariance(n)
    #     if res:
    #         print('.', end='', flush=True)
    #     else:
    #         print('F', end='', flush=True)
    # print()
    # for n in range(3, 7):
    #     res = test_pe_policy_equivariance(n)
    #     if res:
    #         print('.', end='', flush=True)
    #     else:
    #         print('F', end='', flush=True)
    # print()