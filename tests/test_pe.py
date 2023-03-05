import itertools
import numpy as np
import torch
import sys

sys.path.append("..")
from src.policies.equivariant_policy import PermutationLayer, PEPolicy
from src.envs.rdm_environment import QubitsEnvironment


def test_pe_layer_init():
    layer = PermutationLayer(20, 16, 64, ())
    x = torch.complex(torch.randn(1, 20, 16), torch.randn(1, 20, 16))
    y = layer(x)
    assert y.shape == (x.shape[0], 20, 64)

    layer = PermutationLayer(20, 16, 64, (512, 256, 128))
    x = torch.complex(torch.randn(1, 20, 16), torch.randn(1, 20, 16))
    y = layer(x)
    assert y.shape == (x.shape[0], 20, 64)


def test_pe_layer_equivariance():
    env = QubitsEnvironment(3, batch_size=1)
    n_inputs = env.num_actions
    in_features = 2 ** env.L
    out_features = 1
    layer = PermutationLayer(n_inputs, in_features, out_features, (512, 256))
    env.set_random_states()
    # Construct inputs with permutations
    permutation_keys = []
    permutation_vals = []
    # Add permuted
    for Po in itertools.permutations(range(3), 2):
        # Permute state
        p = Po + tuple(q for q in range(env.L) if q not in Po)
        phi = np.transpose(env.states[0], p)
        pvals = []
        for P in itertools.permutations(range(3), 2):
            sysA = P
            sysB = tuple(q for q in range(3) if q not in P)
            permutation = sysA + sysB
            pvals.append(np.transpose(phi, permutation).ravel())
        pvals = np.stack(pvals, axis=0)
        assert pvals.shape == (n_inputs, in_features)

        permutation_keys.append(p)
        permutation_vals.append(pvals)
    
    permutation_vals = np.stack(permutation_vals, axis=0)
    assert permutation_vals.shape == (n_inputs, n_inputs, in_features)
    # Add batch dimension
    result = {}
    for i in range(len(permutation_vals)):
        x = torch.from_numpy(permutation_vals)[i:i+1]
        outputs = np.abs(layer(x).detach().numpy())
        assert outputs.shape == (1, n_inputs, out_features)
        result[permutation_keys[i]] = outputs
    for k, v in result.items():
        print(k)
        print(v[0].ravel())


if __name__ == '__main__':
    test_pe_layer_init()
    test_pe_layer_equivariance()