import itertools
import warnings

import pytest
import numpy as np
import torch

import context
from src.quantum_env import rdm_2q_mean_real
from src.stategen import sample_haar_full
from src.observations import rdm2m


@pytest.mark.parametrize(
        "num_qubits,num_envs",
        itertools.product((4,6,8,12,16), (1,16,32))
)
def test_rdm2m(num_qubits: int, num_envs: int):
    """
    Test passes if `rdm2m()` returns the same tensor as `rdm_2q_mean_real()`
    """
    x_n = np.array([sample_haar_full(num_qubits) for _ in range(num_envs)])
    a = rdm_2q_mean_real(x_n)

    # Test on CPU
    x_t = torch.from_numpy(x_n)
    b = rdm2m(x_t, device="cpu")
    assert np.allclose(a, b, atol=1e-7)

    # Test on CUDA if available
    if not torch.cuda.is_available():
        warnings.warn(
            "No CUDA device found. " \
            "Skipping `rdm2m()` test for `device=\"cuda\"`"
        )
        return
    c = rdm2m(x_t.to(device="cuda"), device="cuda")
    assert np.allclose(a, c.cpu().numpy(), atol=1e-7)