import itertools
import warnings

import pytest
import numpy as np
import torch

import context
from src.quantum_state import (
    VectorQuantumState,
    permute_qubits as numpy_permute_qubits,
    phase_norm as numpy_phase_norm,
    calculate_q0_q1_entropy_from_rhos as numpy_sqe_from_rhos
)
from src.qenv import (
    QEnv as TorchEnv,
    VectorizedQState,
    permute_qubits as torch_permute_qubits,
    phase_norm as torch_phase_norm,
    calculate_sqe_from_rhos as torch_sqe_from_rhos
)
from src.quantum_env import QuantumEnv as NumpyEnv
from src.stategen import StateGenerator, sample_haar_full
from src.util import fidelity


@pytest.mark.parametrize(
    "num_qubits,num_envs,device",
    itertools.product((4,6,8,12), (1,16,32), ("cpu", "cuda"))
)
def test_permute_qubits(num_qubits: int, num_envs: int, device: str):
    """
    Tests if the NumPy implementation of `permute_qubits()` is equivalent
    to the PyTorch implementation in terms of outputs
    """
    # Check if CUDA is available on the current system
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("No CUDA device found. Skipping test...")
        return
    # Sample batch of states
    np.random.seed(7)
    x = np.array([sample_haar_full(num_qubits) for _ in range(num_envs)])
    qubit_pairs = itertools.cycle(itertools.combinations(range(num_qubits), 2))
    # Generate random qubit indicies (actions)
    qubit_indices = list(itertools.islice(qubit_pairs, num_envs))
    
    # --- Test permutation (i,j) -> (0,1)
    a = numpy_permute_qubits(x, qubit_indices, num_qubits, inverse=False, inplace=False)
    b = torch.from_numpy(x).to(device=device)
    torch_permute_qubits(b, qubit_indices, inverse=False)
    assert np.all(a == b.cpu().numpy())

    # --- Test reverse permutation (0,1) -> (i,j)
    a = numpy_permute_qubits(x, qubit_indices, num_qubits, inverse=True, inplace=False)
    b = torch.from_numpy(x).to(device=device)
    torch_permute_qubits(b, qubit_indices, inverse=True)
    assert np.all(a == b.cpu().numpy())


@pytest.mark.parametrize(
        "num_qubits,num_envs,device",
        itertools.product((4,6,8,12), (1,16,32), ("cpu", "cuda"))
)
def test_phase_norm(num_qubits: int, num_envs: int, device: str):
    """
    Tests if the NumPy implementation of `phase_norm()` is equivalent
    to the PyTorch implementation in terms of outputs
    """
    # Check if CUDA is available on the current system
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("No CUDA device found. Skipping test...")
        return
    # Sample batch of states
    np.random.seed(7)
    x = np.array([sample_haar_full(num_qubits) for _ in range(num_envs)])
    z = torch.from_numpy(x).to(device=device)
    a = numpy_phase_norm(x)
    b = torch_phase_norm(z)
    assert np.allclose(a, b.cpu().numpy())


@pytest.mark.parametrize(
        "num_qubits,num_envs,device",
        itertools.product((4,6,8,12), (1,16,32), ("cpu", "cuda"))
)
def test_sqe_from_rhos(num_qubits: int, num_envs: int, device: str):
    """
    Tests if the NumPy implementation of `calculate_sqe_from_rhos()` is
    equivalent to the PyTorch implementation in terms of outputs
    """
    # Check if CUDA is available on the current system
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("No CUDA device found. Skipping test...")
        return
    # Sample batch of states
    np.random.seed(7)
    states = np.array([sample_haar_full(num_qubits) for _ in range(num_envs)])
    # Compute eigenvalues of RDMs
    X = states.reshape(num_envs, 4, -1)
    rdms = X @ np.transpose(X.conj(), [0, 2, 1])
    rdms += np.finfo(rdms.dtype).eps * np.diag([0.0, 1.0, 2.0, 4.0])
    rhos, _ = np.linalg.eigh(rdms)
    # Compute entanglement
    a0, a1 = numpy_sqe_from_rhos(rhos)
    b0, b1 = torch_sqe_from_rhos(torch.from_numpy(rhos).to(device=device))
    assert np.allclose(a0, b0.cpu().numpy())
    assert np.allclose(a1, b1.cpu().numpy())


@pytest.mark.parametrize(
        "num_qubits,num_envs,swaps,device",
        itertools.product((3,4,8,16), (1,16,32), (False, True), ("cpu", "cuda"))
)
def test_apply(num_qubits: int, num_envs: int, swaps: bool, device: str):
    """
    Tests if the NumPy implementation of the `VectorizedQuantumState.apply()`
    method is equivalent to the PyTorch implementation in terms of outputs
    """
    # Check if CUDA is available on the current system
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("No CUDA device found. Skipping test...")
        return
    # Sample batch of states
    np.random.seed(7)
    states = np.array([sample_haar_full(num_qubits) for _ in range(num_envs)])

    # Initialize NumPy & PyTorch quantum state simulators
    numpy_sim = VectorQuantumState(num_qubits, num_envs, "reduced", swaps=swaps)
    torch_sim = VectorizedQState(num_qubits, num_envs, swaps=swaps, device=device)
    numpy_sim.states = states
    torch_sim.states = torch.from_numpy(states)
    assert np.allclose(numpy_sim.states, torch_sim.states.cpu().numpy())

    for k in range(40):
        # Pick random actions. Convert them to qubit indices
        actions = np.random.choice(numpy_sim.num_actions, size=(num_envs,))
        indices = [numpy_sim.actions[a] for a in actions]
        # Step
        numpy_sim.apply(actions)
        torch_sim.apply(indices)

        # Test preswaps
        # assert np.all(sim_ref.preswaps_ == sim_cpu.preswaps_.numpy())
        # Test postswaps
        # assert np.all(sim_ref.preswaps_ == sim_cpu.postswaps_.numpy())
        # # Test RDMs
        # equal_rdms = np.allclose(numpy_sim.rdms_, torch_sim.rdms_.cpu().numpy(), atol=1e-4)
        # if not equal_rdms:
        #     print(f"\nRDMs differ at step {k}!\n")
        #     print("\n\tRDMs in NumPy simulator:\n", numpy_sim.rdms_.round(3))
        #     print("\n\tRDMs in Torch simulator:\n", torch_sim.rdms_.cpu().numpy().round(3))
        # assert equal_rdms

        # /* Test Us */
        # !!!!!!!!!!!!!
        # `numpy.linalg.eigh()` and `torch.linalg.eigh()` produce different
        # different eigenvectors!
        # Testing equivalence of Us does not make sense
        # !!!!!!!!!!!!!
        # equal_Us = np.allclose(numpy_sim.Us_, torch_sim.Us_.cpu().numpy(), atol=1e-4)
        # if not equal_Us:
        #     print(f"Us differ at step {k}!\n")
        #     print("\nref Us:\n", numpy_sim.Us_.round(3))
        #     print("\ncuda Us:\n", torch_sim.Us_.cpu().numpy().round(3))
        # assert equal_Us
        # with np.printoptions(precision=4, suppress=True):
        #     print("\n", indices)
        #     print("Numpy entanglements:\n", numpy_sim.entanglements)
        #     print("Torch entanglements:\n", torch_sim.entanglements.numpy())
       
        # --- Test single qubit entanglements
        assert np.allclose(
            numpy_sim.entanglements,
            torch_sim.entanglements.numpy(), atol=1e-4
        )

        # --- Test order of qubits
        assert np.all(numpy_sim.qubits_order == torch_sim.qubits_order.numpy())
        
        # --- Test fidelity
        x = fidelity(numpy_sim.states.reshape(num_envs, -1),
                     torch_sim.states.cpu().numpy().reshape(num_envs, -1))
        assert np.all(x > 0.999)


@pytest.mark.parametrize(
        "num_qubits,num_envs,device",
        itertools.product((3, 4, 8, 12, 16), (1, 8), ("cpu", "cuda"))
)
def test_step(num_qubits: int, num_envs: int, device: str):
    # Check if CUDA is available on the current system
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("No CUDA device found. Skipping test...")
        return

    # Initialize NumPy and PyTorch RL environments
    sgen = StateGenerator(sample_haar_full, num_qubits)
    numpy_env = NumpyEnv(num_qubits, num_envs, obs_fn="phase_norm",
                         epsi=1e-3, state_generator=sgen)
    torch_env = TorchEnv(num_qubits, num_envs, obs_fn="vec",
                         epsi=1e-3, state_generator=sgen, device=device)
    
    np.random.seed(17)
    numpy_env.reset()
    torch_env.set_states(numpy_env.simulator.states)
    assert np.all(numpy_env.simulator.states == \
                  torch_env.simulator.states.cpu().numpy())

    for n in range(40):
        # Pick random actions
        actions = np.random.choice(torch_env.num_actions, size=(num_envs,))
        n_obs, n_r, n_t, n_tr, n_info = numpy_env.step(actions, reset=False)
        t_obs, t_r, t_t, t_tr, t_info = torch_env.step(list(actions), reset=False)

        # --- Test fidelity
        numpy_states = numpy_env.simulator.states.reshape(num_envs, -1)
        torch_states = torch_env.simulator.states.cpu().numpy().reshape(num_envs, -1)
        overlap = fidelity(numpy_states, torch_states, batched=True)
        if np.any(overlap < 0.999):
            print("\n\nstep:", n+1, "overlap:", overlap)
            # print("\nrdms:\n", numpy_env.simulator.rdms_.round(3), "\n", torch_env.simulator.rdms_.cpu().numpy().round(3))
            # print("\nUs:\n", numpy_env.simulator.Us_.round(3), "\n", torch_env.simulator.Us_.cpu().numpy().round(3))
            # print(numpy_env.simulator.entanglements.round(4))
            # print(torch_env.simulator.entanglements.cpu().numpy().round(4))
            # print(numpy_env.simulator.qubits_order)
            # print(torch_env.simulator.qubits_order)
            # # assert np.all(overlap > 0.99)
        assert np.all(overlap > 0.999)
        
        # --- Test terminated
        if not np.all(n_t == t_t.cpu().numpy()):
            with np.printoptions(suppress=True):
                print(numpy_env.simulator.entanglements.round(4))
                print(torch_env.simulator.entanglements.numpy().round(4))
        assert np.all(n_t == t_t.cpu().numpy())
        
        # --- Test truncated
        if not np.all(n_tr == t_tr.cpu().numpy()):
            print(n_tr, t_tr)
        assert np.all(n_tr == t_tr.cpu().numpy())

        # --- Test rewards
        assert np.allclose(n_r, t_r.cpu().numpy(), atol=1e-4)

        # --- Test info
        if "episode" in n_info:
            assert np.all(n_info["episode"]["l"] == t_info["episode"]["l"])
            assert np.all(n_info["episode"]["r"] == t_info["episode"]["r"])
        else:
            assert n_info == t_info
