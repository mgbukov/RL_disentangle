import itertools
import pickle
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
from src.stategen import StateGenerator, sample_haar_full, sample_haar_product
from src.util import fidelity, sqe, torch_sqe


PATH_ROLLOUTS_FILE = "data/tests/rollouts.pickle"


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


def test_sqe():
    """
    Tests if the NumPy implementation of `sqe()` is
    equivalent to the PyTorch implementation `torch_sqe()` in terms of outputs
    """
    np.random.seed(10)

    s0 = sample_haar_product(4, 4, 4)
    s1 = np.array([sample_haar_product(4, 2, 4) for _ in range(16)])
    s2 = np.array([sample_haar_product(6, 3, 3) for _ in range(8)])
    s3 = sample_haar_product(8, 2, 4)
    s4 = sample_haar_product(12, 2, 4)
    s5 = sample_haar_product(16, 2, 4)

    for s in (s1, s2):
        numpy_ent = sqe(s, batched=True)
        torch_ent = torch_sqe(torch.from_numpy(s), batched=True)
        assert np.allclose(numpy_ent, torch_ent.cpu().numpy())

    for s in (s0, s3, s4, s5):
        numpy_ent = sqe(s, batched=False)
        torch_ent = torch_sqe(torch.from_numpy(s), batched=False)
        assert np.allclose(numpy_ent, torch_ent.cpu().numpy())


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
        itertools.product((4,6,8,12), (1,16,32), (False, True), ("cpu", "cuda"))
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

    for k in range(20):
        # Pick random actions. Convert them to qubit indices
        actions = np.random.choice(numpy_sim.num_actions, size=(num_envs,))
        indices = [numpy_sim.actions[a] for a in actions]
        # Step
        numpy_sim.apply(actions)
        torch_sim.apply(indices)

        # --- Test preswaps
        assert np.all(numpy_sim.preswaps_ == torch_sim.preswaps_.numpy())
        # --- Test postswaps
        assert np.all(numpy_sim.postswaps_ == torch_sim.postswaps_.numpy())

        # --- Test eigenvalues
        numpy_rhos = numpy_sim.rhos_
        torch_rhos = torch_sim.rhos_.cpu().numpy()
        assert np.allclose(numpy_rhos, torch_rhos, atol=1e-4)

        # If more than 1 eigenvalue is 0, then eigenvectors are not defined
        # uniquely. In this case, we end the rollout and exit
        if np.any((numpy_rhos < 1e-6).sum(axis=1) > 1):
            print(f"More than one eigenvalue of an RDM is 0 at step {k}")
            break

        # --- Test Us
        numpy_Us = numpy_sim.Us_
        torch_Us = torch_sim.Us_.cpu().numpy()
        overlap = np.abs(np.einsum("bij,bij->bj", numpy_Us, torch_Us.conj())) ** 2
        equal_Us = np.all(overlap > 0.99)
        # if not equal_Us:
        #     print(f"Us differ at step {k}!\n")
        #     a = (numpy_sim.Us_ - torch_sim.Us_.cpu().numpy())
        #     ai = np.argmax(a.sum(axis=(1,2)))
        #     print(ai)
        #     print("\nref Us:\n", numpy_sim.Us_[ai].round(3))
        #     print("\ncuda Us:\n", torch_sim.Us_[ai].cpu().numpy().round(3))
        #     print("\nref rhos:\n", numpy_sim.rhos_[ai].round(3))
        #     print("\ncuda rhos:\n", torch_sim.rhos_[ai].cpu().numpy().round(3))
        #     print("\nref rmds:\n", numpy_sim.rdms_[ai].round(3))
        #     print("\ncuda rdms:\n", torch_sim.rdms_[ai].cpu().numpy().round(3))
        assert equal_Us

        # --- Test RDMs
        equal_rdms = np.allclose(numpy_sim.rdms_, torch_sim.rdms_.cpu().numpy(), atol=1e-4)
        # if not equal_rdms:
        #     print(f"\nRDMs differ at step {k}!\n")
        #     print("\n\tRDMs in NumPy simulator:\n", numpy_sim.rdms_.round(3))
            # print("\n\tRDMs in Torch simulator:\n", torch_sim.rdms_.cpu().numpy().round(3))
        assert equal_rdms

        # --- Test single qubit entanglements
        assert np.allclose(
            numpy_sim.entanglements,
            torch_sim.entanglements.numpy(), atol=1e-4
        )

        # --- Test order of qubits
        # Ignore the order of qubits which have entanglement below 1e-5
        qubits_order = numpy_sim.qubits_order == torch_sim.qubits_order.cpu().numpy()
        very_low_ent = numpy_sim.entanglements < 1e-5
        assert np.all(qubits_order | very_low_ent)

        # --- Test fidelity
        x = fidelity(numpy_sim.states.reshape(num_envs, -1),
                     torch_sim.states.cpu().numpy().reshape(num_envs, -1))
        assert np.all(x > 0.999)


@pytest.mark.parametrize(
        "num_qubits,num_envs,device",
        itertools.product((4, 8, 12, 16), (1, 8), ("cpu", "cuda"))
)
def test_step(num_qubits: int, num_envs: int, device: str):
    # Check if CUDA is available on the current system
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("No CUDA device found. Skipping test...")
        return

    # Initialize NumPy and PyTorch RL environments
    sgen = StateGenerator(sample_haar_full, num_qubits)
    numpy_env = NumpyEnv(num_qubits, num_envs, obs_fn="phase_norm",
                         epsi=1e-3, state_generator=sgen, reward_fn="relative_delta")
    torch_env = TorchEnv(num_qubits, num_envs, obs_fn="vec",
                         epsi=1e-3, state_generator=sgen,
                         reward_fn="relative_delta",
                         device=device)

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
        assert np.all(overlap > 0.999)

        # --- Test terminated
        # if not np.all(n_t == t_t.cpu().numpy()):
        #     with np.printoptions(suppress=True):
        #         print(numpy_env.simulator.entanglements.round(4))
        #         print(torch_env.simulator.entanglements.numpy().round(4))
        assert np.all(n_t == t_t.cpu().numpy())

        # --- Test truncated
        # if not np.all(n_tr == t_tr.cpu().numpy()):
        #     print(n_tr, t_tr)
        assert np.all(n_tr == t_tr.cpu().numpy())

        # --- Test rewards
        assert np.allclose(n_r, t_r.cpu().numpy(), atol=1e-4)

        # --- Test info
        if "episode" in n_info:
            assert np.allclose(n_info["episode"]["l"].astype(np.float32),
                               t_info["episode"]["l"].astype(np.float32),
                               equal_nan=True)
            assert np.allclose(n_info["episode"]["r"].astype(np.float32),
                               t_info["episode"]["r"].astype(np.float32),
                               equal_nan=True)
        else:
            assert n_info == t_info


@pytest.mark.parametrize("num_qubits", (4,5,10,12,15))
def test_disentangle(num_qubits: int):
    with open(PATH_ROLLOUTS_FILE, mode='rb') as f:
        data = pickle.load(f)

    # for record in data:
    #     state = record["state"]
    #     q = state.ndim
    #     if q != num_qubits:
    #         continue
    #     trajectory = record["actions"]
    #     entanglements = record["entanglements"]
    #     env = VectorQuantumState(num_qubits, 1,)
    #     env.states = np.expand_dims(state, 0)
    #     for i, qpair in enumerate(trajectory):
    #         a = env.actToKey[tuple(qpair)]
    #         print(qpair, entanglements[i].round(3), env.entanglements.round(3))
    #         env.apply([a])
    #     assert np.all(env.entanglements <= record["epsi"])

    # ----

    # for record in data:
    #     state = record["state"]
    #     q = state.ndim
    #     if q != num_qubits:
    #         continue
    #     trajectory = record["actions"]
    #     entanglements = record["entanglements"]
    #     env = VectorizedQState(num_qubits, 1, device="cpu")
    #     env.states = torch.from_numpy(np.expand_dims(state, 0))
    #     for i, qpair in enumerate(trajectory):
    #         print(qpair, entanglements[i].round(3), env.entanglements.numpy().round(3))
    #         env.apply([tuple(qpair)])
    #     assert torch.all(env.entanglements <= record["epsi"])

    # ----

    for record in data:
        state = record["state"]
        q = state.ndim
        if q != num_qubits:
            continue
        trajectory = record["actions"]
        env = TorchEnv(num_qubits, 1, obs_fn="rdm2m", device="cpu", fast_obs=True, fast_ents=True)
        env.reset()
        env.simulator.states = torch.from_numpy(np.expand_dims(state, 0))
        for i, qpair in enumerate(trajectory):
            a = env.act_to_key[tuple(qpair)]
            env.step([a], reset=False)
        assert torch.all(env.simulator.entanglements <= record["epsi"])


@pytest.mark.parametrize(
        "num_qubits,num_envs,device",
        itertools.product((4, 5, 6, 8, 12), (1,4,8), ("cpu",))
)
def test_swaps(num_qubits: int, num_envs: int, device: str):
    np.random.seed(10)
    # Check if CUDA is available on the current system
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("No CUDA device found. Skipping test...")
        return

    # Initialize NumPy and PyTorch RL environments
    sgen = StateGenerator(sample_haar_full, num_qubits)
    env = NumpyEnv(num_qubits, num_envs, obs_fn="phase_norm",
                   epsi=1e-3, state_generator=sgen)
    env.reset()
    env = TorchEnv(num_qubits, num_envs, obs_fn="vec",
                   epsi=1e-3, state_generator=sgen, fast_ents=False, device=device)
    env.reset()

    ax0 = np.arange(num_envs)
    for n in range(10):
        # Pick random actions
        actions = np.random.choice(env.num_actions, size=(num_envs,))
        indices = np.array([env.actions[a] for a in actions])
        # Relation before U
        _q0 = env.simulator.entanglements[ax0, indices[:, 0]]
        _q1 = env.simulator.entanglements[ax0, indices[:, 1]]
        # print(_q0, _q1)
        _rel = (_q0 >= _q1 + 1e-3)
        # Step
        env.step(actions, reset=False)
        # Relation after U
        q0_ = env.simulator.entanglements[ax0, indices[:, 0]]
        q1_ = env.simulator.entanglements[ax0, indices[:, 1]]
        # print(q0_, q1_)
        rel_ = (q0_ >= q1_ + 1e-3 )
        assert torch.all(_rel == rel_)
        # print()


@pytest.mark.parametrize("num_envs, num_qubits", itertools.product((1,8,), (4,8,12)))
def test_fast_obs(num_envs, num_qubits):

    # Initialize state generator
    sg = StateGenerator(sample_haar_full, num_qubits)

    for _ in range(10):
        env_params = {
            "num_qubits":           num_qubits,
            "num_envs":             num_envs,
            "epsi":                 1e-3,
            "max_episode_steps":    100,
            "obs_fn":               "rdm2m",
            "state_generator":      sg,
        }
        base_env = TorchEnv(fast_obs=False, **env_params)
        base_env.reset()
        initial_states = base_env.simulator.states

        fast_env = TorchEnv(fast_obs=True, **env_params)
        fast_env.reset()
        fast_env.set_states(initial_states)

        assert torch.all(base_env.simulator.states == fast_env.simulator.states)

        for i in range(30):
            acts = np.random.choice(
                list(range(base_env.single_action_space.n)),
                base_env.num_envs
            )
            base_obs, base_reward, base_term, base_trunc, _ = base_env.step(acts, reset=False)
            fast_obs, fast_reward, fast_term, fast_trunc, _ = fast_env.step(acts, reset=False)

            assert torch.all(base_term == fast_term)
            assert torch.all(base_trunc == fast_trunc)
            assert torch.allclose(base_reward, fast_reward)

            # Compare states only for non-terminated / non-truncated trajectories
            for n in range(num_envs):
                if base_term[n] == False:
                    assert torch.allclose(base_obs[n], fast_obs[n], atol=1e-5)

            if torch.any(base_term | base_trunc):
                break


@pytest.mark.parametrize("num_envs, num_qubits", itertools.product((1,4), (4,5)))
def test_fast_ents(num_envs, num_qubits):

    # Initialize state generator
    sg = StateGenerator(sample_haar_full, num_qubits)

    for _ in range(100):
        env_params = {
            "num_qubits":           num_qubits,
            "num_envs":             num_envs,
            "epsi":                 1e-3,
            "max_episode_steps":    100,
            "obs_fn":               "rdm2m",
            "reward_fn":            "relative_delta",
            "state_generator":      sg,
        }
        base_env = TorchEnv(fast_ents=False, swaps=False, **env_params)
        base_env.reset()
        initial_states = base_env.simulator.states

        fast_env = TorchEnv(fast_ents=True, swaps=False, **env_params)
        fast_env.reset()
        fast_env.set_states(initial_states)

        assert torch.all(base_env.simulator.states == fast_env.simulator.states)

        for i in range(30):
            print(i)
            acts = np.random.choice(
                list(range(base_env.single_action_space.n)),
                base_env.num_envs
            )
            base_obs, base_reward, base_term, base_trunc, _ = base_env.step(acts, reset=False)
            fast_obs, fast_reward, fast_term, fast_trunc, _ = fast_env.step(acts, reset=False)

            assert torch.all(base_term == fast_term)
            assert torch.all(base_trunc == fast_trunc)
            if not torch.allclose(base_reward, fast_reward, atol=1e-3):
                print(base_env.simulator.entanglements)
                print(fast_env.simulator.entanglements)
                print(base_env.simulator.postswaps_)
                print(fast_env.simulator.postswaps_)
                print(base_reward)
                print(fast_reward)
            assert torch.allclose(base_reward, fast_reward, atol=1e-3)

            # Compare states only for non-terminated / non-truncated trajectories
            for n in range(num_envs):
                assert torch.allclose(base_env.simulator.entanglements[n],
                                     fast_env.simulator.entanglements[n], atol=1e-4)

            if torch.any(base_term | base_trunc):
                break
