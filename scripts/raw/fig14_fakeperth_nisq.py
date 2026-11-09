"""
Generate raw data for FIG. 14: RL agent on a noisy fake-device simulation
mimicking ibm_perth for three initial-state families:
  - Bell-Bell state
  - Shallow random circuit
  - GHZ state (4-qubit)

For each seed the script saves, per experiment folder:
  __2ents_agent_{exp}_{seed}.npy              (n_seeds, 1, length+1)
  __2ents_agent_noisy_{exp}_{seed}.npy
  __2ents_universal_{exp}_{seed}.npy
  __2ents_universal_noisy_{exp}_{seed}.npy
  __2full_ents_agent_noisy_{exp}_{seed}.npy
  __2full_ents_universal_noisy_{exp}_{seed}.npy
  __2conc_exact_{exp}_{seed}.npy              (n_seeds, 1, length+1, 6)
  __2conc_sim_{exp}_{seed}.npy
  __2entform_exact_{exp}_{seed}.npy           (n_seeds, 1, length+1)
  __2entform_sim_{exp}_{seed}.npy
  _rdm_agent_err0_shot{exp}_{seed}_{step}.npy
  _rdm_exact_agent_err0_shot{exp}_{seed}_{step}.npy
"""
import random
from itertools import permutations, combinations
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile
try:
    from qiskit.extensions import UnitaryGate                  # qiskit < 1.0
except ImportError:
    from qiskit.circuit.library import UnitaryGate             # qiskit >= 1.0
from qiskit.quantum_info import concurrence, entanglement_of_formation
from qiskit_aer import AerSimulator
try:
    from qiskit.providers.fake_provider import FakePerth       # qiskit < 2.0
except ImportError:
    from qiskit_ibm_runtime.fake_provider import FakePerth     # qiskit >= 2.0
from qiskit_experiments.library import StateTomography



ROOT        = Path(__file__).parent.parent.parent.parent
PATH_RAW    = ROOT / "data" / "raw" / "fig14"
PATH_AGENT  = ROOT / "models" / "fig7-8" / "4q-agent.pt"

ACTION_SET_4Q_FULL = list(permutations(range(4), 2))
ACTION_SET_4Q = [x for x in ACTION_SET_4Q_FULL if x[0] < x[1]]
ACTION_SET_5Q = list(combinations(range(5), 2))
ACTION_SET_6Q = list(combinations(range(6), 2))


def _load_policy(path):
    import sys as _sys
    _src_code = Path(__file__).parent.parent.parent   # src_code/
    if str(_src_code) not in _sys.path:
        _sys.path.insert(0, str(_src_code))
    agent = torch.load(path, map_location='cpu', weights_only=False)
    for enc in agent.policy_network.net:
        enc.activation_relu_or_gelu = 1
    agent.policy_network.eval()
    return agent.policy_network


ORDERED_POLICY_4Q = _load_policy(PATH_AGENT)


# ── Policy evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def _eval_policy(inputs, policy, savename=None, step=None):
    x = torch.from_numpy(inputs[None, :, :])
    logits = policy(x)[0]
    distr = torch.distributions.Categorical(logits=logits)
    if savename is not None:
        np.save(savename + f"dist_{step}.npy", distr.probs.numpy())
    return torch.argmax(distr.probs).item()


# ── Index helpers ─────────────────────────────────────────────────────────────
def _action_index(rdms, i, j):
    return ACTION_SET_4Q.index((i, j))


def _ij_from_index(a):
    return ACTION_SET_4Q[a]


# ── Input preparation ─────────────────────────────────────────────────────────
def _prepare_reduced_real_input(rdms: Sequence[np.ndarray]) -> np.ndarray:
    rdms = np.asarray(list(rdms))
    rdms_dict = {}
    for idx, rdm in zip(ACTION_SET_4Q, rdms):
        flipped = rdm[[0, 2, 1, 3], :][:, [0, 2, 1, 3]].copy()
        rdms_dict[idx] = 0.5 * (rdm + flipped)
    x = np.array([rdms_dict[k] for k in ACTION_SET_4Q]).reshape(6, 16)
    return np.hstack([x.real, x.imag])


# ── Swap gates ────────────────────────────────────────────────────────────────
def _sqe_from_rdm(rdm):
    rdm_i = np.trace(rdm.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    rdm_j = np.trace(rdm.reshape(2, 2, 2, 2), axis1=0, axis2=2)
    lam = np.linalg.svd(rdm_i, compute_uv=False)
    S_i = -np.sum(lam * np.log(lam + np.finfo(lam.dtype).eps))
    lam = np.linalg.svd(rdm_j, compute_uv=False)
    S_j = -np.sum(lam * np.log(lam + np.finfo(lam.dtype).eps))
    return S_i, S_j


def _preswap(rdms, i, j):
    I = np.eye(4, dtype=np.complex64)
    P = I.copy(); P[[1, 2]] = P[[2, 1]]
    S_i, S_j = _sqe_from_rdm(rdms[_action_index(rdms, i, j)])
    return I if S_i > S_j + np.finfo(S_i.dtype).eps else P


def _postswap(rdms, i, j):
    I = np.eye(4, dtype=np.complex64)
    P = I.copy(); P[[1, 2]] = P[[2, 1]]
    eps = np.finfo(rdms.real.dtype).eps
    rdm_ij = rdms[_action_index(rdms, i, j)]
    S_i, S_j = _sqe_from_rdm(rdm_ij)
    pre = I if S_i > S_j + eps else P
    rdm_tmp = pre @ rdm_ij @ pre.conj().T
    rdm_tmp += eps * np.diag([0, 1, 2, 4])
    _, U = np.linalg.eigh(rdm_tmp)
    max_col = np.abs(U).argmax(axis=0)
    for k in range(4):
        U[:, k] *= np.exp(-1j * np.angle(U[max_col[k], k]))
    U = pre @ U.conj().T @ pre
    rdm_next = U @ rdm_ij @ U.conj().T
    S_i_new, S_j_new = _sqe_from_rdm(rdm_next)
    return P if (S_i >= S_j + eps) ^ (S_i_new >= S_j_new + eps) else I


# ── Unitary computation ───────────────────────────────────────────────────────
def _get_U(rdms, i, j):
    rdm = rdms[_action_index(rdms, i, j)].copy()
    S = _preswap(rdms, i, j)
    rdm = S @ rdm @ S.conj().T
    rdm += np.finfo(rdm.dtype).eps * np.diag([0, 1, 2, 4]).astype(rdm.dtype)
    _, U = np.linalg.eigh(rdm)
    max_col = np.abs(U).argmax(axis=0)
    for k in range(4):
        U[:, k] *= np.exp(-1j * np.angle(U[max_col[k], k]))
    U = S @ U.conj().T @ S
    U = _postswap(rdms, i, j) @ U
    return U


def get_action_4q(rdms: Sequence[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """Return (U, i, j) using the ordered 4-qubit policy."""
    rdms = np.asarray(rdms)
    x = _prepare_reduced_real_input(rdms)
    a = _eval_policy(x, ORDERED_POLICY_4Q)
    i, j = _ij_from_index(a)
    U = _get_U(rdms, i, j)
    return U, i, j


# ── Entanglement ──────────────────────────────────────────────────────────────
def get_entanglements(state: np.ndarray) -> np.ndarray:
    L = int(np.log2(state.size))
    state = state.reshape((2,) * L)
    return np.array([_qubit_ent(state, i) for i in range(L)])


def _qubit_ent(state: np.ndarray, i: int) -> float:
    L = state.ndim
    perm = [i] + [j for j in range(L) if j != i]
    psi = np.transpose(state, perm).reshape(2, 2 ** (L - 1))
    lam = np.linalg.svd(psi, full_matrices=False, compute_uv=False)
    lam += np.finfo(lam.dtype).eps
    return -2.0 * np.sum((lam ** 2) * np.log(lam))


# ── Quantum circuit utilities ─────────────────────────────────────────────────
def get_full_wavefunction(qc):
    return qi.Statevector.from_instruction(qc).data


def get_exact_two_qubit_rdms(qc):
    sv = get_full_wavefunction(qc)
    indices = [[2,3],[1,3],[1,2],[0,3],[0,2],[0,1]]
    return [qi.partial_trace(sv, idx).data for idx in indices]


def get_all_two_qubit_rdms_ionq(qc, backend, shots=1024, seed=123):
    rdms = []
    for idx in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]:
        qst = StateTomography(qc.copy(), measurement_indices=idx)
        qst.analysis.set_options(fitter='cvxpy_gaussian_lstsq')
        qstdata = qst.run(backend, shots=shots)
        rdms.append(qstdata.analysis_results("state").value.data)
    return rdms


def calc_entropy_ionq(backend, qc, shots=1024):
    rdms = []
    for idx in range(4):
        qst = StateTomography(qc, measurement_indices=[idx])
        qst.analysis.set_options(fitter='cvxpy_gaussian_lstsq')
        qstdata = qst.run(backend, shots=shots)
        rdms.append(qstdata.analysis_results("state").value.data)
    ents = [qi.entropy(rdm, base=np.exp(1)) for rdm in rdms]
    return ents, rdms


def calc_full_entropy(backend, qc):
    circuit = transpile(qc, backend=backend)
    circuit.save_density_matrix(label='density_matrix')
    dm = backend.run(circuit).result().data()['density_matrix']
    return qi.entropy(dm, base=np.exp(1))


def calc_concurrence(rdms):
    return [concurrence(rdm) for rdm in rdms]


def calc_entanglement_formation(rdms):
    return [entanglement_of_formation(rdm) for rdm in rdms]


# ── Initial-state circuits ────────────────────────────────────────────────────
def bell_bell_circuit(idxs, param=None):
    qc = QuantumCircuit(4)
    qc.h(idxs[0]); qc.cx(idxs[0], idxs[1])
    qc.h(idxs[2]); qc.cx(idxs[2], idxs[3])
    if param is not None:
        for i in range(4):
            qc.u(param[0, i], param[1, i], param[2, i], i)
    return qc


def random_circuit(idxs, param):
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.u(param[0, i], param[1, i], param[2, i], i)
    qc.cx(idxs[0], idxs[1]); qc.cx(idxs[2], idxs[3]); qc.cx(idxs[1], idxs[2])
    for i in range(4):
        qc.u(param[3, i], param[4, i], param[5, i], i)
    return qc


def ghz4_circuit(idxs, param=None):
    qc = QuantumCircuit(4)
    qc.h(idxs[0])
    qc.cx(idxs[0], idxs[1]); qc.cx(idxs[1], idxs[2]); qc.cx(idxs[2], idxs[3])
    if param is not None:
        for i in range(4):
            qc.u(param[0, i], param[1, i], param[2, i], i)
    return qc


# ── Simulation ────────────────────────────────────────────────────────────────
SEEDS  = np.arange(100) + 30
EXP    = 4
LENGTH = 3


def _run_simulation(circuit_fn, out_dir: Path, seeds=SEEDS, exp=EXP, length=LENGTH):
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = AerSimulator.from_backend(FakePerth())
    shots   = 10 ** exp
    n_seeds = len(seeds)
    shape   = (n_seeds, 1, length + 1)
    shape6  = (n_seeds, 1, length + 1, 6)

    ents_agent                = np.zeros(shape)
    ents_agent_noisy          = np.zeros(shape)
    ents_universal            = np.zeros(shape)
    ents_universal_noisy      = np.zeros(shape)
    full_ents_agent_noisy     = np.zeros(shape)
    full_ents_universal_noisy = np.zeros(shape)
    conc_exact                = np.zeros(shape6)
    conc_sim                  = np.zeros(shape6)
    entform_exact             = np.zeros(shape)
    entform_sim               = np.zeros(shape)

    for m, seed in enumerate(seeds):
        print(f"seed {seed}  ({m+1}/{n_seeds})")
        np.random.seed(seed)
        # The original scripts always generated init_state here even though it
        # was never used (the qc.initialize lines were commented out). This
        # consumes 32 numpy RNG values and must be preserved so that the
        # subsequent `param` draw matches the reference data exactly.
        _ = np.random.normal(size=2**4) + 1j * np.random.normal(size=2**4)
        random.seed(int(seed))

        param  = np.random.uniform(0, 2 * np.pi, size=(6, 4))
        _param2 = np.random.uniform(0, 2 * np.pi, size=(6, 4))  # generated but unused
        idxs   = random.sample(range(4), 4)

        qc    = circuit_fn(idxs, param)
        state = get_full_wavefunction(qc)

        ent0 = np.mean(get_entanglements(state))
        ents_universal[m, 0, 0]       = ent0
        ents_agent[m, 0, 0]           = ent0

        ent0_noisy = np.mean(calc_entropy_ionq(backend, qc, shots=shots)[0])
        ents_universal_noisy[m, 0, 0] = ent0_noisy
        ents_agent_noisy[m, 0, 0]     = ent0_noisy

        full0 = calc_full_entropy(backend, qc)
        full_ents_agent_noisy[m, 0, 0]     = full0
        full_ents_universal_noisy[m, 0, 0] = full0

        for l in range(length):
            rdms       = get_all_two_qubit_rdms_ionq(qc, backend, shots=shots, seed=seed)
            rdms_exact = get_exact_two_qubit_rdms(qc)

            np.save(out_dir / f"_rdm_agent_err0_shot{exp}_{seed}_{l}.npy",       rdms)
            np.save(out_dir / f"_rdm_exact_agent_err0_shot{exp}_{seed}_{l}.npy", rdms_exact)

            U, i, j = get_action_4q(rdms)

            qc.append(UnitaryGate(U), [i, j])
            state = get_full_wavefunction(qc)

            ents_agent[m, 0, l+1]           = np.mean(get_entanglements(state))
            ents_agent_noisy[m, 0, l+1]     = np.mean(calc_entropy_ionq(backend, qc, shots=shots)[0])
            full_ents_agent_noisy[m, 0, l+1] = calc_full_entropy(backend, qc)

            conc_exact[m, 0, l, :]  = calc_concurrence(rdms_exact)
            conc_sim[m, 0, l, :]    = calc_concurrence(rdms)
            entform_exact[m, 0, l]  = np.mean(calc_entanglement_formation(rdms_exact))
            entform_sim[m, 0, l]    = np.mean(calc_entanglement_formation(rdms))

        rdms       = get_all_two_qubit_rdms_ionq(qc, backend, shots=shots, seed=seed)
        rdms_exact = get_exact_two_qubit_rdms(qc)

        np.save(out_dir / f"_rdm_agent_err0_shot{exp}_{seed}_{length}.npy",       rdms)
        np.save(out_dir / f"_rdm_exact_agent_err0_shot{exp}_{seed}_{length}.npy", rdms_exact)

        conc_exact[m, 0, length, :] = calc_concurrence(rdms_exact)
        conc_sim[m, 0, length, :]   = calc_concurrence(rdms)
        entform_exact[m, 0, length] = np.mean(calc_entanglement_formation(rdms_exact))
        entform_sim[m, 0, length]   = np.mean(calc_entanglement_formation(rdms))

        for fname, arr in [
            (f"__2ents_agent_{exp}_{seed}.npy",                ents_agent),
            (f"__2ents_agent_noisy_{exp}_{seed}.npy",          ents_agent_noisy),
            (f"__2ents_universal_{exp}_{seed}.npy",            ents_universal),
            (f"__2ents_universal_noisy_{exp}_{seed}.npy",      ents_universal_noisy),
            (f"__2full_ents_agent_noisy_{exp}_{seed}.npy",     full_ents_agent_noisy),
            (f"__2full_ents_universal_noisy_{exp}_{seed}.npy", full_ents_universal_noisy),
            (f"__2conc_exact_{exp}_{seed}.npy",                conc_exact),
            (f"__2conc_sim_{exp}_{seed}.npy",                  conc_sim),
            (f"__2entform_exact_{exp}_{seed}.npy",             entform_exact),
            (f"__2entform_sim_{exp}_{seed}.npy",               entform_sim),
        ]:
            np.save(out_dir / fname, arr)


# ── Per-circuit entry points ──────────────────────────────────────────────────
def run_bell_bell(out_dir: Path):
    _run_simulation(bell_bell_circuit, out_dir)


def run_shallow_random(out_dir: Path):
    _run_simulation(random_circuit, out_dir)


def run_ghz(out_dir: Path):
    _run_simulation(ghz4_circuit, out_dir)


if __name__ == "__main__":
    run_bell_bell(PATH_RAW / "bell_bell")
    run_shallow_random(PATH_RAW / "shallow_random")
    # run_ghz(PATH_RAW / "ghz")  # computed but not used in the figure
