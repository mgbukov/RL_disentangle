"""
Processing script for FIG. 19 (circuit diagrams).

Each circuit uses the same 4q RL agent applied to a randomly-rotated initial
state.  The initial states are reconstructed from the saved idxs / param files
produced during the original hardware experiments (stored in data/raw/fig19/).
The policy probabilities come from the dist_*.npy files recorded during those
runs (computed from hardware-measured RDMs fed through the RL agent), so the
circuits faithfully reflect what was actually done on each device.

Row mapping (top to bottom in Fig. 19):
    Row 1  IonQ Harmony    |Bell_12>|Bell_34>  (seed 13)
    Row 2  ibm_nairobi     |Bell_12>|Bell_34>  (seed 14)
    Row 3  ibm_nairobi     |0>|GHZ_234>        (seed  8)
    Row 4  ibm_perth       |0>|GHZ_234>        (seed  5)

Initial-state construction (without Qiskit):
  bell_bell: H(idxs[0]) CX(idxs[0],idxs[1]) H(idxs[2]) CX(idxs[2],idxs[3])
             then per-qubit U3(param[0,i], param[1,i], param[2,i]) for i=0..3
  ghz:       H(idxs[0]) CX(idxs[0],idxs[1]) CX(idxs[0],idxs[2])
             then per-qubit U3(...)
  → ghz_circuit leaves idxs[3] in |0>, producing |0>|GHZ3> up to local unitaries

Run from src_code/scripts/processed/:
    python fig19_nisq_hardware.py
"""
import json
from pathlib import Path

import numpy as np



ROOT = Path(__file__).parent.parent.parent.parent
PATH_RAW       = ROOT / "data" / "raw" / "fig19"
PATH_PROCESSED = ROOT / "data" / "processed" / "fig19"
ACTSET = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
N_STEPS = 2


# ── Numpy-based state preparation (matches the Qiskit circuits used on HW) ──

def _u3(theta, phi, lam):
    return np.array([
        [np.cos(theta / 2),                   -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
    ], dtype=np.complex64)


def _apply1(state, gate, k):
    """Apply a 1-qubit gate to qubit k of a 4-qubit state (2,2,2,2)."""
    return np.moveaxis(np.tensordot(gate, state, axes=([1], [k])), 0, k)


def _cnot(state, c, t):
    """Apply CNOT(control=c, target=t) to a 4-qubit state (2,2,2,2)."""
    s = state.copy()
    idx = [slice(None)] * 4
    idx[c] = 1
    sub = s[tuple(idx)]
    # target index within the (c=1)-subspace: shift if t > c
    t_in_sub = t if t < c else t - 1
    s[tuple(idx)] = np.flip(sub, axis=t_in_sub)
    return s


H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)


def build_bell_bell(idxs, param):
    """H(i0) CX(i0,i1) H(i2) CX(i2,i3) then per-qubit U3."""
    state = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    state[0, 0, 0, 0] = 1.0
    state = _apply1(state, H, idxs[0])
    state = _cnot(state, idxs[0], idxs[1])
    state = _apply1(state, H, idxs[2])
    state = _cnot(state, idxs[2], idxs[3])
    for i in range(4):
        state = _apply1(state, _u3(param[0, i], param[1, i], param[2, i]), i)
    return state


def build_ghz(idxs, param):
    """H(i0) CX(i0,i1) CX(i0,i2) then per-qubit U3.
    Leaves qubit idxs[3] in |0>, producing |0>|GHZ3> up to local unitaries."""
    state = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    state[0, 0, 0, 0] = 1.0
    state = _apply1(state, H, idxs[0])
    state = _cnot(state, idxs[0], idxs[1])
    state = _cnot(state, idxs[0], idxs[2])
    for i in range(4):
        state = _apply1(state, _u3(param[0, i], param[1, i], param[2, i]), i)
    return state


# ── Per-experiment configuration ─────────────────────────────────────────────

EXPERIMENTS = [
    # (hw_subdir,                 idxs_file,   param_file,  state_fn,      out_json)
    ("ionq/bell_bell/seed13",     "_idxs.npy", "param.npy", build_bell_bell, "ionq-bell_bell-circuit.json"),
    ("ibmq_nairobi/bell_bell/seed14", "_idxs.npy", "_param.npy", build_bell_bell, "ibmq_nairobi-bell_bell-circuit.json"),
    ("ibmq_nairobi/ghz/seed8",   "_idxs.npy", "_param.npy", build_ghz,      "ibmq_nairobi-ghz-circuit.json"),
    ("ibmq/ghz/seed5",           "_idxs.npy", "_param.npy", build_ghz,      "ibmq-ghz-circuit.json"),
]


if __name__ == "__main__":

    PATH_RAW.mkdir(parents=True, exist_ok=True)
    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)

    for hw_subdir, idxs_file, param_file, state_fn, out_json in EXPERIMENTS:
        hw_path = PATH_RAW / hw_subdir

        # Reconstruct initial state from saved hardware-experiment parameters
        idxs  = np.load(hw_path / idxs_file).astype(int)
        param = np.load(hw_path / param_file)
        state = state_fn(idxs, param)

        # Save initial state .npy for reference
        npy_name = out_json.replace("-circuit.json", ".npy")
        np.save(str(PATH_RAW / npy_name), state)

        # Build circuit using the actual hardware policy distributions (dist_*.npy)
        # and actual gate pairs (idxs_agent_*.npy) recorded during the HW run.
        # This faithfully captures the circuit that was executed on each device.
        circuit = []
        for step in range(N_STEPS):
            gate  = np.load(hw_path / f"idxs_agent_{step}.npy").astype(int).tolist()
            probs = np.load(hw_path / f"dist_{step}.npy").flatten().tolist()
            circuit.append({"qubits": gate, "probabilities": probs, "actions": ACTSET})

        with open(PATH_PROCESSED / out_json, mode="wt") as f:
            json.dump(circuit, f, indent=2)
        print(f"Saved {out_json}  ({len(circuit)} steps, gates={[s['qubits'] for s in circuit]})")
