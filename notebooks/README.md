# Notebooks

Run all notebooks from the **`src_code/notebooks/`** directory.  `context.py` (imported at the top of each notebook) automatically adds `src_code/` to `sys.path` so that `from src.*` imports resolve correctly.

## Environment setup

| File | Platform | How to create |
|------|----------|--------------|
| `environment-macos.yaml` | macOS / Linux | `conda env create -f src_code/environment-macos.yaml` |
| `environment-win64.yml` | Windows x86-64 | `conda env create -f src_code/environment-win64.yml` |

The Windows file is a fully-pinned export with all packages installed via `pip` to avoid the MKL DLL conflict that arises when conda-forge numpy and a pip-installed PyTorch are loaded in the same process.

After creating the environment, launch JupyterLab from `src_code/notebooks/`:

```bash
conda activate rl          # or the env name you chose
jupyter lab
```

---

## Notebooks

### `interactive-demo.ipynb`

**Interactive demo of the trained 4- and 5-qubit RL agents.**

Requires the pre-trained models at `models/4q-agent.pt` and `models/5q-agent.pt` (repo root).

The demo lets you:
- Choose an initial quantum state from a dropdown (Bell pairs, GHZ, W, Haar-random, …) or supply your own amplitudes.
- Step through the agent's disentangling protocol gate by gate.
- Apply arbitrary single- and two-qubit rotation gates to modify the state mid-episode.
- Inspect the policy probabilities π(a|o), the expected entanglement reduction ΔS_avg for every action, and (for the 4-qubit agent) the transformer attention scores.

The demo UI is built with `ipympl` (`%matplotlib widget`) and `ipywidgets`; both must be installed and the notebook must be run inside JupyterLab or the classic Notebook interface (not VS Code's built-in renderer).

---

### `run_agents.ipynb`

**Generates the rollout dataset used by the test suite.**

Loads trained agents and runs them on Haar-random fully-entangled states (4q, 5q, 6q) and on η-entangled product states (10q, 12q) until disentanglement succeeds.  Successfully solved episodes are collected and written to:

```
data/raw/tests/rollouts.pickle
```

This file is consumed by `tests/test_qenv.py::test_disentangle`.  Re-run this notebook whenever the agents are updated or the rollout file is missing.

---

### `eta-study.ipynb`

**Residual entanglement vs. the inter-subsystem coupling parameter η (12q and 16q).**

For several η-entangled product-state families (6×|RR⟩, 4×|RRR⟩, 3×|RRRR⟩ for 12 qubits; and 8×|RR⟩, 5×|RRR⟩+|R⟩, 4×|RRRR⟩ for 16 qubits) the notebook:
1. Samples 25 states per η value across η ∈ [3.0, 6.0].
2. Applies a fixed "optimal" disentangling trajectory and measures residual single-qubit entanglement entropies.
3. Rolls out the trained RL agent and measures the same quantities.
4. Plots per-qubit and envelope scatter plots, saved as PDFs under a `figures/eta-entanglements/` directory.

**Note:** the notebook hard-codes `DEVICE = "mps"` (Apple Silicon GPU) and loads agents from paths under `../logs/` that are not part of the Zenodo record.  Adjust `DEVICE` and the agent paths before running.  Estimated runtimes are annotated in the cells (up to ~100 min for 16-qubit rollouts).

---

### `stategen.ipynb`

**Exploration of the `sample_haar_generalized` state generator and η-dependent disentanglement difficulty.**

Two parts:

**Part 1 — Bond entanglement vs. η.**  Plots how the inter-subsystem bond entanglement scales with η for systems of size L = 6–15 and subsystem sizes 4–5.  Fits an exponential decay and overlays the ε = 10⁻² threshold.

**Part 2 — Expected disentanglement path lengths.**  For 12-qubit 4×|RRRR⟩ states at a range of η values, measures how many random-search steps are needed to reach ε = 10⁻² and compares against beam-search path lengths.  Results are printed in-cell; no files are written.
