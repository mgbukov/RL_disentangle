from helpers import *
import cvxpy
import copy
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit, transpile
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography
from qiskit.extensions import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakePerth
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator

from qiskit_ionq import IonQProvider

def get_all_two_qubit_rdms(qc, shots=1024, seed=123):
    rdms = []
    indices = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
    for i,idx in enumerate(indices):
        # print()
        # print(idx)
        qst = StateTomography(qc, measurement_indices=idx, target='default')
        # [c.draw(output='mpl', filename=f"circ_{j}.png") for j,c in enumerate(qst.circuits())]
        # qst.analysis.set_options(fitter='cvxpy_gaussian_lstsq')
        qstdata = qst.run(backend, shots=shots, seed_simulation=seed).block_for_results()
        state_result = qstdata.analysis_results("state")
        rdms.append(state_result.value.data)
        fid_result = qstdata.analysis_results("state_fidelity")
        # print("State Fidelity = {:.5f}".format(fid_result.value))
    return rdms

def get_full_wavefunction(qc):
    backend2 = StatevectorSimulator()
    circuit = transpile(qc, backend=backend2)
    job = backend2.run(circuit)
    result = job.result()
    return result.get_statevector(circuit, decimals=16).data


seeds = np.arange(20) + 30
print(seeds)

shotss = [2**i for i in range(10,18)]#[:3]
print(shotss)

provider = IonQProvider()
backend3 = provider.get_backend("ionq_simulator")
backend2 = AerSimulator.from_backend(FakePerth())
backend = Aer.get_backend('qasm_simulator')
n = 4 # number of qubits


av_ents = np.zeros((len(shotss), len(seeds)))
for k, shots in enumerate(shotss):
    for l, seed in enumerate(seeds):
        print()
        print(k, l)
        np.random.seed(seed)

        init_state = np.random.normal(size=2**n) + 1j * np.random.normal(size=2**n)
        init_state /= np.linalg.norm(init_state)

        qc = QuantumCircuit(n)
        qc.initialize(init_state)

        # ################# plot transpiled initial state
        # circuit = transpile(qc, backend=backend2)
        # circuit.draw(output='mpl', filename=f"6transpiled_initial_perth{l}.png")
        # # print(dict(circuit.count_ops()))
        # print(dict(circuit.count_ops())['cx'])
        # # print(dict(circuit.count_ops())['cz'])

        state = init_state
        for step in range(5):
            rdms = get_all_two_qubit_rdms(qc, shots=shots, seed=seed)
            U, i, j = get_action_4q(rdms)
            # print(i,j)

            true_state, _, _ = peek_next_4q(copy.deepcopy(state), U, i, j)
            # print(true_state)

            qc.append(UnitaryGate(U), [i, j])
            state = get_full_wavefunction(qc)
            # print()
            # print(state)

            # print()
            ents = entanglement(state)
            # print(ents)
            # print(state.dot(np.conj(true_state)))
            # print()

        print(ents)
        av_ents[k,l] = np.mean(ents)

        # ######################## plot transpiled disentagler circuits
        # del qc.data[0]
        # qc.draw(output='mpl', filename=f"original_circ.png")
        # circuit = transpile(qc, backend=backend3)
        # circuit.draw(output='mpl', filename=f"transpiled_circ_ionq.png")
        # print(dict(circuit.count_ops()))
        # # print(dict(circuit.count_ops())['cx'])
        # print(dict(circuit.count_ops())['cz'])



print(av_ents)
np.save("av_ents_linear_inversion.npy", av_ents)

av_ents = np.load("av_ents.npy")
mean_ents = np.mean(av_ents, axis=1)
std_ents = np.std(av_ents, axis=1)


plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('legend', handlelength=2)
plt.rc('font', size=14)

linestyles = ["-", "--", "dotted"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*"]


plt.plot(shotss, mean_ents, marker='o', c='blue')#, marker=markers[i], ls=linestyles[k], c=colors[i], label=f"depth {depth}", ms=3)
# plt.errorbar(shotss, mean_ents, yerr=std_ents, fmt='o', c='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"shots")
plt.ylabel(r"$\langle N^{-1} \sum S \rangle$")
# plt.legend()
plt.tight_layout()
plt.savefig(f"_mean_ents.png", dpi=300)
plt.close()

plt.plot(shotss, std_ents, marker='s', c='green')#, marker=markers[i], ls=linestyles[k], c=colors[i], label=f"depth {depth}", ms=3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"shots")
plt.ylabel(r"$\langle \bar S^2 \rangle - \langle \bar S \rangle^2$")
# plt.legend()
plt.tight_layout()
plt.savefig(f"_std_ents.png", dpi=300)
plt.close()
