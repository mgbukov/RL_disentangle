import numpy as np
from qiskit.quantum_info import state_fidelity, partial_trace, Statevector, entropy
from qiskit_experiments.library import StateTomography
from .helpers import get_action

def get_rdms_via_tomography(data, qc, n_shots):
    """
    Perform quantum state tomography (QST) on given data and quantum circuit (circ) to obtain reduced density matrices (RDMs).

    Parameters:
    data (numpy.ndarray): The input data to be processed, expected to be a 3D array.
    circ (QuantumCircuit): The quantum circuit on which QST is performed.
    n_shots (int): The number of shots to be used in the tomography.

    Returns:
    rdms (list): List of reduced density matrices obtained from the input data.
    """
    data = (np.rint(data * n_shots)).astype(int)
    rdms = []
    indices = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for i, idx in enumerate(indices):
        # print("Qubit indices on which to perform QST: ", idx)
        res = data[:, i, :].copy()

        res = res[[8,6,7,2,0,1,5,3,4],:] # permute circuit index to match qiskit's ordering
        # Matthew: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
        # qiskit : ZZ, ZX, ZY, XZ, XX, XY, YZ, YX, YY

        outcome_data = res.reshape((1, 9, 4))

        shot_data = np.sum(outcome_data, axis=2).reshape((9,))

        qst = StateTomography(qc, measurement_indices=idx)
        qst.analysis.set_options(fitter='cvxpy_gaussian_lstsq')

        # [print(c) for c in qst.circuits()]

        # State tomography on Matthew's data
        fitter = qst.analysis._get_fitter(qst.analysis.options.fitter)
        measurement_data = np.array([[i, j] for i in range(3) for j in range(3)])
        preparation_data = np.array([]).reshape((9, 0))
        fitter_kwargs = {
            "measurement_qubits": idx,
            "measurement_basis": qst.analysis.options.measurement_basis
        }
        rdms.append(qst.analysis._fit_state_results(fitter, outcome_data, shot_data, measurement_data, preparation_data, False, **fitter_kwargs)[0].value.data)
    return rdms

def get_next_unitary(data, qc, n_shots):
    """
    Generate the next unitary obtained by the RL agent based on the given data and quantum circuit.

    Args:
        data (dict): The input data required for the tomography process.
        qc (QuantumCircuit): The quantum circuit to be used.
        n_shots (int): The number of shots for the tomography process.

    Returns:
        tuple: A tuple containing the next unitary operation (U) and the list of qubit indices [i, j] where the unitary acts on.
    """
    rdms = get_rdms_via_tomography(data.copy(), qc.copy(), n_shots)
    U, i, j = get_action(rdms, policy='transformer')
    return U, [i, j]

def get_exact_two_qubit_rdms(qc):
    """
    Compute the exact (noise-free) reduced density matrices (RDMs) for all pairs of qubits in a given quantum circuit.

    Parameters:
    qc (QuantumCircuit): A quantum circuit from which the statevector is obtained.

    Returns:
    list: A list of reduced density matrices (RDMs) for each pair of qubits.
    """
    st_vector = Statevector.from_instruction(qc).data
    indices = [[2,3],[1,3],[1,2],[0,3],[0,2],[0,1]] # qubit pairs to trace out
    return [partial_trace(st_vector, idxs).data for idxs in indices]

def compute_fidelities(rdms1, rdms2):
    """
    Compute the average state fidelity and individual fidelities between two sets of reduced density matrices (RDMs).

    Parameters:
    rdms1 (list): A list of reduced density matrices (RDMs).
    rdms2 (list): A list of reduced density matrices (RDMs) to compare against rdms1.

    Returns:
    tuple: A tuple containing:
        - float: The mean fidelity between the corresponding RDMs in rdms1 and rdms2.
        - list: A list of individual fidelities for each pair of RDMs.
    """
    # SWAP_gate = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) # in case qubit order is inverted for some reason
    # rdms3 = [np.dot(SWAP_gate, np.dot(rdm, SWAP_gate)) for rdm in rdms2]
    fidelities = [state_fidelity(rdms1[i], rdms2[i]) for i in range(len(rdms1))]
    return np.mean(fidelities), fidelities

def compute_entanglement_entropy(rdms2):
    """
    Compute the average single-qubit entanglement entropy for a given set of reduced density matrices (RDMs).

    Parameters:
    rdms (list): A list of reduced density matrices (RDMs).

    Returns:
    list: The entanglement entropies of each qubit.
    """
    # indices = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    trace_idx = [[1,1,1], [0,1,1], [0,0,1], [0,0,0]]
    rdm_idx = [[0,1,2], [0,3,4], [1,3,5], [2,4,5]]
    entss = []
    for i, p in enumerate(rdm_idx):
        ents = []
        for j, pp in enumerate(p):
            rdm = partial_trace(rdms2[pp],[trace_idx[i][j]]).data
            ent = entropy(rdm, base=np.exp(1))
            ents.append(ent)
        entss.append(np.mean(ents))
    return entss