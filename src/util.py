"""
Miscelaneous utility functions:
    - str2state
    - str2latex
    - rollout
    - peek_policy
"""
import itertools
import numpy as np
import torch
from scipy.linalg import sqrtm

from .agent import PGAgent
from .quantum_env import QuantumEnv
from .quantum_state import random_quantum_state


def str2state(string_descr):
    """Generates Haar random state from string description like 'R-RRR-R'."""
    psi = np.array([1.0], dtype=np.complex64)
    bell = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).astype(np.complex64)
    W = np.array([0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0]).astype(np.complex64)
    zero = np.array([1, 0]).astype(np.complex64)
    one = np.array([0, 1]).astype(np.complex64)
    nqubits = 0
    chars = "abcdefghijklmn"

    for pair in string_descr.split('-'):
        if pair == "B":
            phi = bell.reshape(2,2)
            q = 2
        elif pair == "1":
            phi = one
            q = 1
        elif pair == "0":
            phi = zero
            q = 1
        elif pair == "W":
            phi = W.reshape(2,2,2)
            q = 3
        elif set(pair) == {'R'}:
            q = pair.count('R')
            phi = random_quantum_state(q=q, prob=1.)
        elif pair == '':
            continue
        else:
            raise ValueError(f"Unknown quantum subsystem coding: \"{pair}\"")
        if nqubits == 0:
            psi = phi
        else:
            A = chars[:nqubits]
            B = chars[nqubits:nqubits+q]
            einstr = f"{A},{B}->{A}{B}"
            psi = np.einsum(einstr, psi, phi)
        nqubits += q
    return psi.reshape((2,) * nqubits)


def str2latex(string_descr):
    """Transforms state name like "R-RR" to "|R>|RR>" in Latex."""
    numbers = "123456789"
    codes = string_descr.split("-")
    name = []
    i = 0
    for c in codes:
        if c == "W":
            q = 3
            letter = "W"
        elif c == "1":
            q = 1
            letter = "1"
        elif c == "0":
            q = 1
            letter = "0"
        elif c == "B":
            q = 2
            letter = "Bell"
        elif set(c) == {"R",}:
            q = len(c)
            letter = "R"
        else:
            raise ValueError(f"Unknown quantum subsystem coding: \"{c}\"")
        s = f'|{letter}_{{' + f"{numbers[i:i+q]}" + r"}\rangle"
        # USE FOR "slighly entangled states, Fig. 14"
        # s = r'R_{' + f'{numbers[i:i+len(r)]}' + '}'
        name.append(r'\mathrm{' + s + '}')
        i += q
    return "$" + ''.join(name) + "$"
    # USE FOR "slighty entangled states, Fig. 14"
    # return "$|" + ''.join(name) + r"\rangle$"



def rollout(qstate: np.ndarray, agent: PGAgent, max_steps: int = 30):
    """
    Performs a rollout with `agent` for `qstate`.

    Returns action names, entanglements and policy probabilities for each step.
    """

    # Initialize environment
    num_qubits = int(np.log2(qstate.size))
    shape = (2,) * num_qubits
    qstate = qstate.reshape(shape)
    env = QuantumEnv(num_qubits, 1, obs_fn='rdm_2q_mean_real')
    env.reset()
    env.simulator.states = np.expand_dims(qstate, 0)

    # Set agent to `eval` mode
    agent.policy_network.eval()

    # Rollout a trajectory
    actions, entanglements, probabilities = [], [], []
    for _ in range(max_steps):
        ent = env.simulator.entanglements.copy()
        observation = torch.from_numpy(env.obs_fn(env.simulator.states))
        probs = agent.policy(observation).probs[0].cpu().numpy()
        a = np.argmax(probs)
        actions.append(env.simulator.actions[a])
        entanglements.append(ent.ravel())
        probabilities.append(probs)
        o, r, t, tr, i = env.step([a], reset=False)
        if np.all(t):
            break
    # Append final entangments
    # assert np.all(env.simulator.entanglements <= env.epsi)
    entanglements.append(env.simulator.entanglements.copy().ravel())

    return np.array(actions), np.array(entanglements), np.array(probabilities)


def peek_policy(state):
    """Returns the agent probabilites for this state."""
    _, _, probabilities = rollout(state, max_steps=1)
    return probabilities[0]




def ent_of_formation(rho_2):
    """
    the quantum entanglement between two qubits in a mixed state differs from
    the von Neuman entropy (the latter includes also any statistical
    uncertainty in the mixed state):

    https://en.wikipedia.org/wiki/Entanglement_of_formation

    for pure states, the entanglement of formation reduces to the
    von Neuman entropy.
    """

    # compute concurrence
    C = concurrence(rho_2)
    x = 0.5*(1.0+np.sqrt(1.0-C**2))
    eps = np.finfo(rho_2.dtype).eps

    # compute entanglement of formation
    Sent_form = -( x*np.log(x+eps) + (1.0-x)*np.log(1-x+eps) )
    return max(0.0, Sent_form)


def concurrence(rho_2):
    """
    computes the concurrence of a 2-qubit density matrix:

    https://en.wikipedia.org/wiki/Concurrence_(quantum_computing)

    the concurrence is a measure of the quantum entanglement between the qubits,
    which differs from the von Neuman entropy (the latter includes also any
    statistical uncertainty in the mixed state)
    """

    Y = np.array([[0,-1j],[1j,0]])
    YY = np.einsum('ij,kl->ik jl',Y,Y).reshape(4,4)
    # prevent taking sqrt(-eps) and log(0-eps)
    regularizer = 1E-3*np.finfo(rho_2.dtype).eps * np.eye(4)

    rho_tilde = YY @ np.conj(rho_2) @ YY
    sqrt_rho = sqrtm(rho_2 + regularizer)
    R2 = sqrt_rho @ rho_tilde @ sqrt_rho
    R = sqrtm(R2 + regularizer)
    lmbda = np.linalg.eigvalsh(R)
    C = np.max([0.0,lmbda[3] - lmbda[0:3].sum() ])

    return C


def entfor_matrix(rhos, batch_dim=True, half=True):

    def _entfor_matrix(rhos, L, indices):
        entanglements_f = {}
        for i,j in indices:
            if i > j:
                continue
            k = indices[(i,j)]
            entanglements_f[(i,j)] = ent_of_formation(rhos[k].reshape(4,4))

        matrix = np.zeros((L,L), dtype=np.float32)
        for i,j in itertools.product(range(L), range(L)):
            if i == j:
                matrix[i,j] = 0.0
            elif i < j:
                matrix[i,j] = entanglements_f[(i,j)]
            else:
                matrix[i,j] = entanglements_f[(j,i)]
        return matrix


    n = rhos.shape[1] if batch_dim else rhos.shape[0]
    n = 2 * n if half else n
    L = 1 + int(np.sqrt(n))

    perm = list(itertools.permutations(range(L), 2))
    comb = list(itertools.combinations(range(L), 2))
    indices = {pair: i for i, pair in enumerate(comb if half else perm)}

    # Calculate entanglement of formations
    if batch_dim:
        matrix = np.zeros((rhos.shape[0], L, L), dtype=np.float32)
        for i in range(len(rhos)):
            matrix[i] = _entfor_matrix(rhos[i], L, indices)
        return matrix
    else:
        return _entfor_matrix(rhos, L, indices)
