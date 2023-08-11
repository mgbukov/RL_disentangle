from itertools import combinations, permutations
from collections import namedtuple
import sys
import numpy as np

from src.quantum_state import VectorQuantumState


observation_space = namedtuple("observation_space", ["shape"])
action_space = namedtuple("action_space", ["n"])


class QuantumEnv():
    """QuantumEnv is a wrapper around VectorQuantumState conforming to the
    OpenAI Gym API.
    """

    def __init__(self, num_qubits, num_envs, epsi=1e-3, p_gen=0.95, max_episode_steps=1000,
        reward_fn="sparse", obs_fn="phase_norm"
    ):
        """Init a Quantum environment.

        Args:
            num_qubits: int
                Number of qubits in a quantum state.
            num_envs: int
                Number of quantum states for the vectorized environment.
            epsi: float, optional
                Threshold for disentangling a quantum state. Default: 1e-3.
            p_gen: float, optional
                Probability for drawing the state from the full Hilbert space,
                i.e. all the qubits are entangled. (prob \in (0, 1]). Default 0.95.
            max_episode_steps: int, optional
                Maximum number of steps before truncating the environment. Default: 1000.
            reward_fn: string, optional
                The name of the reward function to be used. One of
                ["sparse", "relative_delta"]. Default: "sparse".
            obs_fn: string, optional
                The name of the observation function to be used. One of
                ["phase_norm", "rdm_1q", "rdm_2q_complex", "rdm_2q_real"]. Default: "phase_norm".
        """
        # Private.
        self.epsi = epsi
        self.max_episode_steps = max_episode_steps
        act_space = "reduced" if obs_fn == "rdm_2q_mean_real" else "full"
        self.simulator = VectorQuantumState(num_qubits, num_envs, p_gen, act_space)
        self.reward_fn = getattr(sys.modules[__name__], reward_fn)  # get from this module
        self.obs_fn = getattr(sys.modules[__name__], obs_fn)        # get from this module
        self.obs_dtype = self.obs_fn(self.simulator.states).dtype

        # Public attributes conforming to the OpenAI Gym API.
        self.num_envs = num_envs

        # Define the observation space.
        self.single_observation_space = observation_space(
            shape=self.obs_fn(self.simulator.states).shape[1:])

        # Define the action space.
        self.single_action_space = action_space(n=self.simulator.num_actions)

    def reset(self, seed=0):
        """Reset the environment to its initial state."""
        for k in range(self.num_envs):
            self.simulator.reset_sub_environment_(k)
        self.episode_len = np.zeros(shape=(self.simulator.num_envs,))
        self.accumulated_return = np.zeros(shape=(self.simulator.num_envs,))
        return self.obs_fn(self.simulator.states), {}

    def close(self):
        pass

    def step(self, acts, reset=True):
        """Step the environment using the provided actions.

        Args:
            acts: np.Array
                Numpy array of shape (N,), giving the actions to be applied
                to each of the quantum states respectively.

        Returns:
            obs: np.Array
                Numpy array of shape (N, *obs_shape), giving the new observations.
            rewards: np.Array
                Numpy array of shape (N,), giving the received rewards.
            done: np.Array
                Boolean array of shape (N,), indicating which of the states are done.
            info: dict
                Dict containing info about done states.
        """
        # Store the current entanglements before applying the actions.
        prev_entanglements = self.simulator.entanglements.copy()

        # Apply the actions on the simulator.
        self.simulator.apply(acts)

        # Check if an environment was truncated or terminated.
        # It is perfectly fine to return both as true.
        self.episode_len += 1
        truncated = (self.episode_len >= self.max_episode_steps)
        terminated = np.all(self.simulator.entanglements <= self.epsi, axis=1)

        # Get the rewards from the current system states and compute the returns.
        rewards = self.reward_fn(
            self.simulator.entanglements, prev_entanglements, self.epsi)
        self.accumulated_return += rewards

        # If any of the sub-environments are done, then fill in the info dict
        # and reset them.
        info = {}
        done = (terminated | truncated)
        if done.any():
            # Adhere to the OpenAI gym environment API.
            # d = np.expand_dims(done, axis=np.arange(len(self.single_observation_space.shape)))
            # d = done.reshape(self.num_envs, *(1,) * len(self._observations().shape[1:]))
            info = {
                # "final_observation": np.where(d, self._observations(), None),
                "episode": {
                    "r": np.where(done, self.accumulated_return, None),
                    "l": np.where(done, self.episode_len, None),
                },
            }

            # Reset only the sub-environments that were done.
            for k in range(self.simulator.num_envs):
                if not done[k] or not reset: continue
                self.simulator.reset_sub_environment_(k)
                self.accumulated_return[k] = 0.
                self.episode_len[k] = 0

        # Finally, get the observations from the next obtained states.
        # Note that we have to do this only after we check for done environments.
        obs = self.obs_fn(self.simulator.states)

        # NOTE: Take notice that if a sub-environment is done we are resetting
        # it and returning the observation for the new state. This means that
        # the SARS tuple for this transition will be: (s_T, a_T, r_T, s_0).
        # For online learning we compute the value of s_T as:
        #   V(s_T) = r_T + V(s_{T+1}),
        # but now we don't have s_{T+1}. Nevertheless, this is perfectly fine
        # because we want to set V(s_{T+1})=0 anyway as the base case. Thus, for
        # final states we simply have: V(s_T) = r_T.
        #
        # There might be a problem with truncated environments, because
        # we would like to see V(s_{T+1}), since it is not 0. For this reason
        # the observation s_{T+1} is returned in the info dict.
        return obs, rewards, terminated, truncated, info


#--------------------------- Observation functions ----------------------------#
def phase_norm(states):
    N = states.shape[0]
    batch = states.reshape(N, -1)
    batch = np.hstack([batch.real, batch.imag])
    return batch

def rdm_1q(states):
    """Returns an observation of the states of the vector environment.
    The observation for a single quantum system is arrived at by calculating
    all single qubit rdm for each and every qubit of the system separately.

        Returns:
            obs: np.Array
                Numpy array of shape (N, Q, 8), giving the observations for the
                current quantum systems, where N = number of environments,
                and Q = number of qubits.
        """
    N = states.shape[0]
    Q = len(states.shape[1:])
    rdms = []
    for q in range(Q):
        sysA = (q+1,)
        sysB = tuple(p+1 for p in range(Q) if p != q)
        permutation = (0,) + sysA + sysB

        psi = np.transpose(states, permutation).reshape(N, 2, -1)
        rdm = psi @ np.transpose(psi, (0, 2, 1)).conj()
        rdms.append(rdm)
    rdms = np.array(rdms)               # rdms.shape == (Q, B, 2, 2)

    # The single qubit rmd is a complex matrix of shape (2, 2). The matrix is
    # flattened and the real and imaginary parts of each number are stacked.
    rdms = rdms.transpose((1, 0, 2, 3)).reshape(N, -1, 4)
    obs = np.dstack([rdms.real, rdms.imag])
    return obs

def rdm_2q_complex(states):
    """
    Returns 2-qubit RDM observations with complex64 dtype.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, Q*(Q-1), 16), where N = number of episodes,
            Q = number of qubits
    """
    N = states.shape[0]
    Q = len(states.shape[1:])
    rdms = []
    qubit_pairs = permutations(range(Q), 2)

    for qubits in qubit_pairs:
        sysA = tuple(q+1 for q in qubits)
        sysB = tuple(q+1 for q in range(Q) if q not in qubits)
        permutation = (0,) + sysA + sysB
        psi = np.transpose(states, permutation).reshape(N, 4, -1)
        rdm = psi @ np.transpose(psi, (0, 2, 1)).conj()
        rdms.append(rdm)
    rdms = np.array(rdms)                # rdms.shape == (Q*(Q-1), N, 4, 4)
    rdms = rdms.transpose((1, 0, 2, 3))  # rdms.shape == (N, Q*(Q-1), 4, 4)
    obs = rdms.reshape(N, Q*(Q - 1), 16) # obs.shape  == (N, Q*(Q-1), 16)
    return obs

def rdm_2q_real(states):
    """
    Returns 2-qubit RDM observations with float32 dtype.

    Returns:
        obs: np.ndarray, dtype=np.float32
            Numpy tensor with shape (N, Q*(Q-1), 32), where N = number of episodes,
            Q = number of qubits.
    """
    rdms = rdm_2q_complex(states)           # rdms.shape = (N, Q*(Q-1), 16)
    return np.dstack([rdms.real, rdms.imag])

def rdm_2q_mean_complex(states):
    """Returns 2-qubit RDM observations with complex64 dtype.
    The rdms resulting from the two different combinations of qubits (i,j) and
    (j, i) are averaged.

    Returns:
        obs: np.ndarray, dtype=np.complex64
            Numpy tensor with shape (N, Q*(Q-1), 16), where N = number of episodes,
            Q = number of qubits
    """
    N = states.shape[0]
    Q = len(states.shape[1:])
    rdms = []
    qubit_pairs = combinations(range(Q), 2)

    for qubits in qubit_pairs:
        sysA = tuple(q+1 for q in qubits)
        sysB = tuple(q+1 for q in range(Q) if q not in qubits)

        # qubit pair (i, j)
        permutation = (0,) + sysA + sysB
        psi = np.transpose(states, permutation).reshape(N, 4, -1)
        rdm = psi @ np.transpose(psi, (0, 2, 1)).conj() # rdm.shape = (N, 4, 4)
        rdm = rdm.reshape(N, 16)

        # qubit pair (j, i)
        permutation_rev = (0,) + tuple(reversed(sysA)) + sysB
        psi = np.transpose(states, permutation_rev).reshape(N, 4, -1)
        rdm_rev = psi @ np.transpose(psi, (0, 2, 1)).conj() # rdm_rev.shape = (N, 4, 4)
        rdm_rev = rdm_rev.reshape(N, 16)

        # Concatenate the rdms for the two separate combinations.
        rdm_avg = 0.5 * (rdm + rdm_rev) # rdm_avg.shape = (N, 16)

        rdms.append(rdm_avg)
    obs = np.array(rdms).transpose((1, 0, 2)) # rdms.shape == (Q*(Q-1)//2, N, 16)
    return obs                                # obs.shape  == (N, Q*(Q-1)//2, 16)

def rdm_2q_mean_real(states):
    rdms = rdm_2q_mean_complex(states) # rdms.shape = (N, Q*(Q-1), 16)
    return np.dstack([rdms.real, rdms.imag])


#------------------------------ Reward functions ------------------------------#
def sparse(entanglements, prev_entanglements, epsi):
    terminated = np.all(entanglements <= epsi, axis=1)
    rewards = -5. + 10. * terminated
    return rewards

def relative_delta(entanglements, prev_entanglements, epsi):
    # We should probably use both deltas. If the agent starts applying a gate
    # to a disentangled pair it results in swapping the entanglements and thus
    # can achieve the max reward at each step. So we have to penalize for the
    # other entanglement.
    entanglements = np.maximum(entanglements, epsi)
    prev_entanglements = np.maximum(prev_entanglements, epsi)
    deltas = (prev_entanglements - entanglements) / np.maximum(prev_entanglements, entanglements)
    entangled_qubits = (entanglements > epsi).sum(axis=1)
    rewards = deltas.sum(axis=1) - entangled_qubits
    return rewards

#