from collections import namedtuple
import numpy as np

from . import observations
from . import rewards
from .quantum_state import VectorQuantumState
from .stategen import StateGenerator, sample_haar_full



observation_space = namedtuple("observation_space", ["shape"])
action_space = namedtuple("action_space", ["n"])

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
DEFAULT_SWAP_THRESHOLD = float(np.finfo(np.float32).eps)


class QuantumEnv():
    """QuantumEnv is a wrapper around VectorQuantumState conforming to the
    OpenAI Gym API.
    """

    def __init__(self,
                 num_qubits,
                 num_envs,
                 epsi=1e-3,
                 max_episode_steps=1000,
                 reward_fn="sparse",
                 obs_fn="phase_norm",
                 state_generator=None,
                 fast_obs=False,
                 swaps=True,
                 swap_threshold=DEFAULT_SWAP_THRESHOLD
    ):
        """Init a Quantum environment.

        Args:
            num_qubits: int
                Number of qubits in a quantum state.
            num_envs: int
                Number of quantum states for the vectorized environment.
            epsi: float, optional
                Threshold for disentangling a quantum state. Default: 1e-3.
            max_episode_steps: int, optional
                Maximum number of steps before truncating the environment.
                Default: 1000.
            reward_fn: string, optional
                The name of the reward function to be used. One of
                ["sparse", "relative_delta"]. Default: "sparse".
            obs_fn: string, optional
                The name of the observation function to be used. One of
                ["phase_norm", "rdm_1q", "rdm_2q_complex", "rdm_2q_real",
                 "rdm_2q_half", "rdm_2q_mean_real", "rdm_2q_nisq_mean",
                 "rdm_2q_nisq_mean_real"].
                Default: "phase_norm".
            state_generator: optional
                Controls how new states are generated in reset(). See
                `VectorQuantumState`
            fast_obs: bool, default=False
                If True, the previous observation and additional info is passed
                to `obs_fn` to compute the next observation.
                This can speed up RDM observations, because RDMs can be
                calculated only for the subset of qubit pairs which the actions
                touches.
            swaps (bool, default=True):
                See VectorQuantumState.__init__()
            swap_threshold (float, default=np.finfo(np.float32).eps):
                Minimum single-qubit entanglement entropy difference required
                for swap gate to be applied
        """
        # Private
        self.num_qubits = num_qubits
        self.epsi = epsi
        self.max_episode_steps = max_episode_steps
        act_space = "reduced"
        self.simulator = VectorQuantumState(
            num_qubits=num_qubits,
            num_envs=num_envs,
            act_space=act_space,
            state_generator=state_generator,
            swaps=swaps,
            swap_threshold=swap_threshold
        )
        self.reward_fn = getattr(rewards, reward_fn)
        self.obs_fn = getattr(observations, obs_fn)
        self.obs_dtype = self.obs_fn(self.simulator.states).dtype
        self.fast_obs = fast_obs
        self.last_obs = None

        # Every time `self.reset()` is called a new state must be generated.
        # The generation of new states is handled by a function,
        # which is selected using `state_generator` and `generator_kwargs`
        # paratemers.
        if state_generator is None:
            self.state_generator = StateGenerator(sample_haar_full, num_qubits)
        else:
            self.state_generator = state_generator

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
        self.episode_len = np.zeros(self.num_envs)
        self.accumulated_return = np.zeros(self.num_envs)
        if self.fast_obs:
            obs = self.obs_fn(self.simulator.states, None, None)
            self.last_obs = obs
        else:
            obs = self.obs_fn(self.simulator.states)
        return obs, {}

    def set_states(self, states):
        self.simulator.states = states
        self.episode_len = np.zeros(self.num_envs)
        self.accumulated_return = np.zeros(self.num_envs)
        self.last_obs = None

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
            for k in range(self.num_envs):
                if not done[k] or not reset:
                    continue
                self.simulator.reset_sub_environment_(k)
                self.accumulated_return[k] = 0.
                self.episode_len[k] = 0

        # Finally, get the observations from the next obtained states.
        # Note that we have to do this only after we check for done environments.
        if self.fast_obs:
            if self.last_obs is None:
                obs = self.obs_fn(self.simulator.states, None, None)
            else:
                modified = []
                for n in range(self.num_envs):
                    if done[n]:
                        m = tuple(range(self.num_qubits))
                    else:
                        m = self.simulator.actions[acts[n]]
                    modified.append(m)
                obs = self.obs_fn(self.simulator.states, self.last_obs, modified)
            self.last_obs = obs
        else:
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
