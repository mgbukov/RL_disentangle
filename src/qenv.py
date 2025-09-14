from itertools import combinations, permutations
from collections import namedtuple
from typing import List, Tuple, Optional, Union

import numpy as np
import torch

from .stategen import StateGenerator, sample_haar_full
from . import observations
from . import rewards
from .util import torch_sqe


observation_space   = namedtuple("observation_space", ["shape"])
action_space        = namedtuple("action_space", ["n"])


class QEnv:

    def __init__(self, num_qubits: int, num_envs: int, epsi: float = 1e-3,
                 max_episode_steps: int = 1000, act_space: str = "reduced",
                 reward_fn: str = "sparse", obs_fn: str = "phase_norm",
                 state_generator: Optional[StateGenerator] = None,
                 fast_ents: bool = False, fast_obs: bool = False,
                 swaps: bool = True, device: Union[str, torch.device] = "cpu"):
        self.device = device
        self.num_envs = num_envs
        self.num_qubits = num_qubits
        self.epsi = epsi
        self.max_episode_steps = max_episode_steps
        self.fast_obs = fast_obs

        # Bind state generator
        if state_generator is None:
            self.state_generator = StateGenerator(sample_haar_full, num_qubits)
        else:
            self.state_generator = state_generator

        # Initialize action space:
        #   * If `act_space == "full"`, then actions are unordered pairs.
        #     Both (i,j) and (j,i) are in the action space.
        #   * If `act_space == "reduced`, then actions are ordered pairs.
        #     Only (i,j) is in the action space and i < j.
        self.act_space = act_space
        if act_space == "full":
            self.actions = dict(enumerate(permutations(range(num_qubits), 2)))
        elif act_space == "reduced":
            self.actions = dict(enumerate(combinations(range(num_qubits), 2)))
        else:
            raise ValueError
        self.act_to_key = {v:k for k, v in self.actions.items()}
        self.num_actions = len(self.actions)
        self.single_action_space = action_space(n=self.num_actions)

        # Initialize state simulator
        self.simulator = VectorizedQState(num_qubits, num_envs, swaps, fast_ents, device)

        # Bind reward and observation functions
        self.reward_fn = getattr(rewards, reward_fn)
        self.obs_fn = getattr(observations, obs_fn)

        # `last_obs` is used when `fast_obs` is True
        self.last_obs = None

        # Initialize `self.episode_len` and `self.accumulated_return`.
        # Because they are small, we keep them in host memory.
        # Get the `dtype` of the returned observation from `reset()`
        # and define the observation space
        self.episode_len = torch.zeros(self.num_envs, dtype=torch.float32)
        self.accumulated_return = torch.zeros(self.num_envs, dtype=torch.float32)
        obs, _ = self.reset()
        self.obs_dtype = obs.dtype
        self.single_observation_space = observation_space(shape=obs.shape[1:])

    def reset(self):
        """Reset the environment to its initial state."""
        for i in range(self.num_envs):
            self.reset_sub_environment(i)
        self.episode_len[:] = 0
        self.accumulated_return[:] = 0
        # Update observations
        if self.fast_obs:
            obs = self.obs_fn(self.simulator.states, None, None)
            self.last_obs = obs
        else:
            obs = self.obs_fn(self.simulator.states)
            self.last_obs = None
        return obs, {}

    def reset_sub_environment(self, i: int):
        x = torch.from_numpy(self.state_generator()).to(device=self.device)
        self.simulator[i] = x

    def set_states(self, x: Union[torch.Tensor, np.ndarray]):
        # Check type
        if isinstance(x, np.ndarray):
            self.simulator.states = torch.from_numpy(x).to(device=self.device)
        elif isinstance(x, torch.Tensor):
            self.simulator.states = x.to(device=self.device)
        else:
            raise ValueError(
                f"Expected `torch.Tensor` or `numpy.ndarray`, got `{type(x)}`"
            )
        # Reset `episode_len`, `accumulated_return``
        self.episode_len[:] = 0.0
        self.accumulated_return[:] = 0.0
        # Update `last_obs`
        if self.fast_obs:
            obs = self.obs_fn(self.simulator.states, None, None)
            self.last_obs = obs
        else:
            obs = self.obs_fn(self.simulator.states)
            self.last_obs = None

    def step(self, acts: List[int], reset=True):
        # Store the current entanglements before applying the actions.
        prev_entanglements = self.simulator.entanglements.clone()

        # Apply the actions on the simulator
        indices = [self.actions[a] for a in acts]
        self.simulator.apply(indices)

        # Check if an environment was truncated or terminated.
        # It is perfectly fine to return both as true.
        self.episode_len += 1.0
        truncated = (self.episode_len >= self.max_episode_steps)
        terminated = torch.all(self.simulator.entanglements <= self.epsi, dim=1)

        # Get the rewards from the current system states and compute the returns.
        rewards = self.reward_fn(
            self.simulator.entanglements, prev_entanglements, self.epsi)
        self.accumulated_return += rewards

        # If any of the sub-environments are done, then fill in the info dict
        # and reset them.
        info = {}
        done = (terminated | truncated)
        if done.any():
            # Adhere to the OpenAI gym environment API
            info = {
                # "final_observation": np.where(d, self._observations(), None),
                "episode": {
                    "r": np.where(done, self.accumulated_return, [None]),
                    "l": np.where(done, self.episode_len, [None]),
                },
            }
            # Reset only the sub-environments that were done
            for k in range(self.num_envs):
                if not done[k] or not reset:
                    continue
                self.reset_sub_environment(k)
                self.accumulated_return[k] = 0.0
                self.episode_len[k] = 0

        # Finally, get the observations from the next obtained states.
        # Note that we have to do this only after we check for done environments
        if self.fast_obs:
            if self.last_obs is None:
                x = torch.from_numpy(self.simulator.states).to(self.device)
                obs = self.obs_fn(x, None, None)
            else:
                modified = []
                for n in range(self.num_envs):
                    if done[n]:
                        m = tuple(range(self.num_qubits))
                    else:
                        m = self.actions[acts[n]]
                    modified.append(m)
                x = self.simulator.states.to(self.device)
                obs = self.obs_fn(x, self.last_obs, modified)
            self.last_obs = obs
        else:
            obs = self.obs_fn(self.simulator.states).to(self.device)

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

        # Cast to torch tensors before returning
        return (
            obs,
            rewards.to(self.device),
            terminated.to(self.device),
            truncated.to(self.device),
            info
        )

    def close(self):
        pass


class VectorizedQState:

    def __init__(self, num_qubits: int, num_envs: int, swaps: bool = True,
                 fast_ents: bool = True, device: str|torch.device = "cpu"):
        assert num_qubits > 1
        self.device = torch.device(device)
        self.num_qubits = num_qubits
        self.num_envs = num_envs
        self.swaps = swaps
        self.fast_ents = fast_ents

        # Every system in the vectorized environment is represented as a
        # torch Tensor of complex numbers with shape (`num_envs`, 2, 2, ..., 2)
        self.shape = (num_envs,) + (2,) * num_qubits
        self._states = torch.zeros(self.shape, dtype=torch.complex64, device=self.device)

        # Cache the single qubit entanglements of each system.
        # Keep this tensor in host memory.
        self.entanglements = torch.empty(
            (num_envs, num_qubits), dtype=torch.float32, device="cpu"
        )

        # Order of qubits
        self.qubits_order = torch.tile(torch.arange(num_qubits), (num_envs, 1))

        # Dynamic attributes - assinged only after call to `self.apply()`:
        #   RDMs
        self.rdms_ = None
        #   Unitary gates applied in last action
        self.Us_ = None
        #   Boolean indicators of (i,j)->(j,i) swaps before U*|psi>
        self.preswaps_ = None
        #   Boolean indicators of (j,i)->(i,j) swaps after U*|psi>
        self.postswaps_ = None

    def __getitem__(self, i: int):
        return self._states[i]

    def __setitem__(self, i: int, x: torch.Tensor):
        assert x.ndim == self.num_qubits
        self._states[i] = x.to(device=self.device)
        e = torch_sqe(x, batched=False)
        # Update entanglements and qubit order
        self.qubits_order[i] = torch.arange(self.num_qubits)
        self.entanglements[i] = e.cpu()

    @property
    def states(self): return self._states

    @states.setter
    def states(self, x: torch.Tensor):
        if x.ndim != self._states.ndim:
            raise ValueError("Different number of qubits!")
        self.num_envs = x.shape[0]
        self.shape = x.shape
        self._states = phase_norm(x).to(dtype=torch.complex64, device=self.device)
        e = torch_sqe(self._states, batched=True)
        self.entanglements = e.cpu()

    def apply(self, indices: List[Tuple[int,int]]):
        # Check input
        if len(indices) != self.num_envs:
            raise ValueError(f"Expected list with length {self.num_envs}")

        N, Q = self.num_envs, self.num_qubits
        # eps = torch.tensor([np.finfo(np.float32).eps], dtype=torch.float32)
        eps = torch.tensor([1e-3], dtype=torch.float32)
        batch = self._states

        ax0 = np.arange(N)
        input_indices = np.array(indices)

        # Constuct an effective list of qubit pairs (i,j) to be moved to
        # positions (0,1)
        # The effective list is the one after appying preswap gate (maybe)
        if self.swaps:
            # We will apply the action so that the leading quibt is the
            # one that has more entanglement entropy.
            ent_q0 = self.entanglements[ax0, input_indices[:, 0]]
            ent_q1 = self.entanglements[ax0, input_indices[:, 1]]
            ent_relation = ent_q0 >= (ent_q1 + eps)
            assert ent_relation.dtype == torch.bool
            indices_01 = np.where(
                ent_relation.cpu().numpy()[:, None],
                input_indices,
                input_indices[:,::-1]
            )
            self.preswaps_ = ~ent_relation
            # print("pre ent_relation:", ent_relation)
        else:
            ent_relation = torch.zeros(N, dtype=torch.bool)
            indices_01 = input_indices.copy()
            self.preswaps_ = torch.zeros(N, dtype=torch.bool)

        # Move qubits which are about to be multiplied with U at indices (0, 1)
        permute_qubits(batch, indices_01, inverse=False)

        # /DEBUG/ Update the order of qubits
        for n in range(self.num_envs):
            is_swapped = self.preswaps_[n]
            if is_swapped:
                i, j = indices_01[n]
                _qi = self.qubits_order[n,i].clone()
                _qj = self.qubits_order[n,j].clone()
                self.qubits_order[n,i] = _qj
                self.qubits_order[n,j] = _qi

        # [CUDA] Compute 4x4 reduced density matrices
        batch = batch.reshape(N, 4, 2 ** (Q - 2))
        # print("batch:\n", batch)
        rdms = torch.einsum("...ik, ...jk-> ...ij", batch, batch.conj())
        # rdms = batch @ torch.permute(batch.resolve_conj(), [0,2,1])
        # print("RDMs:\n", rdms)
        D = torch.diag(torch.tensor([1.0, 2.0, 4.0, 8.0], device=self.device))
        rdms += torch.finfo(rdms.dtype).eps * D
        self.rdms_ = rdms

        # [CUDA] Compute unitary gates
        rhos, Us = torch.linalg.eigh(rdms)
        self.Us_ = Us
        # rhos = rhos.to(dtype=torch.float32)
        # Us = Us.to(dtype=torch.complex64)
        # print("rhos:\n", rhos)
        # print("\n\n[torch.linalg.eigh] Us:\n", Us, "\n\n")
        j = torch.tensor([1.0j], dtype=torch.complex64, device=self.device)
        for n in range(N):
            max_col = torch.abs(Us[n]).argmax(dim=0)
            # print(max_col)
            for k in range(4):
                # print(torch.exp(j * torch.angle(Us[n, max_col[k], k])))
                # print(k, Us[n, :, k])
                Us[n, :, k] *= torch.exp(-j * torch.angle(Us[n, max_col[k], k]))
                # print(k, Us[n,:,k])
        Us = torch.swapaxes(Us.conj(), 1, 2)
        # self.Us_ = Us.resolve_conj()
        self.rhos_ = rhos

        # [CUDA] Apply unitary gates
        batch = torch.matmul(Us, batch).reshape(self.shape)
        # print("batch:", batch)

        # Move qubits from (0,1) to (i,j)
        permute_qubits(batch, indices_01, inverse=True)
        self._states = phase_norm(batch)

        # [CPU] Recalculate entanglements only for qubits (i,j) on which
        # U was applied.
        if self.fast_ents:
            Sent_q0, Sent_q1 = calculate_sqe_from_rhos(rhos.cpu())
            self.entanglements[ax0, indices_01[:, 0]] = Sent_q0
            self.entanglements[ax0, indices_01[:, 1]] = Sent_q1
        else:
            self.entanglements = torch_sqe(self._states, batched=True)
        # print("[VectorizedQState.apply()] post entanglements:\n",
            #   self.entanglements.numpy().round(4))

        # print("input_indices:", input_indices)

        # Do postswaps
        if self.swaps:
            ent_q0 = self.entanglements[ax0, input_indices[:, 0]]
            ent_q1 = self.entanglements[ax0, input_indices[:, 1]]
            post_ent_relation = ent_q0 >= (ent_q1 + eps)
            self.postswaps_ = torch.empty_like(self.preswaps_)
            for n in range(N):
                if post_ent_relation[n] ^ ent_relation[n]:
                    i, j = input_indices[n]
                    self._states[n] = torch.swapaxes(self._states[n].clone(), i, j)
                    ent_i = self.entanglements[n,i].clone()
                    ent_j = self.entanglements[n,j].clone()
                    # print("postswap", n, i, j, ent_i, ent_j)
                    # print("postswap before:", self.entanglements[n])
                    self.entanglements[n,i] = ent_j
                    self.entanglements[n,j] = ent_i
                    # print("postswap after:", self.entanglements[n])
                    self.postswaps_[n] = True
                else:
                    self.postswaps_[n] = False
        else:
            self.postswaps_ = torch.zeros_like(self.preswaps_)

        # /DEBUG/ Update the order of qubits
        for n in range(self.num_envs):
            is_swapped = self.postswaps_[n]
            if is_swapped:
                i, j = indices_01[n]
                _qi = self.qubits_order[n,i].clone()
                _qj = self.qubits_order[n,j].clone()
                self.qubits_order[n,i] = _qj
                self.qubits_order[n,j] = _qi

def phase_norm(states: torch.Tensor):
    """Normalizes the relative phase shift between different qubits in one system."""
    B = states.shape[0]
    L = states.ndim - 1
    first = states.reshape(B, -1)[:, 0]
    phi = torch.angle(first)
    z = torch.cos(phi) - 1j * torch.sin(phi)
    result = states * z.reshape((B,) + (1,) * L)
    # Set explicitly the imaginary part of first component to 0. This operation
    # is mandatory, because above multiplication can leave the imaginary part
    # nonzero, which breaks batch & solo rollout equivalence tests.
    for i in range(B):
        result[i].view(-1)[0] = result[i].view(-1)[0].real.item()
    return result


def permute_qubits(
        batch: torch.Tensor,
        qubits_indices: List[Tuple[int,int]],
        inverse: bool = False
    ):
    """
    Moves qubit pairs specified in `qubits_indices` to
    dimensions (1,2) in `batch`
    """
    assert len(batch) == len(qubits_indices)
    # PyTorch does not allow assigning to tensor with itself
    x = batch.clone()
    for i, pair in enumerate(qubits_indices):
        if not isinstance(pair, tuple):
            pair = tuple(pair)
        if inverse:
            batch[i] = torch.movedim(x[i], (0,1), pair)
        else:
            batch[i] = torch.movedim(x[i], pair, (0,1))


def calculate_sqe_from_rhos(rhos: torch.Tensor):
    a = rhos[:, 0] + rhos[:, 1] # rho_{q0-1}
    b = rhos[:, 2] + rhos[:, 3] # rho_{q0-2}
    c = rhos[:, 0] + rhos[:, 2] # rho_{q1-1}
    d = rhos[:, 1] + rhos[:, 3] # rho_{q1-2}
    eps = torch.tensor([torch.finfo(rhos.dtype).eps]).to(device=rhos.device)
    Sent_q0 = -a * torch.log(torch.maximum(a, eps)) - b * torch.log(torch.maximum(b, eps))
    Sent_q1 = -c * torch.log(torch.maximum(c, eps)) - d * torch.log(torch.maximum(d, eps))
    return Sent_q0, Sent_q1
