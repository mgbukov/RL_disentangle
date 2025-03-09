import itertools

import numpy as np
import pytest

import context
from src.quantum_env import QuantumEnv
import src.stategen as stategen

NUM_TESTS = 100


@pytest.mark.parametrize("num_envs, num_qubits", itertools.product((1,4,64), (4,5,6)))
def test_fast_obs(num_envs, num_qubits):

    # Initialize state generator
    sg = stategen.StateGenerator(stategen.sample_haar_full, num_qubits)

    for _ in range(NUM_TESTS):
        env_params = {
            "num_qubits":           num_qubits,
            "num_envs":             num_envs,
            "epsi":                 1e-2,
            "max_episode_steps":    100,
            "obs_fn":               "rdm_2q_mean_real",
            "state_generator":      sg,
        }
        base_env = QuantumEnv(fast_obs=False, **env_params)
        base_env.reset()
        initial_states = base_env.simulator.states

        fast_env = QuantumEnv(fast_obs=True, **env_params)
        fast_env.reset()
        fast_env.set_states(initial_states)

        assert np.all(base_env.simulator.states == fast_env.simulator.states)

        for i in range(30):
            acts = np.random.choice(
                list(range(base_env.single_action_space.n)),
                base_env.num_envs
            )
            base_obs, base_reward, base_term, base_trunc, _ = base_env.step(acts)
            fast_obs, fast_reward, fast_term, fast_trunc, _ = fast_env.step(acts)

            assert np.all(base_term == fast_term)
            assert np.all(base_trunc == fast_trunc)
            assert np.all(base_reward == fast_reward)

            # Compare states only for non-terminated / non-truncated trajectories
            for n in range(num_envs):
                if base_term[n] == False:
                    assert np.allclose(base_obs[n], fast_obs[n], atol=1e-4)

            if np.any(base_term | base_trunc):
                break


