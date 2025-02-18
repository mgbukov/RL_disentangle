import numpy as np
import torch

from .quantum_env import QuantumEnv


def test_lengths(agent, state_sampler, max_steps, obs_fn="rdm_2q_mean_real",
                 num_tests=100, epsi=1e-3, greedy=True):

        # Initialize RL environment
        env = QuantumEnv(
            num_qubits=         state_sampler.num_qubits,
            num_envs=           num_tests,
            epsi=               epsi,
            max_episode_steps=  max_steps,
            obs_fn=             obs_fn,
            state_generator=    state_sampler,

        )
        env.reset()

        lengths = np.full(env.num_envs, np.nan, dtype=np.float32)

        o = env.obs_fn(env.simulator.states)
        for i in range(1, max_steps + 1):
            o = torch.from_numpy(o)
            pi = agent.policy(o)    # uses torch.no_grad

            if greedy:
                acts = torch.argmax(pi.probs, dim=1).cpu().numpy()
            else:
                acts = pi.sample().cpu().numpy()

            o, r, t, tr, infos = env.step(acts)

            # We'll update the subenv's entry in the `lengths` array,
            # only if a subenv has just just finished
            lengths[np.isnan(lengths) & t] = i

        return lengths
