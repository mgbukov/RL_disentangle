import numpy as np

from src.infrastructure.beam_search import BeamSearch


class SearchExpert:

    def __init__(self, env, beam_size=100):
        self.env = env
        self.b = BeamSearch(beam_size)

    def rollout(self, psi, num_iter=1000, verbose=False):
        b = self.b
        env = self.env
        original_env_actions = self.env.actions

        # Generate a trajectory by disentangling the system qubit by qubit.
        actions, states = [], []
        for q in range(env.L - 1):
            path = b.start(psi.copy(), env, q, num_iter, verbose)
            if path is None:
                return None, None
            actions.extend(path)
            env.states = np.expand_dims(psi, axis=0)
            for a in path:
                s = env.states[0].reshape(-1)
                s = np.hstack([s.real, s.imag])
                states.append(s)
                env.step([a])
            psi = env.states[0]
            # states.append(np.hstack([psi.ravel().real, psi.ravel().imag]))

            new_env_acts = {k:v for k,v in env.actions.items() if q not in v}
            env.actions = new_env_acts
            env.num_actions = len(new_env_acts)
            # if env.disentangled():
            #     break
            # else:
            #     return None, None
        # Append the last state
        if states:
            states.append(np.hstack([psi.ravel().real, psi.ravel().imag]))
        # Restore the environment parameters.
        self.env.actions = original_env_actions
        self.env.num_actions = len(original_env_actions)
        return np.stack(states), np.array(actions)

#