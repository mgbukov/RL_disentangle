import numpy as np

from src.infrastructure.beam_search import BeamSearch


class SearchExpert:
    """SearchExpert is an agent that uses a tree search procedure for solving
    the environment.
    """

    def __init__(self, env, beam_size=100):
        """Init search expert agent.

        Args:
            env (QubitsEnvironment object): Environment object.
            beam_size (int): Maximum size of the beam for the search algorithm.
        """
        self.env = env
        self.b = BeamSearch(beam_size)

    def rollout(self, psi, num_iter=1000, verbose=False):
        """The expert searches for a solution for the given state `psi` and
        returns the trajectory that the search procedure has found.

        Args:
            psi (np.Array): A numpy array of shape (2,2,...,2), giving the state
                for which the expert should produce a solution.
            num_iter (int): Number of iterations for which beam search should be
                run. Effectively this means, the depth of the search tree.
            verbose (bool): If true, print search statistics.

        Returns:
            states (np.Array): A numpy array of shape (t+1, q), giving the states
                produced by the search agent that are along the solution path,
                where t = number of time steps, q = size of the quantum system.
            actions (np.Array): A numpy array of shape (t,) giving the actions
                selected by the expert.

        TODO:
            The expert should return torch tensors instead of numpy arrays in
            order to be in sync with the other agents.
            In addition these should be batches, i.e. states.shape = (1, t+1, q)
            and actions.shape = (1, t).

        TODO 2:
            We could improve the expert to return multiple **best** solutions.
            That is, instead of actions.shape = (1, t) it could be (n, t), where
            n is the number of solution paths that take `t` steps to solve.
        """
        b = self.b
        env = self.env

        # This procedure modifies the action set of the environment. Store the
        # original action set in order to restore the environment at the end.
        original_env_actions = self.env.actions

        # Generate a trajectory by disentangling the system qubit by qubit.
        # For every qubit run the beam search procedure to produce the part of
        # the solution that disentangles that qubit. In the end concatenate the
        # separate parts to produce the full solution.
        # Keep track of the visited states that are along the solution path. Add
        # the initial state before entering the for-loop as it will not be added
        # by the procedure.
        actions, states = [], []
        curr = psi.copy()
        states.append(np.hstack([curr.ravel().real, curr.ravel().imag]))
        for q in range(env.L - 1):
            # Run beam search to disentangle the next qubit from the current state.
            path = b.start(curr.copy(), env, q, num_iter, verbose)
            if path is None: return None, None # error
            actions.extend(path)

            # The search procedure returns only the path for disentangling the
            # qubit. In order to produce the states along the path we need to
            # explore the environment.
            env.states = np.expand_dims(curr, axis=0)
            for a in path:
                env.step([a])
                s = env.states[0]
                states.append(np.hstack([s.ravel().real, s.ravel().imag]))

            # Set the new current state. Qubit `q` is now disentangled.
            curr = env.states[0]

            # Update the action set of the environment. Since qubit `q` is now
            # disentangled, we do not want the search procedure to consider
            # any actions involving `q` -- they will not lead to a solution.
            new_env_acts = {k:v for k,v in env.actions.items() if q not in v}
            env.actions = new_env_acts
            env.num_actions = len(new_env_acts)

        # Restore the original action set of the environment.
        self.env.actions = original_env_actions
        self.env.num_actions = len(original_env_actions)
        return np.stack(states), np.array(actions)

#