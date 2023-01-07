import heapq
import numpy as np


class BeamSearch:
    """Beam search works by limiting the size of the fringe to a fixed size k, called the
    beam size. At every step the algorithm expands all of the nodes in the fringe, instead
    of picking one. All of the generated successors are evaluated, and the k-best
    successors are kept in the fringe for the next iteration.

    Attributes:
        beam_size (int): Size of the fringe.
    """

    #-------------------------------- nested Node class ---------------------------------#
    class Node:
        """Lightweight, nonpublic class for storing a tree node."""

        __slots__ =  ("item", "path", "cost")

        def __init__(self, item, path=tuple(), cost=None):
            """ Initialize a Node instance.

            Args:
                item (Any): The state to which this node corresponds.
                path (tuple): The path from root to `self`. Path is tuple of actions.
                cost (float): The cost associated with the state of this node.
            """
            self.item = item
            self.path = tuple(path)
            self.cost = cost

        def __hash__(self):
            return hash(self.item)

    #------------------------------ container initializer -------------------------------#
    def __init__(self, beam_size=1):
        """Initialize a new beam search instance.

        Args:
            beam_size (int): Maximum size of the beam for the search algorithm.
        """
        self.beam_size = beam_size

    #--------------------------------- public accessors ---------------------------------#
    def start(self, psi, env, qubit=None, num_iters=1000, verbose=True):
        """Run the tree search algorithm and build the solution path.

        Args:
            psi (np.Array): A numpy array of shape (2,2,...,2), giving the initial state
                of the environment.                '--- L ---'
            env (QubitsEnvironment): Environment object.
            qubit (int): The index of the qubit to be disentangled from the system.
                Default value is None.
            num_iters (int): Number of iterations to run the search for.
            verbose (bool): If true, print search statistics.

        Returns:
            path (list[actions]): A list of actions giving the path from the start state
                to the solution state.
        """
        envstate = env.states
        goal = self._run(psi, env, qubit, num_iters, verbose)
        solution = goal.path if goal is not None else []
        env.states = envstate
        return solution

    #--------------------------------- private methods ----------------------------------#
    def _run(self, psi, env, qubit, num_iters, verbose=False):
        """Start the tree search and continuously explore the state space
        until a solution is found, or until the maximum number of iterations
        is reached. Return the goal node if it is found, otherwise return None.
        """
        fringe = [BeamSearch.Node(psi)]
        states = [psi]
        for _ in range(num_iters):
            # Batch the states from the fringe and set them as the
            # environment states
            states = [node.item for node in fringe]
            states = np.repeat(np.array(states), env.num_actions, axis=0)
            assert len(states) == len(fringe) * env.num_actions
            env.states = states
            # Transition into the next states by considering all possible actions
            actions = np.tile(np.array(list(env.actions.keys())), (len(fringe),))
            next_states, _, _ = env.step(actions)
            # Compute child costs
            if qubit is None:
                costs = env.entropy().mean(axis=-1)
            else:
                costs = env.Entropy(next_states)[:, qubit]
            imincost = np.argmin(costs)
            mincost = costs[imincost]
            # Check if a goal state is reached
            if mincost < env.epsi:
                # assert env.Entropy(np.expand_dims(next_states[i], 0)) < env.epsi
                path = fringe[imincost // env.num_actions].path
                act = actions[imincost]
                return BeamSearch.Node(next_states[imincost], path + (act,), mincost)
            # Get the k-best successors and store them as the new fringe
            successors = []
            for i in range(len(next_states)):
                parent = fringe[i // env.num_actions]
                node = BeamSearch.Node(next_states[i], parent.path + (actions[i],), costs[i])
                successors.append(node)
            fringe = self._getKBest(successors, self.beam_size)
            # Maybe plot progress results
            if verbose:
                mean_cost = np.mean([node.cost for node in fringe])
                print("Mean cost in current beam: ", mean_cost)
        # If no solution is found for the given number of iterations return None
        return None

    @staticmethod
    def _getKBest(nodes, k):
        """Heapify the list and pop k times to obtain the k-best items."""
        heap, count = [], 0
        for node in nodes:
            entry = (node.cost, count, node)
            count += 1
            heapq.heappush(heap, entry)
        return [heapq.heappop(heap)[-1] for _ in range(min(k, count))]

#