import numpy as np


class BruteForceSearch:

    #-------------------------------- nested Node class ---------------------------------#
    class Node:
        """Lightweight, nonpublic class for storing a tree node."""

        __slots__ = "item", "parent", "action", "cost"  # Streamline memory usage.

        def __init__(self, item, parent, action, cost):
            """ Initialize a Node instance.

            Args:
                item (Any): The state to which this node corresponds.
                parent (Node): The node in the tree that generated this node.
                action (action): The action that was applied to the parent's state to
                    generate this node's state.
                cost (float): The cost associated with the state of this node.
            """
            self.item = item
            self.parent = parent
            self.action = action
            self.cost = cost

        def __hash__(self): return hash(self.item)

    #--------------------------------- public accessors ---------------------------------#
    def start(self, psi, env, num_iters=1000, verbose=True):
        """Run the tree search algorithm and build the solution path.

        Args:
            psi (np.Array): A numpy array of shape (2,2,...,2), giving the initial state
                of the environment.                '--- L ---'
            env (QubitsEnvironment): Environment object.
            num_steps (int): Depth limit to run the search for.
            verbose (bool): If true, print search statistics.

        Returns:
            path (list[actions]): A list of actions giving the path from the start state
                to the solution state.
        """
        goal_nodes = self._run(psi, env, num_iters, verbose)
        return self._decode(goal_nodes)

    #--------------------------------- private methods ----------------------------------#
    def _run(self, psi, env, num_iters, verbose):
        """Start the tree search and continuously explore the state space until a solution
        is found, or until the maximum number of iterations is reached. Return all the goal
        nodes if any are found, otherwise return None.
        """
        goal_nodes = []

        node = self.Node(item=psi, parent=None, action=None, cost=0)
        fringe = [node]
        for _ in range(num_iters):
            # Batch the states from the fringe and set them as the environment states.
            states = [node.item for node in fringe]
            states = np.repeat(np.array(states), env.num_actions, axis=0)
            env.states = states

            # Transition into the next states by considering all possible actions.
            actions = np.tile(np.array(list(env.actions.keys())), (len(fringe),))
            next_states, _, _ = env.step(actions)

            # Compute child costs and check if the goal is achieved.
            costs = env.entropy()
            for i, c in enumerate(costs):
                if np.all(c < env.epsi):
                    goal_nodes.append(self.Node(
                        next_states[i], fringe[i // env.num_actions], actions[i], costs[i]))
            if len(goal_nodes) > 0: return goal_nodes

            # Get the successors and store them as the new fringe.
            fringe = [self.Node(s, fringe[i // env.num_actions], actions[i], costs[i])
                          for i,s in enumerate(next_states)]

            # Maybe plot progress results.
            if verbose:
                beam_entropies = np.array([node.cost for node in fringe])
                print("Mean entropies in current beam: ", beam_entropies.round(10).ravel())

        # If no solution is found for the given number of iterations, then return None.
        return None

    def _decode(self, goal_nodes):
        """Given a goalNode produced from the tree search algorithm, trace the backward
        path from the goal state to the start state by following the pointers to the
        parents. Return the reversed path.

        Args:
            goalNode (Node): Node object returned from the tree search algorithm.

        Returns:
            path (list[action]): A list of actions giving the path from the start state to
                the goalNode state.
        """
        if goal_nodes is None: return []
        paths = []

        for goal in goal_nodes:
            path = []
            current = goal
            while current.parent is not None:
                path.append(current.action)
                current = current.parent
            path.reverse()
            paths.append(path)
        return paths

#