import heapq
import numpy as np



class Node2:

    def __init__(self, item, path=tuple(), cost=None):
        self.item = item
        self.path = tuple(path)
        self.cost = cost

    def __hash__(self):
        return hash(self.item)


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
        goal = self._run(psi, env, qubit, num_iters, verbose)
        solution = goal.path if goal is not None else []
        env.states = np.expand_dims(psi, 0)
        return solution

    #--------------------------------- private methods ----------------------------------#
    # def _run(self, psi, env, qubit, num_iters, verbose):
    #     """Start the tree search and continuously explore the state space until a solution
    #     is found, or until the maximum number of iterations is reached. Return the goal
    #     node if it is found, otherwise return None.
    #     """
    #     node = self.Node(item=psi, parent=None, action=None, cost=0)
    #     fringe = [node]
    #     for _ in range(num_iters):
    #         # Batch the states from the fringe and set them as the environment states.
    #         states = [node.item for node in fringe]
    #         states = np.repeat(np.array(states), env.num_actions, axis=0)
    #         env.states = states.copy()

    #         # Transition into the next states by considering all possible actions.
    #         actions = np.tile(np.array(list(env.actions.keys())), (len(fringe),))
    #         next_states, _, _ = env.step(actions)

    #         # Compute child costs and check if the goal is achieved.
    #         costs = env.entropy().mean(axis=-1) if qubit is None else env.entropy()[:, qubit]
    #         if (costs < env.epsi).any(): # Return the child node that reaches the goal.
    #             for i, c in enumerate(costs):
    #                 if c < env.epsi:
    #                     assert env.Entropy(np.expand_dims(next_states[i], 0)) == c
    #                     goal = self.Node(
    #                         next_states[i],
    #                         fringe[i // env.num_actions],
    #                         actions[i],
    #                         costs[i]
    #                     )
    #                     path = self._decode(goal)
    #                     env.states = np.expand_dims(psi, 0)
    #                     rollout_states = [env.states.copy()[0]]
    #                     rollout_entropies = [env.entropy()[0,0]]
    #                     print('\t roll:', env.entropy())
    #                     for a in path:
    #                         nxts, _, _ = env.step([a])
    #                         print('\t roll:', env.entropy())
    #                         rollout_states.append(nxts[0])
    #                         rollout_entropies.append(env.entropy()[0,0])
                        
    #                     rollout_states = np.array(rollout_states)
    #                     parent_states = [goal.item]
    #                     parent_costs = [goal.cost]
    #                     p = goal.parent
        #                 while p is not None:
        #                     parent_states.append(p.item)
        #                     parent_costs.append(p.cost)
        #                     p = p.parent
        #                 parent_states.reverse()
        #                 parent_costs.reverse()
                        
        #                 parent_states = np.array(parent_states)
        #                 print(parent_costs, rollout_entropies)
                        
        #                 for s1, s2 in zip(rollout_states, parent_states):
        #                     print(np.isclose(s1, s2).all())
        #                 assert env.entropy()[0,0] <= c
        #                 return goal

        #     # Get the k-best successors and store them as the new fringe.
        #     # successors = []
        #     # for i, s in enumerate(next_states):
        #     #     parent = fringe[i // env.num_actions]
        #     #     assert np.all(parent.item == states[i])
        #     #     action = actions[i]
        #     #     cost = costs[i]
        #     #     env.states = np.expand_dims(parent.item.copy(), 0)
        #     #     nexts, _, _ = env.step([action])
        #     #     assert np.all(nexts[0] ==  s)
        #     #     assert env.entropy()[0,0] == cost
        #     #     successors.append(self.Node(s, parent, action, cost))
        #     successors = [self.Node(s, fringe[i // env.num_actions], actions[i], costs[i])
        #                 for i,s in enumerate(next_states)]
        #     fringe = self._getKBest(successors, self.beam_size)

        #     # Maybe plot progress results.
        #     if verbose:
        #         beam_entropies = np.array([node.cost for node in fringe])
        #         print("Mean entropies in current beam: ", beam_entropies.round(10).ravel())

        # # If no solution is found for the given number of iterations, then return None.
        # return None

    def _run(self, psi, env, qubit, num_iters, verbose=False):
        fringe = [Node2(psi)]
        states = [psi]
        for _ in range(num_iters):
            states = [node.item for node in fringe]
            states = np.repeat(np.array(states), env.num_actions, axis=0)
            assert len(states) == len(fringe) * env.num_actions
            env.states = states
            # Transition into the next states by considering all possible actions.
            actions = np.tile(np.array(list(env.actions.keys())), (len(fringe),))
            next_states, _, _ = env.step(actions)
            if qubit is None:
                costs = env.entropy().mean(axis=-1)
            else:
                costs = env.Entropy(next_states)[:, qubit]
            imincost = np.argmin(costs)
            mincost = costs[imincost]
            if mincost < env.epsi:
                # assert env.Entropy(np.expand_dims(next_states[i], 0)) < env.epsi
                path = fringe[imincost // env.num_actions].path
                act = actions[imincost]
                return Node2(next_states[imincost], path + (act,), mincost)
            successors = []
            for i in range(len(next_states)):
                parent = fringe[i // env.num_actions]
                node = Node2(next_states[i], parent.path + (actions[i],), costs[i])
                successors.append(node)
            fringe = self._getKBest(successors, self.beam_size)
            # Maybe plot progress results
            if verbose:
                mean_cost = np.mean([node.cost for node in fringe])
                print("Mean cost in current beam: ", mean_cost)
        return None

    def _decode(self, goalNode):
        """Given a goalNode produced from the tree search algorithm, trace the backward
        path from the goal state to the start state by following the pointers to the
        parents. Return the reversed path.

        Args:
            goalNode (Node): Node object returned from the tree search algorithm.

        Returns:
            path (list[action]): A list of actions giving the path from the start state to
                the goalNode state.
        """
        if goalNode is None: return []
        path = []
        current = goalNode
        while current.action is not None:
            path.append(current.action)
            current = current.parent
        path.reverse()
        return path

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