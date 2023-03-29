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
    def __init__(self, beam_size=1, epsi=1e-3):
        """Initialize a new beam search instance.

        Args:
            beam_size (int): Maximum size of the beam for the search algorithm.
        """
        self.beam_size = beam_size
        self.epsi = epsi

    #--------------------------------- public accessors ---------------------------------#
    def start(self, psi, env, qubit=None, num_iters=10_000, verbose=True):
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
            path (list[actions]): A list of actions giving the path from the
                start state to the solution state.
                If the input state is already disentangled this function returns
                an empty list. If the search procedure cannot find a solution
                this function returns None.
        """
        envstate = env.states
        goal = self._run(psi, env, qubit, num_iters, verbose)
        solution = goal.path if goal is not None else None
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
            env.apply(actions)
            next_states = env.states
            # Compute child costs
            if qubit is None:
                costs = env.entanglements.mean(axis=-1)
            else:
                costs = env.entanglements[:, qubit]
            imincost = np.argmin(costs)
            mincost = costs[imincost]
            # Check if a goal state is reached
            # if (qubit is not None and mincost < self.epsi) or (qubit is None and (env.entanglements[imincost] < self.epsi).all()):
            if (qubit is not None and mincost < self.epsi) or (qubit is None and (env.entanglements[imincost].mean() < self.epsi)):
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



class SearchExpert:
    """SearchExpert is an agent that uses a tree search procedure for solving
    the environment.
    """

    def __init__(self, beam_size=100, epsi=1e-3):
        """Init search expert agent.

        Args:
            beam_size (int): Maximum size of the beam for the search algorithm.
        """
        self.b = BeamSearch(beam_size, epsi)

    def start(self, psi, env, num_iter=10_000, verbose=False):
        # This procedure modifies the action set of the environment. Store the
        # original action set in order to restore the environment at the end.
        original_env_actions = env.actions

        actions = self._run(psi, env, num_iter, verbose)

        # Restore the original action set of the environment.
        env.actions = original_env_actions
        env.num_actions = len(original_env_actions)
        return actions

    def _run(self, psi, env, num_iter, verbose):
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
        for q in range(env.num_qubits - 1):
            # Run beam search to disentangle the next qubit from the current state.
            path = b.start(curr.copy(), env, q, num_iter, verbose)
            if path is None: return None # error
            actions.extend(path)

            # The search procedure returns only the path for disentangling the
            # qubit. In order to produce the states along the path we need to
            # explore the environment.
            env.states = np.expand_dims(curr, axis=0)
            for a in path:
                env.apply([a])
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

        return actions


class RandomAgent():

    def __init__(self, epsi=1e-3):
        self.epsi = epsi

    def start(self, psi, env, num_iter=10_000, verbose=False):
        path = []
        env.states = np.array([psi])

        for _ in range(num_iter):
            act = np.random.randint(low=0, high=env.num_actions)
            _ = env.apply([act])
            path.append(act)
            # if (env.entanglements < self.epsi).all():
            if env.entanglements.mean() < self.epsi:
                return path
        return None


class GreedyAgent():

    def __init__(self, epsi=1e-3):
        self.epsi = epsi

    def start(self, psi, env, qubit=None, num_iter=10_000, verbose=False):
        path = []
        env.states = np.array([psi])

        for _ in range(num_iter):
            acts = np.arange(env.num_actions)
            env.states = np.repeat(env.states, env.num_actions, axis=0)
            _ = env.apply(acts)

            # Compute costs.
            if qubit is None:
                costs = env.entanglements.mean(axis=-1)
            else:
                costs = env.entanglements[:, qubit]

            # Select the action greedily.
            imincost = np.argmin(costs)
            mincost = costs[imincost]
            path.append(acts[imincost])
            env.states = np.array([env.states[imincost]])

            # Check if a goal state is reached.
            # if (qubit is not None and mincost < self.epsi) or (qubit is None and (env.entanglements < self.epsi).all()):
            if (qubit is not None and mincost < self.epsi) or (qubit is None and (env.entanglements.mean() < self.epsi)):
                return path

        return None

#

if __name__ == "__main__":
    import json
    import pickle
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from src.quantum_state import VectorQuantumState

    # Test how fast is the search procedure running, and how many steps it takes.
    num_test = 5
    num_qubits = [5, 6, 8]
    results = {q : {} for q in num_qubits}

    for q in num_qubits:
        print(f"Testing {q} qubits..")

        env = VectorQuantumState(num_qubits=q, num_envs=1)

        epsi = 1e-3 if q < 8 else 1e-2
        beam = BeamSearch(beam_size=100, epsi=epsi)
        # greedy = BeamSearch(beam_size=1, epsi=epsi)
        greedy = GreedyAgent(epsi=epsi)
        beam_qbyq = SearchExpert(beam_size=100, epsi=epsi)
        greedy_qbyq = SearchExpert(beam_size=1, epsi=epsi)
        rnd = RandomAgent(epsi=epsi)

        agents = {
            "beam": beam, "greedy": greedy, "beam_qbyq": beam_qbyq, "greedy_qbyq": greedy_qbyq, "random": rnd,
        }

        for name, a in agents.items():
            print(f"  running {name} agent")

            path_len, solution_time = [], []
            for i in tqdm(range(num_test)):
                env.set_random_states_()
                psi = env.states[0]

                tic = time.time()
                path = a.start(psi, env, verbose=False)
                toc = time.time()

                if path is None: continue # skip non-solved

                path_len.append(len(path))
                solution_time.append(toc-tic)

            if len(path_len) == 0: continue # skip systems for which no solution was found

            results[q].update({
                name: {
                    "average_length": float(f"{np.array(path_len).mean():.1f}"),
                    "average_time": float(f"{np.array(solution_time).mean():.3f}"),
                },
            })

    with open("search_stats.json", "w") as f:
        json.dump(results, f, indent="  ")
    with open("search_stats.pkl", "wb") as f:
        pickle.dump(results, f)

    plt.style.use("ggplot")

    # Plot lengths
    beam_lengths = [results[q]["beam"]["average_length"] if "beam" in results[q].keys() else 0. for q in num_qubits]
    # greedy_lengths = [results[q]["greedy"]["average_length"] if "greedy" in results[q].keys() else 0. for q in num_qubits]
    greedy_lengths = [results[q]["greedy"]["average_length"] if "new_greedy" in results[q].keys() else 0. for q in num_qubits]
    beam_qbyq_lengths = [results[q]["beam_qbyq"]["average_length"] if "beam_qbyq" in results[q].keys() else 0. for q in num_qubits]
    greedy_qbyq_lengths = [results[q]["greedy_qbyq"]["average_length"] if "greedy_qbyq" in results[q].keys() else 0. for q in num_qubits]
    random_lengths = [results[q]["random"]["average_length"] if "random" in results[q].keys() else 0. for q in num_qubits]

    fig, ax = plt.subplots(facecolor="black")
    ax.plot(num_qubits, beam_lengths, label="Beam search", lw=1.)
    # ax.plot(num_qubits, greedy_lengths, label="Greedy search", lw=1.)
    ax.plot(num_qubits, greedy_lengths, label="Greedy search", lw=1.)
    ax.plot(num_qubits, beam_qbyq_lengths, label="Expert search with beam", lw=1.)
    ax.plot(num_qubits, greedy_qbyq_lengths, label="Expert greedy search", lw=1.)
    ax.plot(num_qubits, random_lengths, label="Random search", lw=1.)
    ax.legend(loc="upper left")
    fig.savefig("search_lengths.png")

    # Plot timings
    beam_times = [results[q]["beam"]["average_time"] if "beam" in results[q].keys() else 0. for q in num_qubits]
    # greedy_times = [results[q]["greedy"]["average_time"] if "greedy" in results[q].keys() else 0. for q in num_qubits]
    greedy_times = [results[q]["greedy"]["average_time"] if "new_greedy" in results[q].keys() else 0. for q in num_qubits]
    beam_qbyq_times = [results[q]["beam_qbyq"]["average_time"] if "beam_qbyq" in results[q].keys() else 0. for q in num_qubits]
    greedy_qbyq_times = [results[q]["greedy_qbyq"]["average_time"] if "greedy_qbyq" in results[q].keys() else 0. for q in num_qubits]
    random_times = [results[q]["random"]["average_time"] if "random" in results[q].keys() else 0. for q in num_qubits]

    fig, ax = plt.subplots(facecolor="black")
    ax.plot(num_qubits, beam_times, label="Beam search", lw=1.)
    # ax.plot(num_qubits, greedy_times, label="Greedy search", lw=1.)
    ax.plot(num_qubits, greedy_times, label="Greedy search", lw=1.)
    ax.plot(num_qubits, beam_qbyq_times, label="Expert search with beam", lw=1.)
    ax.plot(num_qubits, greedy_qbyq_times, label="Expert greedy search", lw=1.)
    ax.plot(num_qubits, random_times, label="Random search", lw=1.)
    ax.legend(loc="upper left")
    fig.savefig("search_times.png")