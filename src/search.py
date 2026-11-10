import heapq
import numpy as np
import torch

from src.qenv import QEnv, VectorizedQState


class BeamSearch:
    """Beam search over the quantum state space.

    At every iteration the algorithm expands all nodes in the current fringe,
    evaluates all resulting successors, and keeps only the k-best (lowest
    entanglement) nodes for the next iteration.

    Attributes:
        beam_size (int): Maximum fringe size (k).
        epsi (float): Entanglement threshold below which a state is considered
            disentangled.
    """

    class Node:
        """Lightweight container for a search-tree node."""

        __slots__ = ("item", "path", "cost")

        def __init__(self, item, path=tuple(), cost=None):
            self.item = item        # torch.Tensor — the quantum state
            self.path = tuple(path) # tuple of action indices leading here
            self.cost = cost        # float — entanglement cost

        def __hash__(self):
            return hash(id(self.item))

    def __init__(self, beam_size=1, epsi=1e-3):
        self.beam_size = beam_size
        self.epsi = epsi

    def start(self, psi, env: QEnv, qubit=None, num_iters=10_000, verbose=False):
        """Run beam search from initial state `psi`.

        Args:
            psi (torch.Tensor | np.ndarray): Initial quantum state, shape (2,...,2).
            env (QEnv): Environment defining the action set.
            qubit (int | None): If set, search only to disentangle this qubit;
                otherwise disentangle the whole system.
            num_iters (int): Maximum search depth.
            verbose (bool): Print mean fringe cost at each step.

        Returns:
            list[int] | None: Sequence of action indices, or None if no solution.
        """
        saved = env.simulator.states.clone()
        goal = self._run(psi, env, qubit, num_iters, verbose)
        env.simulator.states = saved
        return goal.path if goal is not None else None

    def _run(self, psi, env, qubit, num_iters, verbose):
        psi_t = torch.from_numpy(psi) if isinstance(psi, np.ndarray) else psi
        fringe = [BeamSearch.Node(psi_t)]
        num_actions = env.num_actions

        # Dedicated simulator that grows with the fringe × action expansion.
        # Using a separate object avoids corrupting env.simulator.qubits_order.
        expand_sim = VectorizedQState(env.num_qubits, num_envs=1)

        for _ in range(num_iters):
            batch_size = len(fringe) * num_actions
            # Expand every fringe node across every action in one batched call.
            states = torch.stack([n.item for n in fringe])            # (B, 2,...,2)
            states = states.repeat_interleave(num_actions, dim=0)     # (B*A, 2,...,2)
            expand_sim.states = states                                 # resizes num_envs
            # states.setter updates num_envs but not qubits_order; fix that.
            expand_sim.qubits_order = torch.tile(
                torch.arange(env.num_qubits), (batch_size, 1))
            action_idx = list(range(num_actions)) * len(fringe)
            expand_sim.apply([env.actions[a] for a in action_idx])

            ents = expand_sim.entanglements                           # (B*A, L)
            costs = ents[:, qubit] if qubit is not None else ents.mean(dim=-1)
            costs_np = costs.numpy()

            imincost = int(np.argmin(costs_np))
            mincost = costs_np[imincost]

            # Goal check.
            if (qubit is not None and mincost < self.epsi) or \
               (qubit is None and bool((ents[imincost] <= self.epsi).all())):
                parent_path = fringe[imincost // num_actions].path
                return BeamSearch.Node(
                    expand_sim.states[imincost].clone(),
                    parent_path + (action_idx[imincost],),
                    mincost,
                )

            # Build successor list and trim to beam.
            next_states = expand_sim.states
            successors = [
                BeamSearch.Node(
                    next_states[i].clone(),
                    fringe[i // num_actions].path + (action_idx[i],),
                    costs_np[i],
                )
                for i in range(batch_size)
            ]
            fringe = self._k_best(successors, self.beam_size)

            if verbose:
                print("Mean cost in current beam:", np.mean([n.cost for n in fringe]))

        return None

    @staticmethod
    def _k_best(nodes, k):
        heap, count = [], 0
        for node in nodes:
            heapq.heappush(heap, (node.cost, count, node))
            count += 1
        return [heapq.heappop(heap)[-1] for _ in range(min(k, count))]


class SearchExpert:
    """Qubit-by-qubit beam search: disentangles one qubit at a time."""

    def __init__(self, beam_size=100, epsi=1e-3):
        self.b = BeamSearch(beam_size, epsi)

    def start(self, psi, env: QEnv, num_iter=10_000, verbose=False):
        """Disentangle `psi` qubit by qubit and return the full action sequence.

        Returns:
            list[int] | None: Sequence of action indices, or None if unsolvable.
        """
        original_actions = dict(env.actions)
        original_num_actions = env.num_actions

        result = self._run(psi, env, num_iter, verbose)

        env.actions = original_actions
        env.num_actions = original_num_actions
        return result

    def _run(self, psi, env, num_iter, verbose):
        psi_t = torch.from_numpy(psi) if isinstance(psi, np.ndarray) else psi.clone()
        actions = []
        curr = psi_t

        for q in range(env.num_qubits - 1):
            path = self.b.start(curr.clone(), env, q, num_iter, verbose)
            if path is None:
                return None
            actions.extend(path)

            # Replay the path to advance the current state.
            replay_sim = VectorizedQState(env.num_qubits, num_envs=1)
            replay_sim.states = curr.unsqueeze(0)
            for a in path:
                replay_sim.apply([env.actions[a]])
            curr = replay_sim.states[0].clone()

            # Remove all actions that involve the now-disentangled qubit q.
            filtered = {k: v for k, v in env.actions.items() if q not in v}
            env.actions = filtered
            env.num_actions = len(filtered)

        return actions


class RandomAgent:
    """Applies uniformly random actions until the state is disentangled."""

    def __init__(self, epsi=1e-3):
        self.epsi = epsi

    def start(self, psi, env: QEnv, num_iters=10_000, verbose=False):
        """Run a random walk from `psi`.

        Returns:
            (list[int], torch.Tensor) | (None, None):
                Action-index sequence and final entanglements, or (None, None).
        """
        psi_t = torch.from_numpy(psi) if isinstance(psi, np.ndarray) else psi
        env.reset()
        env.set_states(psi_t.unsqueeze(0))
        path = []

        for _ in range(num_iters):
            act = int(np.random.randint(0, env.num_actions))
            env.step([act], reset=False)
            path.append(act)
            if bool((env.simulator.entanglements <= self.epsi).all()):
                return path
        return None


class GreedyAgent:
    """At every step selects the action that minimises entanglement entropy."""

    def __init__(self, epsi=1e-3):
        self.epsi = epsi

    def start(self, psi, env: QEnv, qubit=None, num_iter=10_000):
        """Run greedy search from `psi`.

        Args:
            psi (torch.Tensor | np.ndarray): Initial state, shape (2,...,2).
            env (QEnv): Environment defining the action set.
            qubit (int | None): If set, greedily minimise only this qubit's
                entropy; otherwise minimise mean entropy.

        Returns:
            (list[int], torch.Tensor) | (None, None):
                Action-index sequence and final entanglements, or (None, None).
        """
        num_actions = env.num_actions
        action_pairs = [env.actions[a] for a in range(num_actions)]
        path = []

        psi_t = torch.from_numpy(psi) if isinstance(psi, np.ndarray) else psi
        curr = psi_t.clone()

        # Dedicated simulator sized for one-step expansion across all actions.
        # Kept separate from env.simulator to avoid mutating its qubits_order.
        expand_sim = VectorizedQState(env.num_qubits, num_actions)

        for _ in range(num_iter):
            # Broadcast current state across all actions and evaluate them.
            expand_sim.states = curr.unsqueeze(0).expand(num_actions, *curr.shape).clone()
            expand_sim.qubits_order = torch.tile(
                torch.arange(env.num_qubits), (num_actions, 1))
            expand_sim.apply(action_pairs)

            ents = expand_sim.entanglements                     # (A, L)
            costs = ents[:, qubit] if qubit is not None else ents.mean(dim=-1)
            imincost = int(costs.argmin())
            mincost = float(costs[imincost])

            path.append(imincost)
            curr = expand_sim.states[imincost].clone()

            if (qubit is not None and mincost <= self.epsi) or \
               (qubit is None and bool((ents[imincost] <= self.epsi).all())):
                return path

        return None


class UBSearch:
    """Unbiased random search: runs `trials` independent random walks in
    parallel and returns the first one that disentangles the state."""

    def __init__(self, epsi=1e-3, trials=100):
        self.epsi = epsi
        self.trials = trials

    def start(self, psi, env: QEnv, num_iters=10_000, verbose=False):
        """Run parallel random walks from `psi`.

        Returns:
            list[int] | None: First successful action-index sequence, or None.
        """
        sim = VectorizedQState(env.num_qubits, self.trials)
        psi_t = torch.from_numpy(psi) if isinstance(psi, np.ndarray) else psi
        sim.states = psi_t.unsqueeze(0).expand(self.trials, *psi_t.shape).clone()

        paths = []
        for _ in range(num_iters):
            actions = np.random.randint(0, env.num_actions, size=self.trials)
            sim.apply([env.actions[a] for a in actions])
            paths.append(actions)
            disentangled = (sim.entanglements <= self.epsi).all(dim=1)
            if disentangled.any():
                k = int(disentangled.int().argmax())
                return [paths[i][k] for i in range(len(paths))]
        return None
