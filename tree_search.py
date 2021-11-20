import numpy as np
import heapq
import time
import pickle
import matplotlib.pyplot as plt
import sys

from src.envs.rdm_environment import (QubitsEnvironment,
                                      _random_pure_state,
                                      _ent_entropy)

np.random.seed(34)

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))


def Astar_search(psi, H, G, max_iter=1000, epsi=1e-3, log=sys.stdout):
    L = int(np.log2(len(psi)))
    env = QubitsEnvironment(L, epsi, 1)
    n_actions = env.num_actions
    env._state = np.atleast_2d(psi).copy()
    Q = PriorityQueueWithFunction(lambda item: H(item) + G(item))
    Q.push((env._state.copy(), [], env.Entropy(env._state), 0))
    tick = time.time()
    for i in range(max_iter):
        print_every = 2000
        if (i + 1) % print_every == 0:
            tock = time.time()
            # print(f'Iterations {i - print_every + 1}-{i+1} took {tock - tick:.3f}sec')
            tick = time.time()
        best_node, best_path, current_ent, _ = Q.pop()
        env._state = best_node.copy()
        for a in range(n_actions):
            if len(best_path) and a == best_path[-1]:
                continue
            child = env.next_state(a)
            if env.Disentangled(child, epsi):
                return child, best_path + [a]
            child_ent = env.Entropy(child)
            Q.push((child, best_path + [a], child_ent, current_ent-child_ent))
    # print("max depth reached:", max(Q.heap, key=lambda item: len(item[2][1])))
    # print('First\n', Q.heap[0], file=log)
    # print('Last\n', Q.heap[-1], file=log)
    item = Q.heap[0][-1]
    return item[0], item[1]


def beam_search(psi, C, beam_size=32, max_depth=1000, epsi=1e-3, logfile=sys.stdout):
    L = int(np.log2(len(psi)))
    E = QubitsEnvironment(L, epsi)
    beam, temp = [(np.atleast_2d(psi).copy(), [])], []
    i = 0
    while (i < max_depth):
        i += 1
        tick = time.time()
        for state, path in beam:
            E.state = np.atleast_2d(state).copy()
            for a in range(E.num_actions):
                child = E.next_state(a)
                if E.Disentangled(np.atleast_2d(child), epsi)[0]:
                    return (child, path + [a])
                temp.append((child, path + [a]))
        temp = sorted(temp, key=C)
        beam = temp[:beam_size]
        # tock = time.time()
        # print(f'Expansion {i} took {tock-tick:.1f} seconds')
        beam_states = [x[0] for x in beam]
        beam_entropies = [[_ent_entropy(s, [i], L) for i in range(L)]
                                for s in beam_states]
        print(f'Mean entropies in beam {i+1}:',
              np.mean(beam_entropies, axis=0).round(3).ravel(),
              file=logfile)
    return min(beam, key=C)


def beam_cost(item):
    state, _ = item
    L = int(np.log2(state.shape[1]))
    entropies = [_ent_entropy(state, [i], L) for i in range(L)]
    return np.mean(entropies)

def beam_logcost(item):
    state, _ = item
    L = int(np.log2(state.shape[1]))
    entropies = [_ent_entropy(state, [i], L) for i in range(L)]
    return np.mean(np.log2(np.maximum(entropies, 1e-3)))


def bfs(psi, max_iter):
    return Astar_search(psi, lambda item: len(item[1]), lambda item: 0, max_iter)


def do_astar_search(psi, max_iter=1000, log=sys.stdout):
    L = int(np.log2(len(psi)))
    def G(item):
        entropies = item[2]
        result = 0
        for t in thresholds:
            result += np.count_nonzero(entropies < t)
        return len(thresholds) * L - result
    H = lambda item: len(item[1])
    if L <= 5:
        thresholds = np.array([0.3, 0.1, 0.03, 0.01, 0.003, 0.001])
    else:
        thresholds = np.linspace(0.6, 0.001, 25)
    # threshols = np.log2(np.linspace(1.5, 1, 8))
    # thresholds = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001])
    # G = lambda item: 7 - int(10 * np.mean(item[3]))
    final_state, final_path = Astar_search(psi, H, G, max_iter, log=log)
    return final_state, final_path


# L = 6
# solutions = []
# logfile = open('6qubit-log.txt', mode='w')
# for j in range(12):
#     psi = _random_pure_state(L)
#     logfile.write( '\n' + 80 * '=' + '\n' + f'STATE {j}'.center(80) + '\n' + 80 * '=' + '\n')
#     best_path = therealdeal(psi, 200_000, log=logfile)
#     solutions.append((psi, best_path))
#     with open('6qubit-solutions.pkl', mode='wb') as f:
#         print('Saving...')
#         time.sleep(5)
#         pickle.dump(solutions, f)
#         print('Done saving...')
# logfile.close()
# print(solutions)



# 2. Plot each qubit's entropy during rollout
# *****************************************************************************
# L = 5
# for j in range(1):
#     psi = _random_pure_state(L)
#     path = therealdeal(psi, 10_000)
#     # path = [9, 6, 8, 4, 1, 9, 13, 14, 6, 13, 11, 4, 13, 14, 8, 4, 5, 14, 11, 13, 4, 7, 11, 14, 2, 13, 8, 11, 3, 14, 0, 13, 8, 5, 2, 0, 9, 7, 13, 14, 5, 14]
#     E = QubitsEnvironment(L, epsi=1e-3)
#     E.state = psi
#     entropies = np.zeros((len(path), L), dtype=np.float32)
#     for i, a in enumerate(path):
#         E.step(a)
#         entropies[i] = E.Entropy(E._state).ravel()
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.set_title('Entropies during rollout')
#     for q in range(L):
#         ax.plot(entropies[:, q], linewidth=1, alpha=0.5, label=q)
#     ax.legend()
#     fig.savefig('5qubits_rollout_entropies2.pdf')

# L = 7
# psi = _random_pure_state(L)
# final_state, final_path = beam_search(psi, beam_logcost, 32, 100)
# # final_path = [4, 14, 11, 13, 8, 4, 14, 13, 11, 4, 6, 13, 9, 14, 2, 13, 8, 4, 14, 10, 7, 13, 4, 14, 9, 13, 0, 8, 11, 14, 4, 13, 11, 8, 12, 14, 4, 11, 14, 5, 11, 13, 8, 4, 11, 14, 2, 4, 12, 11, 14, 8, 3, 14, 13, 11, 4, 14, 8, 11, 4, 6, 13, 12]
# E = QubitsEnvironment(L, 1e-3)
# E.state = psi.copy()
# for a in final_path:
#     E.state = E.next_state(a)
#     # E.step(a)

# print('\n\n', '=' * 80)
# print('Final path length:', len(final_path))
# print('Final path:', final_path)
# # [4, 14, 11, 13, 8, 4, 14, 13, 11, 4, 6, 7, 13, 14, 9, 4, 1, 13,
# #  8, 4, 14, 5, 11, 13, 7, 4, 11, 14, 4, 8, 13, 14, 11, 4, 13, 8,
# #  4, 11, 7, 8, 13, 14, 4, 13, 1, 11, 14, 5, 11, 13, 4, 14, 11, 13,
# #  14, 8, 4, 11, 6, 3, 12, 14, 0]
# print('Disentangled ?:', E.Disentangled(E.state, 1e-3))
# print('Final entropies:', E.Entropy(E.state))