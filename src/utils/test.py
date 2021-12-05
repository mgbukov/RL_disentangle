import numpy as np
from itertools import combinations, permutations
import timeit

from batch_transpose import cy_transpose_arr, cy_transpose_batch


def transpose_batch(batch, qubits_indices, permutation_map):
    for i in range(batch.shape[0]):
        q0, q1 = qubits_indices[i]
        batch[i] = np.transpose(batch[i], permutation_map[q0][q1])
        batch[i] = np.ascontiguousarray(batch[i], dtype=np.complex64)
        assert batch[i].flags['C_CONTIGUOUS']
    return batch


def generate_permutation_maps(L):
    C = (L * (L - 1))
    qubits_permutations = np.zeros((L, L, L), dtype=np.int32)
    qubits_inverse_permutations = np.zeros_like(qubits_permutations)
    for q0, q1 in permutations(range(L), 2):
        sysA = [q0, q1]
        sysB = [q for q in range(L) if q not in sysA]
        P = sysA + sysB
        qubits_permutations[q0, q1] = np.array(P, dtype=np.int32)
        qubits_inverse_permutations[q0, q1] = np.argsort(P).astype(np.int32)
    return qubits_permutations, qubits_inverse_permutations


def random_batch(L, batch_size=1):
    states = np.random.randn(batch_size, 2 ** L) + 1j * np.random.randn(batch_size, 2 ** L)
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    shape = (batch_size,) + (2,) * L
    return states.astype(np.complex64).reshape(shape)


def element_strides(shape):
    strides = np.ones_like(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = shape[i+1] * strides[i+1]
    return strides.astype(np.int32)


def test_transpose_arr():
    import itertools
    L = 7
    shape = np.array([2] * L, dtype=np.int32)
    arr = np.arange(2 ** L, dtype=np.complex64).reshape(shape)
    ndims = L
    size = 2 ** L
    strides = element_strides(shape)

    for permutation in itertools.permutations(range(L)):
        P = np.array(permutation, dtype=np.int32)
        stridesT = strides[P].copy()
        truth = np.transpose(arr, P)
        test = cy_transpose_arr(arr.ravel(), ndims, size, strides, stridesT)
        assert np.all(truth == test.reshape(shape))


def test_transpose_batch():
    batch_size = 4
    L = 8
    shape = (2,) * L
    batch = np.arange(batch_size * (2 ** L), dtype=np.complex64)
    batch = batch.reshape((batch_size,) + shape)

    qubits = list(combinations(range(L), 2))
    permutation_map, _ = generate_permutation_maps(L)
    for i in range(0, len(qubits), batch_size):
        qubits_indices = np.array(qubits[i : i + batch_size], dtype=np.int32)

        if len(qubits_indices) != batch_size:
            continue
        # Naive
        truth = np.zeros_like(batch)
        for i in range(batch_size):
            q0, q1 = qubits_indices[i]
            truth[i] = np.transpose(batch[i], permutation_map[q0][q1])
        # Cython
        test = cy_transpose_batch(batch, qubits_indices, permutation_map)
        test = np.zeros_like(batch)
        cy_transpose_batch(batch, qubits_indices, permutation_map, output=test)
        assert np.all(truth == test)

    batch = random_batch(L, batch_size)
    ntests = 1000
    actions = np.random.randint(low=0, high=L, size=ntests*batch_size*2, dtype=np.int32)
    actions = actions.reshape(-1, 2)
    for i in range(0, len(qubits), batch_size):
        qubits_indices = actions[i:i+batch_size]
        assert len(qubits_indices) == batch_size
        if any(q[0] == q[1] for q in qubits_indices):
            continue
        # Naive
        truth = np.zeros_like(batch)
        for i in range(batch_size):
            q0, q1 = qubits_indices[i]
            truth[i] = np.transpose(batch[i], permutation_map[q0][q1])
        # Cython
        test = cy_transpose_batch(batch, qubits_indices, permutation_map)
        assert np.all(truth == test)
        test = np.zeros_like(batch)
        cy_transpose_batch(batch, qubits_indices, permutation_map, output=test)
        assert np.all(truth == test)


def do_python_transpose(L, batch_size, qubits, permutation_map):
    psi = random_batch(L, batch_size)
    transpose_batch(psi, qubits, permutation_map)


def do_cython_transpose(L, batch_size, qubits, permutation_map, out, strides_buffer):
    psi = random_batch(L, batch_size)
    cy_transpose_batch(psi, qubits, permutation_map,
                        output=out, output_strides_buffer=strides_buffer)


def benchmark():
    import matplotlib.pyplot as plt

    cython_times = []
    python_times = []
    B = 128  # batch size
    for L in range(4, 11):
        permutation_map, _ = generate_permutation_maps(L)
        ACTIONS = np.array([list(x) for x in combinations(range(L), 2)], dtype=np.int32)
        actions = np.random.randint(0, len(ACTIONS) - 1, size=B)
        qubits = np.array([ACTIONS[a] for a in actions])
        output_buffer = random_batch(L, B)
        strides_buffer = np.zeros(output_buffer.ndim - 1, dtype=np.int32)
        ctime = timeit.timeit(
            lambda : do_cython_transpose(L, B, qubits, permutation_map,
                                         out=output_buffer,
                                         strides_buffer=strides_buffer),
            number=1000)

        ptime = timeit.timeit(
            lambda : do_python_transpose(L, B, qubits, permutation_map),
            number=1000)

        print(f'Python time, L={L}:', ptime)
        print(f'Cython time, L={L}:', ctime)
        print()
        python_times.append(ptime)
        cython_times.append(ctime)

    fig, ax = plt.subplots()
    ax.set_title(f'Time for 1000 repetitions, batch_size={B}')
    ax.set_xlabel('L')
    ax.set_ylabel('seconds')
    xs = list(range(4, 11))
    ax.plot(xs, cython_times, label='Cython')
    ax.plot(xs, python_times, label='Python')
    ax.legend()
    fig.savefig('cython-vs-python-compare.pdf')


test_transpose_arr()
test_transpose_batch()
benchmark()
