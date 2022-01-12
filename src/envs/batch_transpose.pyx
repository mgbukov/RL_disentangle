import numpy
cimport cython
cimport numpy as np
cimport cpython.array
from cython cimport view

np.import_array()
# distutils: language=c++


def cy_transpose_batch(batch, qubits_indices, permutation_map,
                       output=None, output_strides_buffer=None):

    cdef float complex[::1] in_flat_batch = batch.ravel()
    cdef float complex[::1] out_flat_batch
    if output is None:
        out_flat_batch = numpy.zeros_like(in_flat_batch)
    else:
        out_flat_batch = output.ravel()
    cdef int[:, :, ::1] permutations = permutation_map
    cdef int[:, ::1] qubits = qubits_indices
    cdef int ndims = batch.ndim - 1
    cdef int[::1] strides
    strides = (numpy.array(batch.strides[1:]) // batch.itemsize).astype(numpy.int32)
    cdef int batch_size = batch.shape[0]
    cdef int subarray_size = batch[0].size
    cdef int[::1] out_strides_buffer
    if output_strides_buffer is None:
        out_strides_buffer = numpy.zeros(ndims, dtype=numpy.int32)
    else:
        out_strides_buffer = output_strides_buffer

    _cy_transpose_batch(
        in_flat_batch,
        out_flat_batch,
        permutations,
        qubits,
        ndims,
        strides,
        batch_size,
        subarray_size,
        out_strides_buffer
    )
    if output is None:
        output = numpy.frombuffer(out_flat_batch, dtype=numpy.complex64)
        return output.reshape(batch.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cy_transpose_batch(float complex[::1] in_batch,
                              float complex[::1] out_batch,
                              int[:, :, ::1] permutation_map,
                              int[:, ::1] qubits_indices,
                              int ndims,
                              int[::1] strides,
                              int batch_size,
                              int subarray_size,
                              # Temporary buffers
                              int[::1] out_strides) nogil:
    cdef int[::1] pview
    cdef int q0 = 0
    cdef int q1 = 0
    cdef int i = 0
    for i in range(batch_size):
        # Get transpose permutation for current subarray
        q0 = qubits_indices[i][0]
        q1 = qubits_indices[i][1]
        pview = permutation_map[q0][q1]
        # Permute strides
        for j in range(ndims):
            out_strides[j] = strides[pview[j]]
        # Write the transposed elements in ``out_batch``
        _cy_transpose_arr(
            in_batch[i * subarray_size : (i+1) * subarray_size],
            out_batch[i * subarray_size : (i+1) * subarray_size],
            ndims,
            subarray_size,
            strides,
            out_strides)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cy_transpose_arr(float complex[::1] in_arr,
                            float complex[::1] out_arr,
                            int ndims,
                            int size,
                            int[::1] in_strides,
                            int[::1] out_strides) nogil:

    cdef int j = 0
    cdef int l = 0
    cdef int x = 0
    for i in range(size):
        # Unravel index in `temp`
        # Get index after transpose
        j = i
        l = 0
        for k in range(ndims):
            x = j / in_strides[k]
            l += out_strides[k] * x
            j -= (x * in_strides[k])
        out_arr[i] = in_arr[l]


# Defined for testing only
# ---------------------------------------------------------------------------- #

def cy_transpose_arr(in_arr, ndims, size, in_strides, out_strides):
    out_arr = numpy.zeros_like(in_arr)
    _cy_transpose_arr(in_arr, out_arr, ndims, size, in_strides, out_strides)
    return out_arr