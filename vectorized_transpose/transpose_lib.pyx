# distutils: language=c++
# cython: language_level=2
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False


cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport exp #, sin, cos, acos, sqrt, fabs, M_PI, floor, ceil

from cython.parallel cimport prange, threadid, parallel


DEF _L=6


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def swap_axes(int N, np.ndarray[np.complex128_t, ndim=_L+1] array, np.ndarray[np.uint8_t, ndim=2] transposed_axes, np.ndarray[np.complex128_t, ndim=_L+1] out_array):
    cdef int i;

    #with nogil:
    for i in range(N):
        out_array[i] = array[i].transpose(transposed_axes[i])

    return out_array





