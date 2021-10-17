# distutils: language=c++
# cython: language_level=2
# cython: boundscheck=False  
# cython: wraparound=False   
# cython: cdivision=True
# cython: profile=False      


cimport cython
import numpy as np
cimport numpy as np
#from libcpp.vector cimport vector
#from libcpp cimport bool
#from libc.stdlib cimport rand, srand, RAND_MAX
#from libc.math cimport exp , sin, cos, acos, sqrt, fabs, M_PI, floor, ceil

from cython.parallel cimport prange, threadid, parallel



# load cpp function from header
cdef extern from "transpose_lib.h":
    void transpose_2d(np.complex128_t *, int , np.complex128_t *) nogil


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def swap_axes(int batch_size, int dim, np.complex128_t[::1] array, np.complex128_t[::1] out): #np.ndarray[np.complex128_t, ndim=1+_L, mode='c']
   
    cdef int j,n 

    with nogil:
        for j in range(batch_size): # change to prange to enable OMP parallelization
            n=j*dim*dim
            transpose_2d(&array[n], dim, &out[n])



