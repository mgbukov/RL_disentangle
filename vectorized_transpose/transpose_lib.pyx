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



# load cpp function from header: only variable dtypes listed, no variable names required
cdef extern from "transpose_lib.h":
    void transpose_2d(int, np.complex128_t *, np.complex128_t *) nogil



def swap_axes(int batch_size, int dim, np.complex128_t[::1] array, np.complex128_t[::1] out): 
    """
    
    !!! 
    * memory allocation is always best managed by python, i.e. all variables are preallocated in the python script
    and we pass references to their memory addesses;

    * [::1] above defines a reference to the first element of the memory view in a c-contiguous np.array.  
    
    """
   
    cdef int j,n 
    cdef int array_size = dim*dim

    with nogil: # realeases the gil (i.e., python's global interpereter lock) --> extra speed, but no python objects
        # loop over batch
        for j in prange(batch_size): # prange enables OMP parallelization over the batch, see https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html
            n=j*array_size
            transpose_2d(dim, &array[n], &out[n])



