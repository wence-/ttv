import numpy as np
cimport numpy as np
cdef extern from "ttv.h":
  ctypedef int size_t
  ctypedef double value_t
  void tensor_times_vector_double(size_t, size_t,
                                  value_t *, size_t *, size_t *, size_t *,
                                  value_t *, size_t *,
                                  value_t *)

np.import_array()

def ttv(int q, a, b):
    cdef size_t p = len(a.shape)
    cdef np.ndarray a_, b_, c
    cdef np.ndarray na, wa, pia, nb
    a_ = a
    b_ = b
    na = np.asarray(a.shape)
    wa = np.asarray(a.strides) // a.itemsize
    # C-like (last index varies fastest)
    pia = np.arange(p, 0, step=-1, dtype=np.uint64)
    nb = np.asarray(b.shape)
    # Mode (q) is 1-based
    cshape = np.asarray([s for i, s in enumerate(a.shape) if i != q - 1],
                        dtype=np.uint64)
    c = np.zeros(cshape, dtype=np.float64)
    tensor_times_vector_double(q, p,
                               <value_t *>a_.data,
                               <size_t *>na.data,
                               <size_t *>wa.data,
                               <size_t *>pia.data,
                               <value_t *>b_.data,
                               <size_t *>nb.data,
                               <value_t *>c.data)
    return c
