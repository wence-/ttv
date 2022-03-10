import numpy
from ttv import ttv
A = numpy.arange(24, dtype=numpy.float64).reshape(2, 3, 4)

B = numpy.arange(3, dtype=numpy.float64)

C = ttv(2, A, B)

print(C)

print(numpy.einsum("ijk,j->ik", A, B))
