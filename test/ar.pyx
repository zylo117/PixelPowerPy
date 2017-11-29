from cpython cimport array
import array
cdef array.array a = array.array('i', [1, 2, 3])
cdef str[:] ca = a

cdef int overhead(object a):
    cdef str[:] ca = a
    return ca[1]

cdef int no_overhead(str[:] ca):
    return ca[1]

print overhead(a)  # new memory view will be constructed, overhead
print no_overhead(ca)  # ca is already a memory view, so no overhead
