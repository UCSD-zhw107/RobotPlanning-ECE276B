from pqdict import pqdict



open_heap = pqdict().minpq()

point = (1.2, 3, 0.0)
f = 2

open_heap[point] = (f, point)

print(open_heap.popitem())
