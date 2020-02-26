import numpy as np
cimport numpy as np

cdef class BaseTree:
    cdef int _capacity
    cdef np.ndarray _value

    def __init__(self, int capacity, float fill_value):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2."
        assert fill_value >= 0, "Fill values must be positive."
        self._capacity = capacity
        self._value = np.full(capacity * 2, fill_value, dtype=np.float32)

    def __getitem__(self, np.ndarray[long, ndim=1] idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]

    cdef _calc_indexes_update(self, long idx, float value):
        idx += self._capacity
        self._value[idx] = value
        return idx // 2

cdef class SumSegmentTree(BaseTree):
    def __init__(self, int capacity):
        super(SumSegmentTree, self).__init__(capacity, 0.0)

    def __setitem__(self, long idx, float val):
        idx = self._calc_indexes_update(idx, val)
        while idx > 0:
            self._value[idx] = self._value[2 * idx] + self._value[2 * idx + 1]
            idx = idx // 2

    cdef _reduce_helper(self, int start, int end, int node, int node_start, int node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        cdef int mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._reduce_helper(start, mid, 2 * node, node_start, mid) + \
                       self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)

    cdef reduce(self, int start=0, int end=-2147483647):
        if end is -2147483647:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    cpdef sum(self, int start=0, int end=-2147483647):
        return self.reduce(start, end)

    def find_prefix_sum_idx(self, np.ndarray[double, ndim=1] prefixsum):
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1.0e-5

        cdef np.ndarray idx = np.ones(len(prefixsum), dtype=np.int)
        cdef np.ndarray cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(self._value[idx] <= prefixsum, prefixsum - self._value[idx], prefixsum)
            idx = np.where(np.logical_or(self._value[idx] > prefixsum, np.logical_not(cont)), idx, idx + 1)
            prefixsum = prefixsum_new
            cont = idx < self._capacity
        return idx - self._capacity

cdef class MinSegmentTree(BaseTree):
    def __init__(self, int capacity):
        super(MinSegmentTree, self).__init__(capacity, float('inf'))

    def __setitem__(self, long idx, float val):
        idx = self._calc_indexes_update(idx, val)
        while idx > 0:
            self._value[idx] = np.minimum(self._value[2 * idx], self._value[2 * idx + 1])
            idx = idx // 2

    cdef _reduce_helper(self, int start, int end, int node, int node_start, int node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        cdef int mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return np.minimum(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
            )

    cdef reduce(self, int start=0, int end=-2147483647):
        if end is -2147483647:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def min(self, int start=0, int end=-2147483647):
        return self.reduce(start, end)
