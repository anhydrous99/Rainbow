import numpy as np
cimport numpy as np

cdef unique(np.ndarray[np.int64_t, ndim=1] sorted_array):
    if len(sorted_array) == 1:
        return sorted_array
    left = sorted_array[:-1]
    right = sorted_array[1:]
    cdef np.ndarray uniques = np.append(right != left, True)
    return sorted_array[uniques]

cdef class SumSegmentTree:
    cdef int _capacity
    cdef np.ndarray _value

    def __init__(self, int capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = np.full(capacity * 2, 0.0, dtype=np.float32)

    def __setitem__(self, int idx, float val):
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int) or isinstance(idxs, np.int32):
            idxs = np.array([idxs])
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            self._value[idxs] = self._value[2 * idxs] + self._value[2 * idxs + 1]
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]

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
        assert np.max(prefixsum) <= self.sum() + 1.0e-5

        cdef np.ndarray idx = np.ones(len(prefixsum), dtype=np.int)
        cdef np.ndarray cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(self._value[idx] <= prefixsum, prefixsum - self._value[idx],
                                                     prefixsum)
            idx = np.where(np.logical_or(self._value[idx] > prefixsum, np.logical_not(cont)), idx, idx + 1)
            prefixsum = prefixsum_new
            cont = idx < self._capacity
        return idx - self._capacity

cdef class MinSegmentTree:
    cdef int _capacity
    cdef np.ndarray _value

    def __init__(self, int capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = np.full(capacity * 2, float('inf'), dtype=np.float32)

    def __setitem__(self, int idx, float val):
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int):
            idxs = np.array([idxs])
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            self._value[idxs] = np.minimum(self._value[2 * idxs], self._value[2 * idxs + 1])
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]

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
