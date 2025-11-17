# python/tests/test_mac.py
import numpy as np


def dot_i8(a, b):
    return int(np.sum(a.astype(np.int32) * b.astype(np.int32)))


def test_small_example():
    a = np.array([3, -2, 5], dtype=np.int8)
    b = np.array([-1, 4, 6], dtype=np.int8)
    assert dot_i8(a, b) == 19  # sanity: matches hand calc
