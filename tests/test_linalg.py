import numpy as np
import pytest
from physics_toolkit.linealg import is_hermitian

def test_is_hermitian_standard():
    mat = np.array([[2, 3 +1j], [3 -1j, 4]])
    assert is_hermitian(mat) is True

def test_is_hermitian_real_symmetric():
    mat = np.array([[2, 3], [3, 4]])
    assert is_hermitian(mat) is True

def test_not_hermitian_complex_diagonal():
    mat = np.array([[2, 3 +1j], [3 -1j, 4j]])
    assert is_hermitian(mat) is False

def test_not_hermitian_symmetric_complex():
    mat = np.array([[2, 3j], [3j, 4]])
    assert is_hermitian(mat) is False

