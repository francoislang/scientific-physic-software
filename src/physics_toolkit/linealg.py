import numpy as np
from numpy.typing import NDArray

from utils.utils import is_inversible, is_square_matrix, is_vector


def is_hermitian(matrix: NDArray[np.complexfloating | np.floating]) -> bool:
    if not is_square_matrix(matrix):
        raise ValueError("Must be a square matrix")

    return np.allclose(matrix, matrix.conj().T)


def eigh_sorted(
    matrix: NDArray[np.complexfloating | np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    if not is_hermitian(matrix):
        raise ValueError("Matrix is not Hermitian")

    eigvalues, eigvecs = np.linalg.eigh(matrix)
    sort_indices = np.argsort(eigvalues)

    return eigvalues[sort_indices], eigvecs[:, sort_indices]


def solver(
    A: NDArray[np.complexfloating | np.floating],
    b: NDArray[np.complexfloating | np.floating],
) -> NDArray[np.complexfloating | np.floating]:
    if not is_square_matrix(A):
        raise ValueError("Matrix A must be square")
    if not is_vector(b):
        raise ValueError("Vector b must be a vector")
    if not is_inversible(A):
        raise ValueError("Matrix A is not invertible")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Matrix A and vector b must have the same size")
    return np.linalg.solve(A, b)
