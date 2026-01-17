import numpy as np
from numpy.typing import NDArray


def is_hermitian(matrix: NDArray[np.complexfloating | np.floating]) -> bool:
    return np.allclose(matrix, matrix.conj().T)

def eigh_sorted(matrix: NDArray[np.complexfloating | np.floating]) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    if not is_hermitian(matrix):
        raise ValueError("Matrix is not Hermitian")
    eigvalues, eigvecs = np.linalg.eigh(matrix)
    sort_indices = np.argsort(eigvalues)
    return eigvalues[sort_indices], eigvecs[:, sort_indices]