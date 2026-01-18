import numpy as np
from numpy.typing import NDArray


def is_square_matrix(matrix: NDArray[np.complexfloating | np.floating]) -> bool:
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]


def is_vector(vector: NDArray[np.complexfloating | np.floating]) -> bool:
    return vector.ndim == 1
