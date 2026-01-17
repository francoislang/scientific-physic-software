import numpy as np
from numpy.typing import NDArray

def is_hermitian(matrix: NDArray[np.complexfloating | np.floating]) -> bool:
    return np.allclose(matrix, matrix.conj().T)