import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from physics_toolkit.linealg import (
    eigh_sorted,
    is_hermitian,
    solver,
)
from utils.utils import is_inversible

# --- Tests Unitaires Classiques (Exemples fixes) ---


def test_is_hermitian_standard():
    mat = np.array([[2, 3 + 1j], [3 - 1j, 4]])
    assert is_hermitian(mat) is True


def test_is_hermitian_real_symmetric():
    mat = np.array([[2, 3], [3, 4]])
    assert is_hermitian(mat) is True


def test_not_hermitian_complex_diagonal():
    mat = np.array([[2, 3 + 1j], [3 - 1j, 4j]])
    assert is_hermitian(mat) is False


def test_not_hermitian_symmetric_complex():
    mat = np.array([[2, 3j], [3j, 4]])
    assert is_hermitian(mat) is False


# --- Stratégies Hypothesis ---

# Stratégie pour générer des matrices carrées complexes de taille 2x2 à 10x10
complex_matrix = arrays(
    dtype=np.complex128,
    shape=st.integers(min_value=2, max_value=10).map(lambda n: (n, n)),
    elements=st.complex_numbers(
        min_magnitude=0, max_magnitude=1e6, allow_nan=False, allow_infinity=False
    ),
)

# Stratégie pour générer des matrices non-carrées (lignes != colonnes)
non_square_matrix = arrays(
    dtype=np.complex128,
    shape=st.tuples(
        st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10)
    ).filter(lambda shape: shape[0] != shape[1]),
    elements=st.complex_numbers(
        min_magnitude=0, max_magnitude=1e6, allow_nan=False, allow_infinity=False
    ),
)

# Stratégie pour générer des matrices carrées et inversibles
# Filtre les matrices carrées aléatoires pour ne conserver que celles qui sont inversibles.
invertible_matrix = complex_matrix.filter(lambda mat: is_inversible(mat))

# Stratégie pour générer des matrices carrées non-inversibles (singulières)
# Filtre les matrices carrées aléatoires pour ne conserver que celles qui sont non-inversibles.
non_invertible_matrix = complex_matrix.filter(lambda mat: not is_inversible(mat))


# --- Tests pour eigh_sorted et is_hermitian ---


@given(complex_matrix)
def test_prop_constructed_hermitian_is_true(mat):
    """
    Propriété : Pour toute matrice carrée M, H = M + M.H est hermitienne.
    """
    hermitian_mat = mat + mat.conj().T
    assert is_hermitian(hermitian_mat) is True


@given(complex_matrix)
def test_prop_modified_hermitian_is_false(mat):
    """
    Propriété : Si on modifie un élément hors-diagonale d'une matrice hermitienne
    sans modifier son symétrique, elle n'est plus hermitienne.
    """
    hermitian_mat = mat + mat.conj().T
    assume(hermitian_mat.shape[0] > 1)
    row, col = 0, 1
    val = hermitian_mat[row, col]
    perturbation = 1.0 + 0.5 * np.abs(val)
    hermitian_mat[row, col] += perturbation
    assert is_hermitian(hermitian_mat) is False


@given(complex_matrix)
def test_mathematical_properties_eigh_sorted(mat):
    """
    Vérifie les propriétés mathématiques fondamentales de la décomposition en valeurs propres.
    Pour une matrice hermitienne H, H * V = V * D, où V sont les vecteurs propres et D les valeurs propres.
    """
    # Important : On symétrise d'abord pour avoir une matrice hermitienne valide
    H = mat + mat.conj().T
    eigvals, eigvecs = eigh_sorted(H)
    lambda_matrix = np.diag(eigvals)
    left_side = H @ eigvecs
    right_side = eigvecs @ lambda_matrix
    assert np.allclose(left_side, right_side, atol=1e-8)


@given(complex_matrix)
def test_sorting_consistency_eigh_sorted(mat):
    """
    Vérifie que les valeurs propres retournées sont triées de manière croissante
    et que la relation H * V = V * D est maintenue après le tri.
    """
    H = mat + mat.conj().T
    eigvals, eigvecs = eigh_sorted(H)
    # 1. Vérifie que les valeurs sont triées croissantes
    assert np.all(np.diff(eigvals) >= -1e-15)
    # 2. Vérifie la cohérence vecteurs/valeurs
    assert np.allclose(H @ eigvecs, eigvecs @ np.diag(eigvals), atol=1e-8)


@given(complex_matrix)
def test_real_values_eigh_sorted(mat):
    """
    Vérifie que les valeurs propres d'une matrice hermitienne sont bien réelles.
    """
    H = mat + mat.conj().T
    eigvals, _ = eigh_sorted(H)
    assert np.allclose(eigvals.imag, 0, atol=1e-8)


@given(complex_matrix)
def test_orthonormal_eigh_sorted(mat):
    """
    Vérifie que les vecteurs propres d'une matrice hermitienne sont orthonormaux.
    """
    H = mat + mat.conj().T
    _, eigvecs = eigh_sorted(H)
    product = eigvecs.conj().T @ eigvecs
    identity = np.eye(product.shape[0])
    assert np.allclose(product, identity, atol=1e-8)


def test_degenerated_eigh_sorted():
    """
    Vérifie l'orthogonalité des vecteurs propres même en présence de valeurs propres dégénérées.
    """
    # test à la main car hypothesis fournit rarement des matrices
    # avec des valeurs propres identiques
    mat = np.diag([1.0, 1.0, 3.0])
    _, eigvecs = eigh_sorted(mat)
    assert np.allclose(eigvecs.conj().T @ eigvecs, np.eye(3), atol=1e-8)


# --- Tests pour solver ---


@given(invertible_matrix)
def test_solver_solves_linear_system_correctly(A):
    """
    Vérifie que le solveur trouve la bonne solution pour un système linéaire
    avec une matrice carrée et inversible.
    """
    b = np.random.rand(A.shape[0]) + 1j * np.random.rand(A.shape[0])
    b = b.astype(np.complex128)
    solution = solver(A, b)
    assert np.allclose(A @ solution, b, atol=1e-8)


@given(non_square_matrix)
def test_solver_non_square_matrix_raises_error(mat):
    """Vérifie que solver lève une ValueError pour une matrice non-carrée."""
    b = np.ones(mat.shape[0], dtype=np.complex128)
    with pytest.raises(ValueError, match="Matrix A must be square"):
        solver(mat, b)


@given(complex_matrix)
def test_solver_mismatched_dimensions_raises_error(mat):
    """Vérifie que solver lève une ValueError pour un vecteur b de dimension incompatible."""
    b = np.ones(mat.shape[0] + 1, dtype=np.complex128)
    with pytest.raises(
        ValueError, match="Matrix A and vector b must have the same size"
    ):
        solver(mat, b)


@given(non_invertible_matrix)
def test_solver_singular_matrix_raises_error(A):
    """
    Vérifie que le solveur lève une ValueError pour une matrice singulière (non-inversible).
    """
    b = np.random.rand(A.shape[0]) + 1j * np.random.rand(A.shape[0])
    b = b.astype(np.complex128)
    with pytest.raises(ValueError, match="Matrix A is not invertible"):
        solver(A, b)