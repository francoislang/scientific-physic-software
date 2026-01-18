import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from physics_toolkit.linealg import eigh_sorted, is_hermitian

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


# --- Tests basés sur des propriétés (Hypothesis) ---

# Stratégie pour générer des matrices carrées complexes de taille 2x2 à 10x10
# On limite la magnitude des éléments pour éviter les problèmes d'overflow (inf)
# qui faussent les tests de logique algébrique pure.
complex_matrices = arrays(
    dtype=np.complex128,
    shape=st.integers(min_value=2, max_value=10).map(lambda n: (n, n)),
    elements=st.complex_numbers(
        min_magnitude=0, max_magnitude=1e6, allow_nan=False, allow_infinity=False
    ),
)


@given(complex_matrices)
def test_prop_constructed_hermitian_is_true(mat):
    """
    Propriété : Pour toute matrice carrée M, H = M + M. H est hermitienne.
    """
    # Construction d'une matrice hermitienne
    hermitian_mat = mat + mat.conj().T

    assert is_hermitian(hermitian_mat) is True


@given(complex_matrices)
def test_prop_modified_hermitian_is_false(mat):
    """
    Propriété : Si on modifie un élément hors-diagonale d'une matrice hermitienne
    sans modifier son symétrique, elle n'est plus hermitienne.
    """
    # 1. On part d'une matrice propre
    hermitian_mat = mat + mat.conj().T

    # 2. On choisit une position hors diagonale (0, 1) car min size est 2x2
    row, col = 0, 1

    # 3. On perturbe cette valeur de manière significative
    val = hermitian_mat[row, col]
    perturbation = 1.0 + 0.5 * np.abs(val)
    hermitian_mat[row, col] += perturbation

    # Maintenant H[0,1] != conj(H[1,0]), donc ce n'est plus hermitien
    assert is_hermitian(hermitian_mat) is False


@given(complex_matrices)
def test_mathematical_properties_eigh_sorted(mat):
    # Important : On symétrise d'abord pour avoir une matrice hermitienne valide
    H = mat + mat.conj().T

    eigvals, eigvecs = eigh_sorted(H)

    # Créer une matrice diagonale de valeurs propres (lambda)
    lambda_matrix = np.diag(eigvals)

    # Calculer H @ eigvecs (et non mat @ eigvecs)
    left_side = H @ eigvecs

    # Calculer eigvecs @ lambda_matrix
    right_side = eigvecs @ lambda_matrix

    assert np.allclose(left_side, right_side, atol=1e-8)


@given(complex_matrices)
def test_sorting_consistency_eigh_sorted(mat):
    # On symétrise
    H = mat + mat.conj().T

    eigvals, eigvecs = eigh_sorted(H)

    # 1. Vérifie que les valeurs sont triées croissantes
    assert np.all(np.diff(eigvals) >= -1e-15)

    # 2. Vérifie la cohérence vecteurs/valeurs On utilise la forme matricielle sûre
    assert np.allclose(H @ eigvecs, eigvecs @ np.diag(eigvals), atol=1e-8)


@given(complex_matrices)
def test_real_values_eigh_sorted(mat):
    H = mat + mat.conj().T

    eigvals, _ = eigh_sorted(H)

    assert np.allclose(eigvals.imag, 0, atol=1e-8)


@given(complex_matrices)
def test_orthonormal_eigh_sorted(mat):
    H = mat + mat.conj().T
    _, eigvecs = eigh_sorted(H)

    product = eigvecs.conj().T @ eigvecs
    identity = np.eye(product.shape[0])

    assert np.allclose(product, identity, atol=1e-8)


# test à la main car hypothesis fournit rarement des matrices
# avec des valeurs propres identiques
def test_degenerated_eigh_sorted():
    mat = np.diag([1.0, 1.0, 3.0])
    _, eigvecs = eigh_sorted(mat)

    # Orthonormalité même en cas dégénéré
    assert np.allclose(eigvecs.conj().T @ eigvecs, np.eye(3), atol=1e-8)
