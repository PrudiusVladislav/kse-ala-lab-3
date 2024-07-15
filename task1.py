import numpy as np

from svd_result import SvdResult


def manual_svd(A):
    AtA = np.dot(A.T, A)
    AAt = np.dot(A, A.T)

    eigvals_AtA, V = np.linalg.eigh(AtA)  # right
    eigvals_AAt, U = np.linalg.eigh(AAt)  # left

    idx_AtA = np.argsort(eigvals_AtA)[::-1]
    eigvals_AtA = eigvals_AtA[idx_AtA]
    V = V[:, idx_AtA]

    idx_AAt = np.argsort(eigvals_AAt)[::-1]
    # eigvals_AAt = eigvals_AAt[idx_AAt]
    U = U[:, idx_AAt]

    singular_values = np.sqrt(eigvals_AtA)

    U_corrected = np.zeros_like(U)
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-10:  # to avoid division by zero for near zero values
            U_corrected[:, i] = (A @ V[:, i]) / singular_values[i]

    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, singular_values)

    return SvdResult(U_corrected, Sigma, V.T)


A = np.array([
    [8, 0, 8],
    [2, 2, 4],
    [3, 9, 9]
])

svd_result = manual_svd(A)

A_reconstructed = np.dot(svd_result.u, np.dot(svd_result.sigma, svd_result.v_t))
A_reconstructed_rounded = np.around(A_reconstructed, decimals=2)
A_reconstructed_rounded[np.abs(A_reconstructed_rounded) < 1e-10] = 0

print("Original Matrix A:\n", A)
print("\nMatrix U:\n", svd_result.u)
print("\nMatrix Sigma:\n", svd_result.sigma)
print("\nMatrix V^T:\n", svd_result.v_t)
print("\nReconstructed Matrix A (U * Sigma * V^T):\n", A_reconstructed_rounded)

assert np.allclose(A, A_reconstructed_rounded), "Test failed! should just delete it or smth..."
print("\nThe reconstruction is correct!")
