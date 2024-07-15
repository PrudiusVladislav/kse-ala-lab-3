import numpy as np

from svd_result import SvdResult


def manual_svd(a: np.ndarray):
    at_a = a.T @ a
    a_at = a @ a.T

    eigvals_at_a, v = np.linalg.eigh(at_a)  # right
    eigvals_a_at, u = np.linalg.eigh(a_at)  # left

    idx_at_a = np.argsort(eigvals_at_a)[::-1]
    eigvals_at_a = eigvals_at_a[idx_at_a]
    eigvals_at_a[eigvals_at_a < 0] = 0
    v = v[:, idx_at_a]

    # idx_a_at = np.argsort(eigvals_a_at)[::-1]
    # eigvals_AAt = eigvals_AAt[idx_AAt]
    # u = u[:, idx_a_at]

    singular_values = np.sqrt(eigvals_at_a)

    u_corrected = np.zeros_like(u)
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-10:  # to avoid division by zero for near zero values
            u_corrected[:, i] = (a @ v[:, i]) / singular_values[i]

    sigma = np.zeros_like(a, dtype=float)
    np.fill_diagonal(sigma, singular_values)

    return SvdResult(u_corrected, sigma, v.T)


A = np.array([
    [8, 0, 8, 10],
    [2, 2, 4, 5],
    [3, 9, 9, 3]
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
