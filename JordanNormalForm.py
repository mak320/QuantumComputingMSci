import numpy as np

# Define your matrix
A = np.array([[1, 1, 0],
              [0, 2, 0],
              [0, 1, 1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Construct the Jordan normal form
n = len(A)
J = np.zeros_like(A, dtype=float)

for i in range(n):
    J[i, i] = eigenvalues[i]

    # Check if eigenvalue is repeated
    if i < n - 1 and np.isclose(eigenvalues[i], eigenvalues[i + 1]):
        J[i, i + 1] = 1
        J[i + 1, i] = 1

# Diagonalize using eigenvectors
P = eigenvectors
P_inv = np.linalg.inv(P)

# Verify the relationship A = P J P^{-1}
verification_result = np.allclose(A, P @ J @ P_inv)

# Print the Jordan normal form, the matrix of eigenvectors, and the verification result
print("Jordan Normal Form:")
print(J)
print("\nMatrix of Eigenvectors P:")
print(P)
print("\nInverse of Matrix of Eigenvectors P:")
print(P_inv)
print("\nVerification Result (A = P J P^{-1}):", verification_result)

