import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.sparse import diags, bmat, csc_matrix

# Parameters
N = 6  # Grid size (example, adjust as needed)
dx = 0.5  # Grid spacing in x
dy = 0.5  # Grid spacing in y
M = N**2  # Total number of points

block_count=N
# Define diagonals
diagonals = [np.ones(N)/(2*dx), np.zeros(N), -np.ones(N)/(2*dx)]  # +1, 0, -1
offsets = [1, 0, -1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

# Create the NxN block matrix
block = diags(diagonals, offsets, shape=(N, N))

# Convert to a format that allows element modification (csc_matrix allows assignment)
block = csc_matrix(block)

# Modify the top-left element directly
block[0, 0] = -1/dx
block[0, 1] = 1/dx
block[N-1, N-2] = -1/dx
block[N-1, N-1] = 1/dx

# If you want to convert back to a dia_matrix, you can do so:
block = block.todia()

# Create a larger block matrix made of block_count x block_count blocks
large_matrix = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)])
A = large_matrix

# Convert to CSR format for efficiency
A = A.tocsr()

# Visualize the sparse matrix structure
plt.figure(figsize=(8, 8))
plt.spy(A, markersize=5)
plt.title("Structure of the Finite Difference Matrix")
plt.show()

# Convert to dense matrix (only for small grids)
A_dense = A.toarray()
print("Dense Representation of the Matrix:")
print(np.round(A_dense, 2))  # Rounded for clarity