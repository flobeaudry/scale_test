import numpy as np
from scipy.sparse import diags, bmat


def create_sparse_matrix_dx(N):
    block_count=N
    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [-np.ones(N), np.zeros(N), np.ones(N)]  # +1, 0, -1
    offsets = [1, 0, -1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

    # Create a single NxN block
    block = diags(diagonals, offsets, shape=(N, N))

    # Create a larger block matrix made of block_count x block_count blocks
    large_matrix = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)])

    return large_matrix


def create_sparse_matrix_dy(N):
    # Size of the sparse matrix
    size = N * N

    # Create the diagonals for the top-left to bottom-right direction
    diagonals = [-np.ones(size - N), np.ones(size - N)]
    
    # Offset positions for the diagonals
    offsets = [N, -N]  # N for +1, -N for -1

    # Create the sparse diagonal matrix
    sparse_matrix = diags(diagonals, offsets, shape=(size, size), format='csr')

    return sparse_matrix



def create_sparse_double_matrix_dydx(N, dx=1, dy=1):
    # Size of the sparse matrix
    size = N * N
    
    # Create the sparse matrix for dy (top-left to bottom-right direction)
    diagonals_dy = [-(1/dy)*np.ones(size - N), (1/dy)*np.ones(size - N)]
    offsets_dy = [N, -N]  # N for +1, -N for -1
    sparse_matrix_dy = diags(diagonals_dy, offsets_dy, shape=(size, size), format='csr')

    # Create the sparse matrix for dx (x-direction, block diagonal matrix)
    block_count = N
    diagonals_dx = [-(1/dx)*np.ones(N), np.zeros(N), (1/dx)*np.ones(N)]  # +1, 0, -1
    offsets_dx = [1, 0, -1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

    # Create a single NxN block
    block = diags(diagonals_dx, offsets_dx, shape=(N, N))

    # Create a larger block matrix made of block_count x block_count blocks
    sparse_matrix_dx = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)], format='csr')

    # Combine the two sparse matrices by adding them together
    combined_sparse_matrix = sparse_matrix_dy + sparse_matrix_dx

    return combined_sparse_matrix


def create_sparse_double_matrix_dxdy(N, dx=1, dy=1):
    # Size of the sparse matrix
    size = N * N
    
    # Create the sparse matrix for dy (top-left to bottom-right direction)
    diagonals_dy = [-(1/dx)*np.ones(size - N), (1/dx)*np.ones(size - N)]
    offsets_dy = [N, -N]  # N for +1, -N for -1
    sparse_matrix_dy = diags(diagonals_dy, offsets_dy, shape=(size, size), format='csr')

    # Create the sparse matrix for dx (x-direction, block diagonal matrix)
    block_count = N
    diagonals_dx = [-(1/dy)*np.ones(N), np.zeros(N), (1/dy)*np.ones(N)]  # +1, 0, -1
    offsets_dx = [1, 0, -1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

    # Create a single NxN block
    block = diags(diagonals_dx, offsets_dx, shape=(N, N))

    # Create a larger block matrix made of block_count x block_count blocks
    sparse_matrix_dx = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)], format='csr')

    # Combine the two sparse matrices by adding them together
    combined_sparse_matrix = sparse_matrix_dy + sparse_matrix_dx

    return combined_sparse_matrix