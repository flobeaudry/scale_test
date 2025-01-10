import numpy as np
from scipy.sparse import diags, bmat, csc_matrix, eye


def create_sparse_matrix_dx_og_works(N):
    block_count=N
    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [-np.ones(N), np.zeros(N), np.ones(N)]  # +1, 0, -1
    offsets = [1, 0, -1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

    # Create a single NxN block
    block = diags(diagonals, offsets, shape=(N, N))

    # Create a larger block matrix made of block_count x block_count blocks
    large_matrix = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)])

    return large_matrix


def create_sparse_matrix_dx_oldok(N):
    dx, dy = 1, 1
    block_count=N
    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [np.ones(N)/(2*dx), np.zeros(N), -np.ones(N)/(2*dx)]  # +1, 0, -1
    offsets = [1, 0, -1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal
    

    # Create the NxN block matrix
    block = diags(diagonals, offsets, shape=(N, N))

    # Convert to a format that allows element modification (csc_matrix allows assignment)
    block = csc_matrix(block)

    # Modify the top-left element directly
    #block[0, 0] = -1/dx
    #block[0, 1] = 1/dx
    #block[N-1, N-2] = -1/dx
    #block[N-1, N-1] = 1/dx

    # If you want to convert back to a dia_matrix, you can do so:
    block = block.todia()

    # Create a larger block matrix made of block_count x block_count blocks
    large_matrix = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)])

    return large_matrix

def create_sparse_matrix_dx(N, dx=1):
    block_count = N

    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [-np.ones(N) / dx, np.ones(N) / dx]  # -1/dx on main, +1/dx on super diagonal
    offsets = [0, 1]  # Main diagonal, super diagonal
    
    block = diags(diagonals, offsets, shape=(N, N), format="csc")

    # Replace the last row of each block with [-1, 0,...0] to introduce the static -1 behavior
    block = block.toarray()
    block[-1, :] = 0  # Reset the last row
    block[-1, -1] = -1
    block = csc_matrix(block)  # Convert back to sparse

    # Create the large block matrix structure
    blocks = []
    for i in range(block_count):
        row = []
        for j in range(block_count):
            if i == j:  # Diagonal block
                row.append(block)
            else:  # Off-diagonal blocks are zeros
                row.append(None)  # Sparse zero blocks
        blocks.append(row)

    large_matrix = bmat(blocks, format="csc")    
    
    return large_matrix

def create_sparse_matrix_dx_bndyConditionsNotImplicit(N):
    dx, dy = 1, 1
    block_count=N
    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [np.ones(N)/(dx), -np.ones(N)/(dx)]  # +1, 0, -1
    offsets = [1, 0]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal
    
    # Create the NxN block matrix
    block = diags(diagonals, offsets, shape=(N, N))

    # Convert to a format that allows element modification (csc_matrix allows assignment)
    block = csc_matrix(block)
    # If you want to convert back to a dia_matrix, you can do so:
    block = block.todia()

    # Create a larger block matrix made of block_count x block_count blocks
    large_matrix = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)])

    return large_matrix

def create_sparse_matrix_dudy(N):
    dx, dy = 1, 1
    block_count=N
    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [np.ones(N)/(4*dy), np.ones(N)/(4*dy)]  # +1, +1
    offsets = [0, 1]  # +1 on the super diagonal, +1 on the main diagonal
    
    # Create the NxN block matrix
    block_negative = diags(diagonals, offsets, shape=(N, N), format='csc')
    block_positive = block_negative * -1

    large_matrix = []
    for i in range(N):
        row_blocks = []
        for j in range(N):
            if i + 1 == j:  # Place the -1 block
                row_blocks.append(block_negative)
            elif i == j + 1:  # Place the +1 block
                row_blocks.append(block_positive)
            else:  # Fill the rest with zero matrices
                row_blocks.append(csc_matrix((N, N)))
        large_matrix.append(row_blocks)

    # Combine blocks into a single sparse matrix
    final_matrix = bmat(large_matrix, format='csc')

    return final_matrix

def create_sparse_matrix_dvdx(N):
    dx, dy = 1, 1
    diagonals = [np.ones(N-1)/(4*dx), -np.ones(N-1)/(4*dx)]
    offsets = [1, -1]  # Superdiagonal and subdiagonal
    base_block = diags(diagonals, offsets, shape=(N, N), format='csc')

    # Create the full matrix
    blocks = []
    for i in range(N):
        row_blocks = []
        for j in range(N):
            if i == j:  # Main diagonal
                row_blocks.append(base_block)
            elif i == j + 1:  # Below the main diagonal
                row_blocks.append(base_block)
            else:  # All other positions
                row_blocks.append(csc_matrix((N, N)))
        blocks.append(row_blocks)

    # Combine into a full sparse matrix
    full_matrix = bmat(blocks, format='csc')

    return full_matrix


def create_sparse_matrix_dy_old_ok(N):
    # Size of the sparse matrix
    size = N * N

    # Create the diagonals for the top-left to bottom-right direction
    diagonals = [-np.ones(size - N), np.ones(size - N)]
    
    # Offset positions for the diagonals
    offsets = [N, -N]  # N for +1, -N for -1

    # Create the sparse diagonal matrix
    sparse_matrix = diags(diagonals, offsets, shape=(size, size), format='csr')

    return sparse_matrix

def create_sparse_matrix_dy_boundConditionsNotClear(N):
    # Size of the sparse matrix
    size = N * N

    # Create the diagonals for the top-left to bottom-right direction
    diagonals = [-np.ones(size), np.ones(size - (N))]
    
    # Offset positions for the diagonals
    offsets = [0, N]  # N for +1, -N for -1

    # Create the sparse diagonal matrix
    sparse_matrix = diags(diagonals, offsets, shape=(size, size), format='csr')

    return sparse_matrix


def create_sparse_matrix_dy_sure(N, dy=1):
    # Size of the sparse matrix
    size = N * N

    # Create the diagonals for the sparse matrix
    diagonals = [-np.ones(size) / dy, np.ones(size - N) / dy]

    # Offset positions for the diagonals
    offsets = [0, N]  # Main diagonal, and offset by N for super diagonal

    # Create the sparse diagonal matrix
    sparse_matrix = diags(diagonals, offsets, shape=(size, size), format='csr').toarray()

    # Adjust last few rows to have static -1 values
    for i in range(size - N, size):
        sparse_matrix[i, :] = 0  # Reset the row
        sparse_matrix[i, i] = -1

    return csc_matrix(sparse_matrix)

def create_sparse_matrix_dy_slow(N, dy=1):
    # Size of the sparse matrix
    size = N * N

    # Create the diagonals directly in sparse form
    main_diagonal = -1 / dy  # The main diagonal values
    super_diagonal = 1 / dy  # The diagonal offset by N

    # Define positions of diagonals and values
    diagonals = [
        [main_diagonal] * size,  # Main diagonal
        [super_diagonal] * (size - N)  # Super diagonal (offset by N)
    ]
    offsets = [0, N]  # Main diagonal, and offset by N for super diagonal

    # Create the sparse diagonal matrix
    sparse_matrix = diags(diagonals, offsets, shape=(size, size), format='csc')

    # Adjust last few rows directly in sparse format
    for i in range(size - N, size):
        sparse_matrix[i, :] = 0  # Reset the row
        sparse_matrix[i, i] = -1  # Set the diagonal to -1

    return sparse_matrix

def create_sparse_matrix_dy(N, dy=1):
    size = N * N
    main_diagonal = -1 / dy
    super_diagonal = 1 / dy

    # Create the primary matrix
    sparse_matrix = diags(
        diagonals=[[-1 / dy] * size, [1 / dy] * (size - N)],
        offsets=[0, N],
        shape=(size, size),
        format='csc',
    )

    # Modify the last N rows efficiently
    last_rows = eye(N, format='csc', dtype=float) * -1
    sparse_matrix[-N:, -N:] = last_rows

    return sparse_matrix

def create_sparse_double_matrix_dydx(N, dx=1, dy=1):
    # Size of the sparse matrix
    size = N * N
    
    # Create the sparse matrix for dy (top-left to bottom-right direction)
    diagonals_dy = [-(1/dx + 1/(2*dy))*np.ones(size), (1/(2*dy))*np.ones(size - N)]
    offsets_dy = [0, N]  # N for +1, -N for -1
    sparse_matrix_dy = diags(diagonals_dy, offsets_dy, shape=(size, size), format='csr')
    
    block_count = N
    diagonals_dx = [(1/dx)*np.ones(N)]  # +1, 0, -1
    offsets_dx = [1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

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
    diagonals_dy = [-(1/dy + 1/(2*dx))*np.ones(size), (1/(dy))*np.ones(size - N)]
    offsets_dy = [0, N]  # N for +1, -N for -1
    sparse_matrix_dy = diags(diagonals_dy, offsets_dy, shape=(size, size), format='csr')
    
    block_count = N
    diagonals_dx = [(1/(2*dx))*np.ones(N)]  # +1, 0, -1
    offsets_dx = [1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

    # Create a single NxN block
    block = diags(diagonals_dx, offsets_dx, shape=(N, N))

    # Create a larger block matrix made of block_count x block_count blocks
    sparse_matrix_dx = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)], format='csr')

    # Combine the two sparse matrices by adding them together
    combined_sparse_matrix = sparse_matrix_dy + sparse_matrix_dx

    return combined_sparse_matrix

def create_sparse_double_matrix_dxdy_oldok(N, dx=1, dy=1):
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