import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from numpy.linalg import matrix_rank
from scipy.sparse import diags, bmat, csc_matrix, eye, vstack, lil_matrix


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

def create_sparse_matrix_dx_2025_01_27(N, dx=1):
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

def create_sparse_matrix_dy_shear_singular(N, dy=1):
    size = (N) * (N)

    diag_N = np.ones(size-N) * (1/dy)
    diag_Nplus1 = np.ones(size - (N+1)) * (1/dy)
    diag_minusN = np.ones(size- (N+1)) * (-1/dy)
    diag_minusNm = np.ones(size - (N)) * (-1/dy)

    base_matrix = diags(
        #diagonals=[super_plus_diagonal, super_plus_diagonal, super_neg_diagonal, super_neg_diagonal],
        #offsets=[N, (N+1), -N, -(N-1)],  # Main diagonal and super-diagonal (shifted by -N rows)
        diagonals = [diag_N, diag_Nplus1, diag_minusN, diag_minusNm],
        offsets = [N, N+1, -(N+1), -(N)],
        shape=(size, size),
        format='lil'
    )
    
    # (periodic boundaries)
    for i in range(N):
        if i + 1 < N:
            base_matrix[i, size - N + i]     = -1 / dy
            base_matrix[i, size - N + i + 1] = -1 / dy
    # Add bottom-left corner diagonal: last N rows, first N columns
    for i in range(N):
        row = size - N + i
        #if i + 1 < N:
        base_matrix[row, i]     = 1 / dy
        base_matrix[row, i + 1] = 1 / dy
    
    # Step 2: Replace every Nth row (starting from 0) with a row having 1 at the diagonal
    for i in range(N, size, N):
    #for i in range(0, size, N):
        print("here")
        base_matrix[i, :] = 0  # set only one non-zero at the diagonal
        #base_matrix[i, i] = 0.0
        base_matrix[i, i] = 1.0

    # Step 3: Convert to CSC or CSR for final format
    sparse_matrix = base_matrix.tocsc()
    
    #A = sparse_matrix.toarray()
    #print("Rank:", matrix_rank(A))
    #print("Is square:", A.shape[0] == A.shape[1])
    
    # Plot sparsity pattern
    plt.figure(figsize=(8, 8))  # adjust size as needed
    plt.spy(sparse_matrix, markersize=0.5)
    plt.title("Sparsity Pattern")
    # Save as image (e.g., PNG)
    plt.savefig("SIMS/project/sparse_matrix_pattern.png", dpi=300, bbox_inches='tight')
    plt.close()
    return sparse_matrix
    
def create_sparse_matrix_dy_shear_2025_06_17(N, dy=1):
    blocks = []
    Nx = N
    Ny = N

    def wrapped(j):
        return j % Ny

    for i in range(Nx):
        row_blocks = []
        for j in range(Nx):
            block = lil_matrix((Ny, Ny))

            if j == i:  # Diagonal block
                for k in range(Ny):
                    block[k, k] += 1 / (2 * dy)
                    block[k, wrapped(k - 2)] += -1 / (2 * dy)
                row_blocks.append(block.tocsc())

            elif j == i - 1:  # Lower diagonal block
                for k in range(Ny):
                    block[k, k] += 1 / (2 * dy)
                    block[k, wrapped(k - 2)] += -1 / (2 * dy)
                row_blocks.append(block.tocsc())

            else:
                row_blocks.append(None)

        blocks.append(row_blocks)

    return bmat(blocks, format="csc")


def create_sparse_matrix_dy_shear_also_singular(N, dy=1):
    blocks = []
    Nx = N
    Ny = N

    def wrapped(j):
        return j % Ny

    for i in range(Nx):
        row_blocks = []
        for j in range(Nx):
            block = lil_matrix((Ny, Ny))

            if j == i:  # Diagonal block
                for k in range(Ny):
                    block[k, k] += 1 / (2 * dy)
                    block[k, wrapped(k - 2)] += -1 / (2 * dy)
                row_blocks.append(block.tocsc())

            elif j == i - 1:  # Lower diagonal block
                for k in range(Ny):
                    block[k, k] += 1 / (2 * dy)
                    block[k, wrapped(k - 2)] += -1 / (2 * dy)
                row_blocks.append(block.tocsc())

            else:
                row_blocks.append(None)

        blocks.append(row_blocks)

    return bmat(blocks, format="csc")

def create_sparse_matrix_dy_shear_real(N, dy=1):
    size = N * N
    M = lil_matrix((size, size))

    for i in range(size):
        if i % N == 0:
            # Set identity row: only a 1 on the diagonal
            M[i, i] = 1
        else:
            col_pos = i + N
            col_diag = i + 2 * N

            if col_pos < size:
                M[i, col_pos] += 1 / dy
            if col_diag < size:
                M[i, col_diag] += 1 / dy

            if i + N < size:
                M[i + N, i] += -1 / dy
            if i + 2 * N < size:
                M[i + 2 * N, i] += -1 / dy

    # Top-right corner block
    for i in range(N):
        if i % N == 0:
            M[i, i] = 1
        else:
            M[i, size - N + i] += -1 / dy
            if i + 1 < N and (i + 1) % N != 0:
                M[i + 1, size - N + i] += -1 / dy

    # Bottom-left corner block
    for i in range(N):
        base = size - N
        row = base + i
        if row % N == 0:
            M[row, row] = 1
        else:
            M[row, i] += 1 / dy
            if i + 1 < N and (row + 1) % N != 0:
                M[row + 1, i] += 1 / dy

    return M.tocsc()

def create_sparse_matrix_dy_shear_anothersingularhihi(N, dy=1):
    size = N * N
    M = lil_matrix((size, size))

    for i in range(size):
        if i % N == 0:
            # Identity row for Dirichlet rows
            M[i, i] = 1
        else:
            if i + N < size:
                M[i, i + N] += 1 / dy
            if i - N >= 0:
                M[i, i - N] += -1 / dy

    # Top-right corner: apply -1/dy diagonals
    for i in range(N):
        row = i
        col = size - N + i
        M[row, col] += -1 / dy
        if row + 1 < N:
            M[row + 1, col] += -1 / dy

    # Bottom-left corner: apply +1/dy diagonals
    for i in range(N):
        row = (N * (N - 1)) + i
        col = i
        M[row, col] += 1 / dy
        if row + 1 < size:
            M[row + 1, col] += 1 / dy

    return M.tocsc()



def create_sparse_matrix_dy_shear(N, dy=1):
    print('hi there')
    size = N * N
    M = lil_matrix((size, size))

    for i in range(size):
        if i % N == 0:
            M[i, i] = 1  # Dirichlet condition
            #M[i, i+N] = 1  # Dirichlet condition
        else:
            M[i, i] = -1 / dy
            if i + 1 < size:
                M[i, i + 1] = 1 / dy

    return M.tocsc()

def create_sparse_matrix_dx_shear(N, dx=1):
    Nx = N
    Ny = N
    
    def idx(i, j):
        return i % Nx * Ny + j  # periodic in i, Dirichlet in j

    D = lil_matrix((Nx * Ny, Nx * Ny))

    for i in range(Nx):
        for j in range(Ny):
            row = i * Ny + j

            # f[i, j]
            D[row, idx(i, j)] += 1 / (2 * dx)

            # f[i, j-1] — Dirichlet BC in j
            if j - 1 >= 0:
                D[row, idx(i, j - 1)] += 1 / (2 * dx)

            # f[i-1, j] — periodic in i
            D[row, idx(i - 1, j)] += -1 / (2 * dx)

            # f[i-2, j-1] — both i-2 (wrap) and j-1 (Dirichlet check)
            if j - 1 >= 0:
                D[row, idx(i - 2, j - 1)] += -1 / (2 * dx)

    return D.tocsc()

def create_sparse_matrix_dx(N, dx=1):
    block_count = N

    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [-np.ones(N) / dx, np.ones(N) / dx]  # -1/dx on main, +1/dx on super diagonal
    offsets = [-1, 0]  # Main diagonal, super diagonal
    
    block = diags(diagonals, offsets, shape=(N, N), format="csc")

    # Replace the last row of each block with [-1, 0,...0] to introduce the static -1 behavior
    block = block.toarray()
    block[0, :] = 0  # Reset the last row
    block[0, 0] = 1
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

def create_sparse_matrix_dy_20250127(N, dy=1):
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

def create_sparse_matrix_dy(N, dy=1):
    size = N * N
    
    main_diagonal = np.ones(size) * 1 / dy
    super_diagonal = np.ones(size - N) * -1 / dy

    sparse_matrix = diags(
        diagonals=[main_diagonal, super_diagonal],
        offsets=[0, -N],  # Main diagonal and super-diagonal (shifted by -N rows)
        shape=(size, size),
        format='csc'
    )

        # Create the identity block for the bottom-right N x N block
    identity_block = eye(N, format='csc', dtype=float)

    # Replace the bottom-right N x N block in the sparse matrix
    sparse_matrix = sparse_matrix.tolil()  # Switch to LIL format for efficient row slicing
    sparse_matrix[:N, :N] = identity_block  # Replace the last N rows/columns with the identity block
    sparse_matrix = sparse_matrix.tocsc()  # Convert back to CSC format

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