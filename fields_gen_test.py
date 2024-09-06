import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# The problem has this form:
# (u_i - u_(i-1))/Δx + (v_j - v_(j-1))/Δy= f

# Which ca be rewritten like:
#   A  *  U   +   B  *  V   =   F
# (9x9)*(9x1) + (9x9)*(9x1) = (9x1)

# And:
#  [ A  0 ] [ U ]  =   F
#  [ 0  B ] [ V ]
#  (18x18) (18x1)  = (18x1)

# And we will solve like:
#  UV = AB^(-1) * F

# Initializing the variables
N = 3   # Put a value of the shape of your F matrix
N2 = 9 # Square of your N value

dx = 1
dy = 1

# F being the divergence matrix (18x1)
F_grid = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
#F_grid = np.array([[-1, 2, -2, 2, -1], [-1, 2, -2, 2, -1], [-1, 2, -2, 2, -1], [-1, 2, -2, 2, -1], [-1, 2, -2, 2, -1]])
F = np.vstack([F_grid,F_grid]).flatten()
print(np.shape(F))

# A and B being the finite differences matrices (9x9)
A = (np.diag(np.full(N2, 1)) + np.diag(-np.ones(N2-1), -1)) / dx
B = (np.diag(np.full(N2, 1)) + np.diag(-np.ones(N2-1), -1)) / dy
# Get them together in a square finite differences matrix AB (18x18)
z = np.zeros((N2, N2))
AB = np.vstack([np.hstack([A , z]), np.hstack([z, B])])
print(np.shape(AB))

# Inverse AB
AB_inv = np.linalg.inv(AB)

# Compute UV
UV = np.dot(AB_inv, F)
print(np.shape(UV))
U_grid = UV[:N2].reshape((N,N))
V_grid = UV[N2:].reshape((N,N))
print(U_grid)
print(V_grid)

# Get F back to validate
f_valid = np.zeros((N2*2))
for i in range(len(UV)):
    # For u 
    if i < N2+1:
        if i == 0: f_valid[i] = 1 # Boundary condition of u
        else:
            f_valid[i] = (UV[i] - UV[i-1]) / dx      
    # For v
    else:
        if i == N2: f_valid[i] = 1 # Boundary condition of v
        else:
            f_valid[i] = (UV[i] - UV[i-1]) / dy  
            
f_valid_grid = f_valid.reshape((N,2*N))[:,N:]


print("The initial F field: ")
print(F_grid)
print("The computed F field: ")
print(f_valid_grid)
