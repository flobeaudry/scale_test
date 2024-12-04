# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, bmat, csc_matrix, csr_matrix, block_diag, vstack, hstack
from scipy.sparse.linalg import spsolve, gmres
import os
from scipy import optimize
from scipy.optimize import KrylovJacobian
from datetime import datetime, timedelta
import matplotlib.cm as cm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv
from scipy.linalg import pinv
from scipy.ndimage import gaussian_filter
import skimage 
from skimage import filters
from numpy.linalg import cond
import scipy.sparse as sp
from scipy.linalg import lu
from scipy.ndimage import sobel
from scipy.optimize import minimize

# The problem has this form:
# (u_i - u_(i-1))/Δx + (v_j - v_(j-1))/Δy= f

# Which ca be rewritten like:
#   A  *  U   +   B  *  V   =   F
# (9x9)*(9x1) + (9x9)*(9x1) = (9x1)

# And:
#  [ A  0 ] [ U ]  =   [Fx Fy]
#  [ 0  B ] [ V ]
#  (18x18) (18x1)  = (18x1)

# And we will solve like:
#  UV = AB^(-1) * F

# Initializing the variables
# -------------- Determine your F field --------------------------------
# Output 11 !!!
# out = '11'
#F_grid = np.random.randn(500,500)
# Output 20 !!!
#out = '20'
#pattern = [-1] + [0] * 254 + [1 ,1] + [0] * 254 + [-1]
# Output 21 !!!
#out = '21'
#pattern = [-1, 1] + [1 , -1] *255 + [1, -1]
# Output 22 !!!
#out = '22'
#pattern = [-1, 1] + [+1, 0 , -1, 0] *127 + [1, -1]
# Output 23 !!!
#out = '23'
#pattern = [-1, 1] + [1, 0, 0, -1, 0, 0] *85 + [1, -1]
# Output 24 !!!
#out = '24'
#pattern = [-1, 1] + [1,  0, 2, 0, -1, 0, -2, 0] *64 + [1, -1]
# Output 25 !!!
#out = '25'
#pattern = [-1, 1] + [1 ,0,0 ,0, 0, 0, 0, 0, 0,0, -1 ,0,0 ,0, 0, 0, 0, 0, 0,0]*26 + [1, -1]
# Output 26 !!
#out = '26'
#pattern = [-1, 1] + [1 ,0,0 ,0, 0, 0, 0, 0, 0,0, 0,0,0 ,0, 0, 0, 0, 0, 0,0, -1 ,0,0 ,0, 0, 0, 0, 0, 0,0, 0 ,0,0 ,0, 0, 0, 0, 0, 0,0]*13 + [1, -1]

# Small test output
out = "test"
#pattern = [0, 0 ,0 ,0, -1, 1, 0,0,0,0]
pattern = [3,-3,3,-3,3,-3]
F_grid = np.tile(pattern, (len(pattern), 1))
# -------------------------------------------------------------------------------

# Construct the full F matrix
N = len(F_grid[0])
N2 = N**2
z_grid = np.zeros((N,N))
# Stack both F_matrixes for U and V deformations; F_grid, z_grid for U defo and z_grid, F_grid for V defo
#F = np.vstack([F_grid, z_grid, z_grid, F_grid]).flatten()
F = np.vstack([F_grid, z_grid]).flatten()

# Define the resolution
dx = 1
dy = 1

# A and B being the finite differences matrices (NxN)
# A = dU/dx and B = dV/dy (div terms)
A_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dx
B_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dy 
# C = dU/dy and D = dV/dx (shear terms)
C_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dy
D_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dx

ABCD_sparse = bmat([[A_sparse, -B_sparse], 
                     [C_sparse, D_sparse]])

#ABCD_sparse = block_diag([
#    block_diag([A_sparse, -B_sparse]),
#    block_diag([C_sparse, D_sparse])
#])

# Compute UV
UV = spsolve(ABCD_sparse, F)

print('Welcome in the artificial fields generation program! Your velocity fields are of shape:',np.shape(UV))
# Get the U and V fields
# Ad
#U_grid = ((UV[:N2].reshape((N,N)))**2 + (UV[2*N2:3*N2].reshape((N,N)))**2)**(1/2)
U_grid = (UV[:N2].reshape((N,N)))
U_grid = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
#V_grid = ((UV[N2:N2*2].reshape((N,N)))**2 + (UV[3*N2:].reshape((N,N)))**2)**(1/2)
V_grid = (UV[N2:].reshape((N,N)))
V_grid = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)

# Show the U, V field in quivers
plt.figure()
plt.quiver( U_grid, V_grid, cmap=cm.viridis)
plt.title("Speeds field")
plt.show()

# Show the divergence field you prescibed
plt.figure()
plt.title("Divergence field")
plt.pcolormesh(F_grid)
plt.colorbar()
plt.show()

# Get F back to validate
f_valid = np.zeros((N2*4))
for i in range(len(UV)):
    # For u 
    if i < N2+1:
        if i == 0: f_valid[i] = -1 # Boundary condition of u
        else:
            f_valid[i] = (UV[i] - UV[i-1]) / dx      
    # For v
    else:
        if i == N2: f_valid[i] = -1 # Boundary condition of v
        else:
            f_valid[i] = (UV[i] - UV[i-1]) / dy  
            
#f_valid_grid = f_valid.reshape((N,2*N))[:,N:]
#f_valid_grid = f_valid.reshape((2*N,N))

f_valid_grid = f_valid.reshape((4*N,N))
f_new_grid = F.reshape((4*N, N))

# Show the recomputed divergence field to validate its the same as the prescribed one
plt.figure()
plt.title("Recomputed divergence field")
#plt.pcolormesh(f_valid_grid[:N])
plt.pcolormesh(f_valid_grid)
plt.colorbar()
plt.show()
# Show the recomputed divergence field to validate its the same as the prescribed one
plt.figure()
plt.title("Recomputed divergence field")
#plt.pcolormesh(f_valid_grid[:N])
plt.pcolormesh(f_new_grid)
plt.colorbar()
plt.show()

# -------------- Saving the files ----------------------------------
start_date = datetime(2002, 1, 1)
end_date = datetime(2002, 1, 31, 18) # Full end date

time_delta = timedelta(hours=6)
time_steps = int((end_date - start_date).total_seconds() // 3600 // 6) + 1

# Where to put the files
output_dir = f"/aos/home/fbeaudry/git/scale_test/output{out}"
os.makedirs(output_dir, exist_ok=True)

# Create and save the fields over time
current_time = start_date
for t in range(time_steps):
    # Add a small random perturbation so the background is not zero everywhere
    u = U_grid+0.05*np.random.rand(N,N+1)
    v = V_grid+0.05*np.random.rand(N+1,N)
    
    # Filenames gossage
    file_suffix = f"{current_time.strftime('%Y_%m_%d_%H_%M')}.{out}"
    u_filename = os.path.join(output_dir, f"u{file_suffix}")
    v_filename = os.path.join(output_dir, f"v{file_suffix}")

    # Save the u, v files
    #np.savetxt(u_filename, u, fmt='%.6f')
    #np.savetxt(v_filename, v, fmt='%.6f')

    current_time += time_delta
    
plt.figure()
plt.title("u")
plt.pcolormesh(u)
plt.colorbar()
plt.show()

plt.figure()
plt.title("U_grid")
plt.pcolormesh(U_grid)
plt.colorbar()
plt.show()

#%%


# Define the shear equation to minimize
def shear_and_divergence(UV, N, F_shear):
    # Reshape UV back to U and V grids
    U = UV[:N*N].reshape((N, N))
    V = UV[N*N:].reshape((N, N))

    # Compute finite differences for derivatives
    du_dx = np.hstack([np.zeros((N, 1)), np.diff(U, axis=1)])
    dv_dy = np.vstack([np.zeros((1 ,N)), np.diff(V, axis=0)])
    du_dy = np.vstack([np.zeros((1,N)), np.diff(U, axis=0)])
    dv_dx = np.hstack([np.zeros((N, 1)), np.diff(V, axis=1)])
    print(du_dy)
    print(du_dx)

    # Shear terms
    shear_term_1 = (du_dx - dv_dy)**2
    shear_term_2 = (du_dy + dv_dx)**2
    #shear_term_2 = (du_dy + dv_dx)**2
    # Total shear field from the velocity field
    shear_field = np.sqrt(shear_term_1 + shear_term_2)
    #shear_field = shear_term_2
    print(shear_field)

    # Compute the error between the shear field and the desired F_shear field
    error = np.sum((shear_field[1:,:-1] - F_shear[1:, :-1])**2)  # Exclude boundary effects
    return error

# Initialize a grid and an example shear field
N = 5  # Grid size
#F_shear = np.array([[3, 2, 2, 2, 2], [-3, -2, -2, -2, -2], [3, 2, 2, 2, 2], [-3, -2, -2, -2, -2], [-4, 1, 1, 1, 1]])
#F_shear = np.array([[3, 2, 2, 2, 2], [-3, -2, -2, -2, -2], [3, 2, 2, 2, 2], [-3, -2, -2, -2, -2], [-4, 1, 1, 1, 1]])
pattern = [3,3,3,3,3]
F_shear = np.tile(pattern, (len(pattern), 1))

# Initial guess for U and V (flattened)
V_initial = np.zeros(N * N)
U_initial = np.ones(N * N)
#UV_initial = np.vstack((U_initial, V_initial)).flatten()
#print(np.shape(UV_initial))
UV_initial = np.array([[1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]).flatten()

print(np.shape(UV_initial))

# Minimize the shear error 
#result = minimize(shear_and_divergence, UV_initial, args=(N, F_shear), method='L-BFGS-B')
#result = optimize.newton_krylov(fun, [0, 0])


# Extract the optimized U and V fields
UV_opt = result.x
U_opt = UV_opt[:N*N].reshape((N, N))
V_opt = UV_opt[N*N:].reshape((N, N))

# Plot the resulting velocity field (U, V)
plt.figure()
plt.quiver(np.round(U_opt,1), np.round(V_opt,1))
plt.title("Optimized Velocity Field (U, V)")
plt.show()

plt.figure()
plt.title("Recomputed divergence field")
plt.pcolormesh(F_shear)
plt.colorbar()
plt.show()

#%%
# Define the shear equation to minimize
def shear_and_divergence(UV, N, F_shear, lambda_reg=0.01):
    # Reshape UV back to U and V grids
    U = UV[:N*N].reshape((N, N))
    V = UV[N*N:].reshape((N, N))

    # Compute finite differences for derivatives
    du_dx = np.hstack([np.zeros((N, 1)), np.diff(U, axis=1)])
    dv_dy = np.vstack([np.zeros((1 ,N)), np.diff(V, axis=0)])
    du_dy = np.vstack([np.zeros((1,N)), np.diff(U, axis=0)])
    dv_dx = np.hstack([np.zeros((N, 1)), np.diff(V, axis=1)])

    # Shear terms
    shear_term_1 = (du_dx - dv_dy)**2
    shear_term_2 = (du_dy + dv_dx)**2
    shear_field = np.sqrt(shear_term_1 + shear_term_2)

    # Compute the error between the shear field and the desired F_shear field
    error = np.sum((shear_field[1:,:-1] - F_shear[1:, :-1])**2)  # Exclude boundary effects
    
    # Add a regularization term to prevent large velocities
    regularization = np.sum(U**2 + V**2)
    total_error = error + lambda_reg * regularization
    print(shear_field)
    print(error)
    return total_error

def fun(UV):
    U = UV[:N*N].reshape((N, N))
    V = UV[N*N:].reshape((N, N))

    # Compute finite differences for derivatives
    du_dx = np.hstack([np.zeros((N, 1)), np.diff(U, axis=1)])
    dv_dy = np.vstack([np.zeros((1 ,N)), np.diff(V, axis=0)])
    du_dy = np.vstack([np.zeros((1,N)), np.diff(U, axis=0)])
    dv_dx = np.hstack([np.zeros((N, 1)), np.diff(V, axis=1)])

    # Shear terms
    shear_term_1 = (du_dx - dv_dy)**2
    shear_term_2 = (du_dy + dv_dx)**2
    shear_field = np.sqrt(shear_term_1 + shear_term_2)
    return(shear_field)

# Smoothing function
def smooth_initial_guess(UV, sigma=1):
    U = UV[:N*N].reshape((N, N))
    V = UV[N*N:].reshape((N, N))
    U_smooth = gaussian_filter(U, sigma=sigma)
    V_smooth = gaussian_filter(V, sigma=sigma)
    return np.hstack([U_smooth.flatten(), V_smooth.flatten()])

# Initialize a grid and an example shear field
N = 6  # Grid size
#pattern = [3, 3, 3, 3, 3]
pattern = [3, -3, 3, -3, 3,-3]
#pattern = [0, 0, 0, 0, 0]
F_shear = np.tile(pattern, (len(pattern), 1))

# Initial guess for U and V (flattened), small random values for initial velocities
#np.random.seed(0)
#UV_initial = np.random.uniform(-0.1, 0.1, 2 * N * N)

# Apply a smoothing filter to avoid sharp gradients in the initial guess
#UV_initial = smooth_initial_guess(UV_initial, sigma=1)

#V_initial = np.zeros(N * N)
#U_initial = np.ones(N * N)
#UV_initial = np.vstack((U_initial, V_initial)).flatten()

UV_initial = np.vstack((U_grid[:,:-1],V_grid[:-1,:])).flatten()

#UV_initial = np.array([[1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]).flatten()


# Define bounds for the optimizer to keep U and V in a reasonable range
bounds = [(-3, 3)] * (2 * N * N)  # Adjust bounds based on physical expectations

# Minimize the shear error with regularization and bounds
#result = minimize(shear_and_divergence, UV_initial, args=(N, F_shear), method='L-BFGS-B', bounds=bounds)
#result = minimize(shear_and_divergence, UV_initial, args=(N, F_shear), 
#                  method="Nelder-Mead", bounds=bounds, tol=toler)
                
error_threshold = 10
current_error = float('inf')

# Iteration loop
#while current_error >= error_threshold:
#    result = minimize(shear_and_divergence, UV_initial, args=(N, F_shear), 
 #                 method="Nelder-Mead", bounds=bounds)

#    # Update parameters and current error
#    UV_initial = result.x
#    current_error = result.fun

#result = optimize.newton_krylov(shear_and_divergence, UV_initial, inner_maxiter = 1000)   
result = optimize.newton_krylov(fun, UV_initial, inner_maxiter = 1000)   

# Extract the optimized U and V fields
UV_opt = result.x
U_opt = UV_opt[:N*N].reshape((N, N))
V_opt = UV_opt[N*N:].reshape((N, N))

# Plot the resulting velocity field (U, V)
plt.figure()
plt.quiver(np.round(U_opt,1), np.round(V_opt,1))
plt.title("Optimized Velocity Field (U, V)")
plt.show()

plt.figure()
plt.title("Recomputed divergence field")
plt.pcolormesh(F_shear)
plt.colorbar()
plt.show()