# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, bmat, csc_matrix, csr_matrix, block_diag
from scipy.sparse.linalg import spsolve, gmres
import os
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
#N = 5   # Put a value of the shape of your F matrix
#N2 = 25 # Square of your N value
# F being the divergence matrix (18x1)
# Function to get F for linear deformation along vertical lines
#  i.e. if N = 5, defo_info = [-1, 2, -2, 2, -1]
#         [ -1  2 -2  2 -1]
#         [ -1  2 -2  2 -1]
#         [ -1  2 -2  2 -1]
#         [ -1  2 -2  2 -1]
#         [ -1  2 -2  2 -1]
def get_f (def_info):
    N = len(def_info)
    F = np.reshape(np.array(list(def_info)*N), (N,N))
    N2 = N**2
    return(F, N, N2)



# -------------- Determine your F field --------------------------------
# Field with one lead in the middle (u are like: 0, -1, +1, 0)
#F_grid, N, N2= get_f([1, -2, 1])

# Field with alterning conv and div (u are like: 0, -1, +1, -1, +1, 0)
#F_grid, N, N2= get_f([-1, 2, -2, 2, -1])

# Field with one big lead in the middle (u are like: 0, -1, -1, +1, +1, 0)
#F_grid, N, N2= get_f([-1, 0, 2, 0, -1])

#F_grid, N, N2= get_f([-1,  2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2,2,-2,2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2,2,-2,2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2,2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 2, -1])

#F_grid = np.diag(np.full(10,-2))+np.diag(np.ones(9),1)+np.diag(np.ones(9),-1)+np.diag(np.ones(7),-3)


# Output 15 !!!
#F_grid = (np.diag(np.full(500,-2))+np.diag(np.ones(499),1)+np.diag(np.ones(499),-1)+np.diag(np.ones(450),-50))
#F_grid = (np.diag(np.full(50,-2))+np.diag(np.ones(49),1)+np.diag(np.ones(49),-1)+np.diag(np.ones(47),-3))
#F_grid = (np.diag(np.full(5,-2))+np.diag(np.ones(4),1)+np.diag(np.ones(4),-1)+np.diag(np.ones(2),-3))

##F_grid = (np.diag(np.full(500,-2))+np.diag(np.ones(499),1)+np.diag(np.ones(499),-1))*0.5+0.1*np.random.randn(500,500)

# Output 16 !!!
#pattern = [-1] + [2, -2] * 249 + [-1]
#F_grid = np.tile(pattern, (500, 1))

# Output 17 !!!
#pattern = [-1] + [2, 2, -2, -2] * 124  +[2] + [2] + [-1]
#F_grid = np.tile(pattern, (500, 1))

# Output 11 !!!
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

out = '26'
pattern = [-1, 1] + [1 ,0,0 ,0, 0, 0, 0, 0, 0,0, 0,0,0 ,0, 0, 0, 0, 0, 0,0, -1 ,0,0 ,0, 0, 0, 0, 0, 0,0, 0 ,0,0 ,0, 0, 0, 0, 0, 0,0]*13 + [1, -1]

out = "test"
pattern = [-1,0 ,0 ,0, 1, 1, 0,0,0,-1]

F_grid = np.tile(pattern, (len(pattern), 1))
# -------------------------------------------------------------------------------


# Construct the full F matrix
N = len(F_grid[0])
N2 = N**2
z_grid = np.zeros((N,N))
F = np.vstack([F_grid, z_grid]).flatten()

# Define the resolution
dx = 1
dy = 1


# A and B being the finite differences matrices (9x9)
#A = (np.diag(np.full(N2, 1)) + np.diag(-np.ones(N2-1), -1)) / dx
#B = (np.diag(np.full(N2, 1)) + np.diag(-np.ones(N2-1), -1)) / dy
# Get them together in a square finite differences matrix AB (18x18)
# Trying with sparse matrixes to make lighter computationaly to inverse
A_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dx
B_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dy 
AB_sparse = block_diag([A_sparse, B_sparse])

# Compute UV
UV = spsolve(AB_sparse, F)

print('Welcome in the artificial fields generation program! Your velocity fields are of shape:',np.shape(UV))
U_grid = UV[:N2].reshape((N,N))
U_grid = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
V_grid = UV[N2:].reshape((N,N))
V_grid = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)

# Show the U, V fiel in quivers
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
f_valid = np.zeros((N2*2))
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
f_valid_grid = f_valid.reshape((2*N,N))

# Show the recomputed divergence field to validate its the same as the prescribed one
plt.figure()
plt.title("Recomputed divergence field")
plt.pcolormesh(f_valid_grid[:N])
plt.colorbar()
plt.show()


# Saving the files ----------------------------------
start_date = datetime(2002, 1, 1)
#end_date = datetime(2002, 1, 10, 18) # Testing end date
end_date = datetime(2002, 1, 31, 18) # Full end date

time_delta = timedelta(hours=6)
time_steps = int((end_date - start_date).total_seconds() // 3600 // 6) + 1

# Where to put the files
#output_dir = "/aos/home/fbeaudry/git/scale_test/output25"
output_dir = f"/aos/home/fbeaudry/git/scale_test/output{out}"
os.makedirs(output_dir, exist_ok=True)

# Create and save the fields over time
current_time = start_date
for t in range(time_steps):
    u = U_grid+0.05*np.random.rand(N,N+1)
    v = V_grid+0.05*np.random.rand(N+1,N)
    #u = np.random.randn(500,501)
    #v = np.random.randn(501,500)
    
    # Filenames gossage
    #file_suffix = current_time.strftime("%Y_%m_%d_%H_%M") + ".25"
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
# Parameters
nx, ny = 100, 100  # Grid size
dx, dy = 1, 1  # Grid spacing
S = np.random.rand(nx, ny)  # Example: known S field

# Initialize u and v fields
u = np.zeros((nx, ny+1))
v = np.zeros((nx+1, ny))

# Function to compute S from u and v
def compute_S(u, v, dx, dy):
    dudx = (u[1:, :] - u[:-1, :]) / dx
    dvdy = (v[:, 1:] - v[:, :-1]) / dy
    dudy = (u[:, 1:] - u[:, :-1]) / dy
    dvdx = (v[1:, :] - v[:-1, :]) / dx
    print(np.shape(v[1:, :]), np.shape(v[:-1, :] ))
    strain_rate = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)
    return strain_rate

# Iterative solver
for iteration in range(1000):  # Adjust iteration number as needed
    S_computed = compute_S(u, v, dx, dy)
    residual = S[1:, 1:] - S_computed
    
    # Update u and v fields based on residual (simple gradient descent)
    u[1:, :] += 0.01 * residual  # Adjust step size (0.01) as needed
    v[:, 1:] += 0.01 * residual

    # Check convergence
    if np.max(np.abs(residual)) < 1e-6:
        print(f"Converged after {iteration} iterations")
        break