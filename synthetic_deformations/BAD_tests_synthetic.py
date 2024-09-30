### div_shear_synthetic.py ###
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


#%% 
### gen_du.py ###

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime, timedelta


### Background noise and general diverging field in x-dir
# output 12

# Initialize the grid
nx, ny = 500, 500  # Number of grid points
dx, dy = 1.0, 1.0  # Grid spatial resolution
base_divergence_rate = 1e-6  # Base divergence rate
noise_amplitude = 0.1  # Amplitude of the random noise

# Function to create a time-evolving diverging velocity field in the x-direction with background noise
def create_fields(t):
    # Time-evolving divergence rate
    divergence_rate = base_divergence_rate * t

    # Linear divergence in the x-direction that evolves with time
    u = np.linspace(0, divergence_rate * nx, nx+1).reshape(1, nx+1) * np.ones((ny, 1))
    
    # Adding a small random noise to the u field
    u += noise_amplitude * np.random.randn(ny, nx+1)
    
    # v field remains as zero but with small random noise added
    v = noise_amplitude * np.random.randn(ny+1, nx)
    
    return u, v

# Time settings
start_date = datetime(2002, 1, 1)
end_date = datetime(2002, 1, 31, 18)  # January 31st, 18:00
time_delta = timedelta(hours=6)

# Directory to save the files
output_dir = "output12"
os.makedirs(output_dir, exist_ok=True)

current_time = start_date
u_fields = []
v_fields = []

while current_time <= end_date:
    time_step = int((current_time - start_date).total_seconds() / 3600)  # Convert to hours
    u, v = create_fields(time_step)

    # Store fields for plotting
    u_fields.append(u)
    v_fields.append(v)

    # Formatting filename
    file_suffix = current_time.strftime("%Y_%m_%d_%H_%M") + ".12"
    u_filename = os.path.join(output_dir, f"u{file_suffix}")
    v_filename = os.path.join(output_dir, f"v{file_suffix}")

    # Save the fields with .11 extension
    np.savetxt(u_filename, u, fmt='%.6f')
    np.savetxt(v_filename, v, fmt='%.6f')

    # Increment time by 6 hours
    current_time += time_delta

# Convert lists to arrays for easy slicing
u_fields = np.array(u_fields)
v_fields = np.array(v_fields)

# Function to plot the fields at different time steps
def plot_fields(u_fields, v_fields, time_steps):
    fig, axes = plt.subplots(len(time_steps), 2, figsize=(12, len(time_steps) * 5))

    for i, t in enumerate(time_steps):
        u = u_fields[t]
        v = v_fields[t]

        # Plot u field
        ax_u = axes[i, 0]
        c_u = ax_u.imshow(u, cmap='seismic', origin='lower')
        ax_u.set_title(f'u Field at Time Step {t}')
        fig.colorbar(c_u, ax=ax_u)

        # Plot v field
        ax_v = axes[i, 1]
        c_v = ax_v.imshow(v, cmap='seismic', origin='lower')
        ax_v.set_title(f'v Field at Time Step {t}')
        fig.colorbar(c_v, ax=ax_v)

    plt.tight_layout()
    plt.show()

# Plot the fields at selected time steps
time_steps_to_plot = [0, len(u_fields)//4, len(u_fields)//2, len(u_fields)-1]  # Start, quarter, half, and end
plot_fields(u_fields, v_fields, time_steps_to_plot)

"""
### Only background noise and plates diverging ! -----------------------
# output11

# Initialize the grid
nx, ny = 500, 500  # Number of grid points
dx, dy = 1.0, 1.0    # Grid spatial resolution
divergence_rate = 5  # Divergence rate

num_plates = 20
plate_width = 20

# Initialize positions of plates on the left side
initial_x = np.linspace(0, plate_width, num_plates)
plate_y = ny // 2  # Middle of the grid for y position

# Define random dimensions for each plate
np.random.seed(0)  # For reproducibility
plate_widths = np.random.randint(3, 10, size=num_plates)
plate_heights = np.random.randint(3, 10, size=num_plates)

# Function to create the ice field and velocity field with divergence in the x-direction only
def create_fields(t):
    u = np.zeros((ny, nx+1))
    v = np.zeros((ny+1, nx))
    
    for i in range(num_plates):
        x_divergence = divergence_rate * t * i
        plate_width = plate_widths[i]
        plate_height = plate_heights[i]
        
        x_start = int(initial_x[i] + x_divergence)
        x_end = int(initial_x[i] + x_divergence + plate_width)
        y_start = plate_y - plate_height // 2
        y_end = plate_y + plate_height // 2

        # Ensure the indices are within bounds
        x_start = max(x_start, 0)
        x_end = min(x_end, nx)
        y_start = max(y_start, 0)
        y_end = min(y_end, ny)

        # Set initial velocity field based on divergence (u component only)
        u[y_start:y_end, x_start:x_end] = divergence_rate * t

    return u, v

# Time settings
start_date = datetime(2002, 1, 1)
end_date = datetime(2002, 1, 31, 18)  # January 31st, 18:00
time_delta = timedelta(hours=6)

# Directory to save the files
#output_dir = "output10"
output_dir = "output11"
os.makedirs(output_dir, exist_ok=True)

current_time = start_date
while current_time <= end_date:
    time_step = int((current_time - start_date).total_seconds() / 3600)  # Convert to hours
    u, v = create_fields(time_step)

    # Formatting filename
    file_suffix = current_time.strftime("%Y_%m_%d_%H_%M") + ".11"
    u_filename = os.path.join(output_dir, f"u{file_suffix}")
    v_filename = os.path.join(output_dir, f"v{file_suffix}")

    ## Save the fields as .npy files
    #np.save(u_filename, u)
    #np.save(v_filename, v)
    # Save the fields with .10 extension
    #u.astype(np.float32).tofile(u_filename)
    #v.astype(np.float32).tofile(v_filename)
    
    
    np.savetxt(u_filename, np.random.rand(ny, nx+1)/100, fmt='%.6f')
    np.savetxt(v_filename, np.random.rand(ny+1, nx)/100, fmt='%.6f')
    #np.savetxt(u_filename, u+np.random.rand(ny, nx+1), fmt='%.6f')
    #np.savetxt(v_filename, v+np.random.rand(ny+1, nx), fmt='%.6f')

    # Increment time by 6 hours
    current_time += time_delta
    
"""

'''

# Initialize the grid
nx, ny = 100, 100 # Number of grid points
dx, dy = 1.0, 1.0 # Grid spatial resolution
time_steps = 30  # Number of days
dt = 1 # Increment (in days)
divergence_rate = 3

num_plates = 10
plate_width = 50


# Initialize the ice field and store through time
ice_field_over_time = np.zeros((ny, nx, time_steps))
u_over_time = np.zeros((ny, nx, time_steps))
v_over_time = np.zeros((ny, nx, time_steps))

# Define random dimensions for each plate
np.random.seed(0)  # For reproducibility
plate_widths = np.random.randint(3, 10, size=num_plates)
plate_heights = np.random.randint(3, 10, size=num_plates)

# Initialize positions of plates on the left side
initial_x = np.linspace(0, plate_widths[0], num_plates)
plate_y = ny // 2  # Middle of the grid for y position

# Function to create the ice field and velocity field with divergence in the x-direction only at time t
def update_fields(t):
    ice_field = np.zeros((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    
    # Apply divergence to each plate
    for i in range(num_plates):
        # Divergence in the x direction only
        x_divergence = divergence_rate * t * i
        plate_width = plate_widths[i]
        plate_height = plate_heights[i]
        
        x_start = int(initial_x[i] + x_divergence)
        x_end = int(initial_x[i] + x_divergence + plate_width)
        y_start = plate_y - plate_height // 2
        y_end = plate_y + plate_height // 2

        # Ensure the indices are within bounds
        x_start = max(x_start, 0)
        x_end = min(x_end, nx)
        y_start = max(y_start, 0)
        y_end = min(y_end, ny)

        ice_field[y_start:y_end, x_start:x_end] = 1
        
        # Set initial velocity field based on divergence (u component only for simplicity)
        u[y_start:y_end, x_start:x_end] = divergence_rate * t
        
    return ice_field, u, v

# Update the fields over time
for t in range(time_steps):
    ice_field, u, v = update_fields(t)
    ice_field_over_time[:, :, t] = ice_field
    u_over_time[:, :, t] = u
    v_over_time[:, :, t] = v
    
    
# Compute derivatives
def compute_derivatives(u, v):
    dudx = np.diff(u, axis=1) / dx
    dudy = np.diff(u, axis=0) / dy
    dvdx = np.diff(v, axis=1) / dx
    dvdy = np.diff(v, axis=0) / dy
    
    # Pad the derivatives to ensure they have consistent shapes
    dudx = np.pad(dudx, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    dudy = np.pad(dudy, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    dvdx = np.pad(dvdx, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    dvdy = np.pad(dvdy, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    
    return dudx, dudy, dvdx, dvdy

# Compute derivatives for each time step
dudx_over_time = np.zeros((ny, nx, time_steps))
dudy_over_time = np.zeros((ny, nx, time_steps))
dvdx_over_time = np.zeros((ny, nx, time_steps))
dvdy_over_time = np.zeros((ny, nx, time_steps))

for t in range(time_steps):
    u = u_over_time[:, :, t]
    v = v_over_time[:, :, t]
    dudx, dudy, dvdx, dvdy = compute_derivatives(u, v)
    dudx_over_time[:, :, t] = dudx
    dudy_over_time[:, :, t] = dudy
    dvdx_over_time[:, :, t] = dvdx
    dvdy_over_time[:, :, t] = dvdy

# Save the velocity derivatives
np.save("../artificial_fields/DUDX.npy", dudx_over_time)
np.save("../artificial_fields/DUDY.npy", dudy_over_time)
np.save("../artificial_fields/DVDX.npy", dvdx_over_time)
np.save("../artificial_fields/DVDY.npy", dvdy_over_time)
'''
'''

# Compute derivatives
def compute_derivatives(u, v):
    dudx = np.zeros((ny, nx-1))
    dudy = np.zeros((ny-1, nx))
    dvdx = np.zeros((ny, nx-1))
    dvdy = np.zeros((ny-1, nx))
    
    dudx[:, :] = np.diff(u, axis=1) / dx
    dudy[:, :] = np.diff(u, axis=0) / dy
    dvdx[:, :] = np.diff(v, axis=1) / dx
    dvdy[:, :] = np.diff(v, axis=0) / dy
    
    return dudx, dudy, dvdx, dvdy

# Compute derivatives for each time step
dudx_over_time = np.zeros((ny, nx-1, time_steps))
dudy_over_time = np.zeros((ny-1, nx, time_steps))
dvdx_over_time = np.zeros((ny, nx-1, time_steps))
dvdy_over_time = np.zeros((ny-1, nx, time_steps))

for t in range(time_steps):
    u = u_over_time[:, :, t]
    v = v_over_time[:, :, t]
    dudx, dudy, dvdx, dvdy = compute_derivatives(u, v)
    dudx_over_time[:, :, t] = dudx
    dudy_over_time[:, :, t] = dudy
    dvdx_over_time[:, :, t] = dvdx
    dvdy_over_time[:, :, t] = dvdy

# Visualization using animation for one of the derivatives (e.g., dudx)
fig, ax = plt.subplots()
def animate(t):
    ax.clear()
    ax.imshow(dudx_over_time[:, :, t], cmap='coolwarm', origin='lower', vmin=-0.1, vmax=0.1)
    ax.set_title(f'dudx at Time Step {t}')

ani = animation.FuncAnimation(fig, animate, frames=time_steps, repeat=False)
plt.show()

# save the velocity derivatives
np.save("artificial_fields/DUDX.npy", dudx_over_time)
np.save("artificial_fields/DUDY.npy", dudy_over_time)
np.save("artificial_fields/DVDX.npy", dvdx_over_time)
np.save("artificial_fields/DVDY.npy", dvdy_over_time)
'''



'''
# Initialize the ice field (Ice = 1; No ice = 0)
ice_field_over_time = np.zeros((ny,nx, time_steps))
#center_x, center_y = nx // 2, ny // 2

# Define random dimensions for each plate
np.random.seed(0)  # For reproducibility
plate_widths = np.random.randint(5, 20, size=num_plates)
plate_heights = np.random.randint(50, 1000, size=num_plates)

# Initialize positions of plates on the left side
initial_x = np.linspace(0, plate_widths[0], num_plates)
plate_y = ny // 2  # Middle of the grid for y position

# Function to create the ice field with divergence in the x-direction only at time t
def update_ice_field(t):
    ice_field = np.zeros((ny, nx))
    
    # Apply divergence to each plate
    for i in range(num_plates):
        # Divergence in the x direction only
        x_divergence = divergence_rate * t * i
        
        plate_width = plate_widths[i]
        plate_height = plate_heights[i]
        
        x_start = int(initial_x[i] + x_divergence)
        x_end = int(initial_x[i] + x_divergence + plate_width)
        y_start = plate_y - plate_height // 2
        y_end = plate_y + plate_height // 2

        # Ensure the indices are within bounds
        x_start = max(x_start, 0)
        x_end = min(x_end, nx)
        y_start = max(y_start, 0)
        y_end = min(y_end, ny)

        ice_field[y_start:y_end, x_start:x_end] = 1
    
    return ice_field

# Update the ice field over time
for t in range(time_steps):
    ice_field_over_time[:, :, t] = update_ice_field(t)

# Visualization using animation
fig, ax = plt.subplots()
def animate(t):
    ax.clear()
    ax.imshow(ice_field_over_time[:, :, t], cmap='Blues', origin='lower')
    ax.set_title(f'Ice Field at Time Step {t}')

ani = animation.FuncAnimation(fig, animate, frames=time_steps, repeat=False)
plt.show()
'''

'''
# Create the plates of ice at the center of the grid (can be moved)
ice_field[center_y - plate_width: center_y + plate_width, center_x - plate_width: center_x + plate_width] = 1


# Initialize the velocity field
u = np.zeros((ny, nx, time_steps))
v = np.zeros((ny, nx, time_steps))


# Apply divergence to the plates
for t in range(time_steps):
    divergence = np.linspace(-divergence_rate * t, divergence_rate * t, 2 * plate_width)
    
    u[center_y - plate_width:center_y + plate_width, center_x - plate_width:center_x + plate_width, t] = (np.outer(np.ones(2 * plate_width), divergence))
    
# Compute velocity derivatives
dudx = np.gradient(u, axis=1) / dx
dudy = np.gradient(u, axis=0) / dy
dvdx = np.gradient(v, axis=1) / dx
dvdy = np.gradient(v, axis=0) / dy


# Visualization (optional)
plt.imshow(ice_field, cmap='Blues', origin='lower')
plt.title('Initial Ice Field')
plt.colorbar(label='Ice presence (1: Ice, 0: No ice)')
plt.show()
'''


#%%

### fields_gen_test.py ###

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