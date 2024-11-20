# ------------------------------------------------------------------------------------
#   Synthetic data generation of u,v fields according to a deformations field
# ------------------------------------------------------------------------------------
#   Its purpose is to create files of velocities etc blablabla
#      Gotta clean this up
#
#
# ------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from datetime import datetime, timedelta
from scipy import sparse
from scipy.sparse import diags, block_diag, bmat,csr_matrix
from scipy.sparse.linalg import spsolve, eigs, norm
from scipy import optimize
from scipy.optimize import KrylovJacobian, BroydenFirst
import scipy.sparse as sp
from numpy.linalg import det
#from matplotlib import rc
import scienceplots
from matplotlib import rcParams


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

def create_sparse_double_matrix_dydx(N, dx, dy):
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

def create_sparse_double_matrix_dxdy(N, dx, dy):
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



def synthetic_divergence(F, dx, dy, vel_fig=False, div_fig=False):
    """
        Function that computes u, v fields based on the divergence/convergence field given.

        Args:
            F (np.ndarray): array of size (ny, nx, 2) where each ny,nx component represents a divergence (-) or convergence (+), in x (1) and y (2).
                            shape must be nx = ny
            dx, dy (str): resolution in x and y
            vel_fig (bool): =True if you want the velocities (quivers) figure; default False
            div_fig (bool): =True if you want the divergence figure; default False
            
        Returns:
            u, v (np.ndarray): the velocity fields u (ny, nx+1) and v (ny+1, nx)
            
    """

    # Get matrix shape
    N = len(F[0,:])
    N2 = N**2
    
    '''
    # Check if all elements of F add up to 0 (if not; it won't solve properly); sould fix this someday
    if np.sum(F) != 0:
        raise SystemExit(
                "Need to have a F field that adds up to 0."
            )
    '''
        
    # Flatten the F matrix (2*N2, 1)
    F_flat = F.flatten()
    
    print(F_flat)
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = create_sparse_matrix_dx(N)
    B_sparse = create_sparse_matrix_dy(N)
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[A_sparse, zero_matrix], 
                       [zero_matrix, B_sparse]])
    
    A_sparse_csr = A_sparse.tocsr()
    dense_section = A_sparse_csr[:10, :10].todense()  # Inspect a small section
    print(dense_section)
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))

    print('Your velocities have been computed in a diverging field!')

    # Centered finite differences
    u = U_grid
    v = V_grid
    zeros_j = np.zeros(len(v[0,:])) # boundary conditions
    v_jp1 = np.append(zeros_j, v[:-1,:]).reshape(v.shape)
    v_jm1 = np.append(v[1:,:], zeros_j).reshape(v.shape)
    zeros_i = np.zeros((u.shape[0], 1)) # boundary conditions
    u_ip1 = np.hstack((zeros_i,u[:,:-1]))
    u_im1 = np.hstack((u[:,1:],zeros_i))
    dudx = (u_ip1 - u_im1)/(2*dy)
    dvdy = (v_jp1 - v_jm1)/(2*dx)

    # Compute the divergence
    div = (dudx + dvdy)
    
    if vel_fig == True:
        # Show the U, V field in quivers
        plt.figure()
        plt.quiver(U_grid, V_grid, cmap=cm.viridis)
        plt.title("Optimized speeds fields")
        plt.show() 

    
    if div_fig == True:
        
         # Show the divergence field you prescibed
        plt.figure()
        plt.title("Recomputed Divergence field")
        plt.pcolormesh(div, cmap=cm.RdBu,vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
        
    U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)

    return U_grid_o, V_grid_o


def synthetic_shear(S, dx, dy, vel_fig=False, shear_fig=False):
    """
        Function that computes u, v fields based on the shearing field given.

        Args:
            S (np.ndarray): array of size (ny, nx, 2) where each ny,nx component represents a divergence (-) or convergence (+), in x (1) and y (2).
                            shape must be nx = ny
            dx, dy (str): resolution in x and y
            vel_fig (bool): =True if you want the velocities (quivers) figure; default False
            div_fig (bool): =True if you want the divergence figure; default False
            
        Returns:
            u, v (np.ndarray): the velocity fields u (ny, nx+1) and v (ny+1, nx)
            
    """

    # Get matrix shape
    N = len(S[0,:])
    N2 = N**2

    # Flatten the F matrix (2*N2, 1)
    S_flat = S.flatten()
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = create_sparse_matrix_dy(N)
    B_sparse = create_sparse_matrix_dx(N)
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[A_sparse, zero_matrix], 
                       [zero_matrix, B_sparse]])
    
    B_sparse_csr = B_sparse.tocsr()
    dense_section = B_sparse_csr[:10, :10].todense()  # Inspect a small section
    print(dense_section)
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, S_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    
    plt.figure()
    speed = np.sqrt(U_grid**2 + V_grid**2)
    #plt.pcolormesh(speed, cmap=cm.viridis, shading='auto')
    plt.quiver( U_grid, V_grid,color='k')
    plt.colorbar( label="speed")
    plt.title("Computed speeds fields")
    plt.show() 
    
    print('Your velocities have been computed in a diverging field!')
    u=U_grid
    v=V_grid
    # Centered finite differences
    zeros_j = np.zeros(len(u[0,:])) # boundary conditions
    u_jp1 = np.append(zeros_j, u[:-1,:]).reshape(u.shape)
    u_jm1 = np.append(u[1:,:], zeros_j).reshape(u.shape)
    
    zeros_i = np.zeros((v.shape[0], 1)) # boundary conditions
    v_ip1 = np.hstack((zeros_i,v[:,:-1]))
    v_im1 = np.hstack((v[:,1:],zeros_i))
    
    dudy = (u_jp1 - u_jm1)/(2*dy)
    dvdx = (v_ip1 - v_im1)/(2*dx)

    shear = dudy + dvdx

    plt.figure()
    plt.title("Recomputed shear field")
    plt.pcolormesh(shear, cmap=cm.RdBu, vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
    
    U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
        
    return U_grid_o, V_grid_o

def synthetic_deformations(F, dx, dy, vel_fig=False, shear_fig=False):
    """
        Function that computes u, v fields based on the deformations (div/conv + shear) field given.

        Args:
            S (np.ndarray): array of size (ny, nx, 2) where each ny,nx component represents a divergence (-) or convergence (+), in x (1) and y (2).
                            shape must be nx = ny
            dx, dy (str): resolution in x and y
            vel_fig (bool): =True if you want the velocities (quivers) figure; default False
            div_fig (bool): =True if you want the divergence figure; default False
            
        Returns:
            u, v (np.ndarray): the velocity fields u (ny, nx+1) and v (ny+1, nx)
            
    """

    # Get matrix shape
    N = len(F[0,:])
    N2 = N**2

    # Flatten the F matrix (2*N2, 1)
    F_flat = F.flatten()
    
    # Define the sparse finite differences matrices (2N, 2N)
    dy_sparse = create_sparse_double_matrix_dxdy(N, 1, 1.00001)
    dx_sparse = create_sparse_double_matrix_dydx(N, 1, 1.00001)
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[dx_sparse, zero_matrix], 
                       [zero_matrix, dy_sparse]])
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    
    plt.figure()
    speed = np.sqrt(U_grid**2 + V_grid**2)
    #plt.pcolormesh(speed, cmap=cm.viridis, shading='auto')
    plt.quiver( U_grid, V_grid,color='k')
    plt.colorbar( label="speed")
    plt.title("Computed speeds fields")
    plt.show() 
    
    print('Your velocities have been computed in a diverging field!')
    u=U_grid
    v=V_grid
    # Centered finite differences
    zeros_j = np.zeros(len(u[0,:])) # boundary conditions
    u_jp1 = np.append(zeros_j, u[:-1,:]).reshape(u.shape)
    u_jm1 = np.append(u[1:,:], zeros_j).reshape(u.shape)
    zeros_i = np.zeros((v.shape[0], 1)) # boundary conditions
    v_ip1 = np.hstack((zeros_i,v[:,:-1]))
    v_im1 = np.hstack((v[:,1:],zeros_i))
    dudy = (u_jp1 - u_jm1)/(2*dy)
    dvdx = (v_ip1 - v_im1)/(2*dx)
    
    zeros_j = np.zeros(len(v[0,:])) # boundary conditions
    v_jp1 = np.append(zeros_j, v[:-1,:]).reshape(v.shape)
    v_jm1 = np.append(v[1:,:], zeros_j).reshape(v.shape)
    zeros_i = np.zeros((u.shape[0], 1)) # boundary conditions
    u_ip1 = np.hstack((zeros_i,u[:,:-1]))
    u_im1 = np.hstack((u[:,1:],zeros_i))
    dudx = (u_ip1 - u_im1)/(2*dy)
    dvdy = (v_jp1 - v_jm1)/(2*dx)

    defo = dudx + dvdy+ dudy + dvdx

    plt.figure()
    plt.title("Recomputed deformations field")
    plt.pcolormesh(defo, cmap=cm.RdBu, vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
    
    U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
        
    return U_grid_o, V_grid_o


def save_fields(u, v, out, start_date, end_date):
    """
        Function that saves u, v fields to a given output file (needs to be outputxx; xx between 0 and 99) for specified "fake times".
        For now, the u and v don't change in time, but need to adapt the function for that
        
        Args:
            u (np.ndarray): u velocity field (ny, nx+1)
            v (np.ndarray): v velocity field (ny+1, nx)
            out (int): two number int; the experiment number to be saved, being outputxx     
            start_date, end_date (datetime): start and end time of the form datetime(yyyy, mm, dd)
    """
    
    time_delta = timedelta(hours=6)
    time_steps = int((end_date - start_date).total_seconds() // 3600 // 6) + 1
    
    # Where to put the files
    output_dir = f"/aos/home/fbeaudry/git/scale_test/output{out}"
    os.makedirs(output_dir, exist_ok=True)

    u = np.pad(u, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=0)
    v = np.pad(v, pad_width=((0, 1), (0, 0)), mode='constant', constant_values=0)
    
    # Create and save the fields over time
    current_time = start_date
    for t in range(time_steps):
        # Add a small noize factor
        #u_n = u+0.05*np.random.rand(N,N+1)
        #v_n = v+0.05*np.random.rand(N+1,N)
        
        u_n = u
        v_n = v
    
        # Filenames gossage
        file_suffix = f"{current_time.strftime('%Y_%m_%d_%H_%M')}.{out}"
        u_filename = os.path.join(output_dir, f"u{file_suffix}")
        v_filename = os.path.join(output_dir, f"v{file_suffix}")

        # Save the u, v files
        np.savetxt(u_filename, u_n, fmt='%.6f')
        np.savetxt(v_filename, v_n, fmt='%.6f')

        # Update the time
        current_time += time_delta
    

#%%
# -----------------------Construct deformation matrix-----------------------------------------------------#
def create_div(u, v, N, dx, dy):
    
    # Plot the initial speeds
    plt.figure()
    speed = np.sqrt(u**2 + v**2)
    #plt.pcolormesh(speed, cmap=cm.seismic, shading='auto')
    plt.quiver( u, v,color='k')
    plt.colorbar( label="speed")
    plt.title("Initial speeds fields")
    plt.show() 

    # Compute the div
    zeros_j = np.zeros(len(v[0,:])) # boundary conditions
    v_jp1 = np.append(zeros_j, v[:-1,:]).reshape(v.shape)
    v_jm1 = np.append(v[1:,:], zeros_j).reshape(v.shape)
    zeros_i = np.zeros((u.shape[0], 1)) # boundary conditions
    u_ip1 = np.hstack((zeros_i,u[:,:-1]))
    u_im1 = np.hstack((u[:,1:],zeros_i))
    dvdy = (v_jp1 - v_jm1)/(2*dy)
    dudx = (u_ip1 - u_im1)/(2*dx)

    # Plot the initial div
    div = dudx+dvdy
    plt.figure()
    plt.pcolormesh(div, cmap=cm.RdBu, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Initial Div')
    plt.show()
    
    # To fill the matrix if you only want div in u or v
    z_grid = np.zeros((N,N))
    
    if np.all(u == 0):
        # For v-div
        F = np.vstack([z_grid, dvdy])
        
    if np.all(v == 0):
        # For u-div
        F = np.vstack([dudx, z_grid])

    else:
        # For u-v div
        F = np.vstack([dudx, dvdy])

    u, v = synthetic_divergence(F, dx, dy, vel_fig=True, div_fig=True)
    
def create_shear(u, v, N, dx, dy):
    
    # Plot the initial speeds
    plt.figure()
    speed = np.sqrt(u**2 + v**2)
    #plt.pcolormesh(speed, cmap=cm.seismic, shading='auto')
    plt.quiver( u, v,color='k')
    plt.colorbar( label="speed")
    plt.title("Initial speeds fields")
    plt.show() 

    # Compute the shear
    zeros_j = np.zeros(len(u[0,:])) # boundary conditions
    u_jp1 = np.append(zeros_j, u[:-1,:]).reshape(u.shape)
    u_jm1 = np.append(u[1:,:], zeros_j).reshape(u.shape)
    zeros_i = np.zeros((v.shape[0], 1)) # boundary conditions
    v_ip1 = np.hstack((zeros_i,v[:,:-1]))
    v_im1 = np.hstack((v[:,1:],zeros_i))
    dudy = (u_jp1 - u_jm1)/(2*dy)
    dvdx = (v_ip1 - v_im1)/(2*dx)

    # Plot the initial shear
    shear = dudy+dvdx
    plt.figure()
    plt.pcolormesh(shear, cmap=cm.RdBu, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Initial Shear')
    plt.show()
    
    # To fill the matrix if you only want shear in u or v
    z_grid = np.zeros((N,N))
    
    if np.all(u == 0):
        # For v-shear
        F = np.vstack([z_grid, dvdx])
        
    if np.all(v == 0):
        # For u-shear
        F = np.vstack([dudy, z_grid])

    else:
        # For u-v shear
        F = np.vstack([dudy, dvdx])

    u, v = synthetic_shear(F, dx, dy, vel_fig=True, shear_fig=True)



def create_deformations(u, v, N, dx, dy):
    
    # Plot the initial speeds
    plt.figure()
    speed = np.sqrt(u**2 + v**2)
    #plt.pcolormesh(speed, cmap=cm.seismic, shading='auto')
    plt.quiver( u, v,color='k')
    plt.colorbar( label="speed")
    plt.title("Initial speeds fields")
    plt.show() 

    # Centered finite differences
    zeros_j = np.zeros(len(u[0,:])) # boundary conditions
    u_jp1 = np.append(zeros_j, u[:-1,:]).reshape(u.shape)
    u_jm1 = np.append(u[1:,:], zeros_j).reshape(u.shape)
    zeros_i = np.zeros((v.shape[0], 1)) # boundary conditions
    v_ip1 = np.hstack((zeros_i,v[:,:-1]))
    v_im1 = np.hstack((v[:,1:],zeros_i))
    dudy = (u_jp1 - u_jm1)/(2*dy)
    dvdx = (v_ip1 - v_im1)/(2*dx)
    
    zeros_j = np.zeros(len(v[0,:])) # boundary conditions
    v_jp1 = np.append(zeros_j, v[:-1,:]).reshape(v.shape)
    v_jm1 = np.append(v[1:,:], zeros_j).reshape(v.shape)
    zeros_i = np.zeros((u.shape[0], 1)) # boundary conditions
    u_ip1 = np.hstack((zeros_i,u[:,:-1]))
    u_im1 = np.hstack((u[:,1:],zeros_i))
    dudx = (u_ip1 - u_im1)/(2*dy)
    dvdy = (v_jp1 - v_jm1)/(2*dx)

    defo = dudx + dvdy + dudy + dvdx

    du = dudx+dudy
    dv = dvdx+dvdy

    # Plot the initial shear
    plt.figure()
    plt.pcolormesh(defo, cmap=cm.RdBu, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Initial Deformations')
    plt.show()
    
    # To fill the matrix if you only want shear in u or v
    z_grid = np.zeros((N,N))
    
    if np.all(u == 0):
        # For v-shear
        F = np.vstack([z_grid, dv])
        
    if np.all(v == 0):
        # For u-shear
        F = np.vstack([du, z_grid])

    else:
        # For u-v shear
        F = np.vstack([du, dv])

    u, v = synthetic_deformations(F, dx, dy, vel_fig=True, shear_fig=True)

        

N= 130
dy, dx = 1,1 #to be defined

# u-div example
v = np.zeros((N,N))
u = np.ones((N,N))
u[:, 2:4] = -1


# v-u-div example
u = np.ones((N,N))
v = np.ones((N,N))
v[10:11, :] = -1
u[:, 10:20] = -1
v = np.zeros((N,N))
#create_div(u, v, N, dx, dy)

# v-shear example
u = np.zeros((N,N))
v = np.ones((N,N))
v[:, 10:11] = -1

mean = 0
std = 0.1
u = u + np.random.normal(mean, std, u.shape)
v = v + np.random.normal(mean, std, v.shape)
#save_fields(u, v, '61', datetime(2002, 1, 1), datetime(2002, 1, 31, 18))
#create_div(u, v, N, dx, dy)
#print("NEXT ---------------------------")
#create_shear(u, v, N, dx, dy)
#print("NEXT ---------------------------")

"""
# u-v shear+div example
# Initialize velocity components
u = np.zeros((N, N))
v = np.ones((N, N))
# Introduce shear: horizontal shearing line
v[:, 10:11] = -1  # Shearing line in the middle
# Add divergence: radial flow pattern
x = np.linspace(-N/2, N/2, N) * dx
y = np.linspace(-N/2, N/2, N) * dy
X, Y = np.meshgrid(x, y)
u += 0.1 * X / (np.sqrt(X**2 + Y**2) + 1e-3)  # Radial u component
v += 0.1 * Y / (np.sqrt(X**2 + Y**2) + 1e-3)  # Radial v component
# Add nonlinearities: sinusoidal and quadratic variations
u += 0.05 * np.sin(2 * np.pi * Y / N)
v += 0.05 * (X / N)**2



x = np.linspace(-N/2, N/2, N) * dx
y = np.linspace(-N/2, N/2, N) * dy
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2) + 1e-3  # Avoid division by zero
theta = np.arctan2(Y, X)
# 1. Vortex structure
u_vortex = -Y / r
v_vortex = X / r
# 2. Divergence (radial expansion with sinusoidal modulation)
u_div = 0.1 * X * (1 + 0.5 * np.sin(2 * np.pi * Y / N))
v_div = 0.1 * Y * (1 + 0.5 * np.cos(2 * np.pi * X / N))
# 3. Random perturbations
np.random.seed(42)  # For reproducibility
u_noise = 0.05 * np.random.randn(N, N)
v_noise = 0.05 * np.random.randn(N, N)
# 4. Localized Gaussian anomalies
u_gaussian = 0.2 * np.exp(-((X - N/4)**2 + (Y - N/4)**2) / (2 * (N/10)**2))
v_gaussian = -0.2 * np.exp(-((X + N/4)**2 + (Y + N/4)**2) / (2 * (N/10)**2))
# Combine components
u = u_vortex + u_div + u_noise + u_gaussian
v = v_vortex + v_div + v_noise + v_gaussian

"""

#v = np.zeros((N,N))
#u = np.ones((N,N))
#u[:, 2:4] = -1


#save_fields(u, v, '60', datetime(2002, 1, 1), datetime(2002, 1, 31, 18))
#create_div(u, v, N, dx, dy)

print("NEXT ---------------------------")
#create_shear(u, v, N, dx, dy)
print("NEXT ---------------------------")



#create_deformations(u, v, N, dx, dy)

#create_shear(u, v, N, dx, dy)

# u-shear example
#u=np.ones((N,N))
#u[10:20, :] = -1
#u[30:40, :] = -1
#u[50:60, :] = -1
#u[70:80, :] = -1
#u[90:100, :] = -1
#v=np.zeros((N,N))
#create_shear(u, v, N, dx, dy)


#%%
# SCALING TEST
dx, dy = 1, 1
u_center = 0.5 * (u[:, :-1] + u[:, 1:])  # Average along x
v_center = 0.5 * (v[:-1, :] + v[1:, :])  # Average along y

du_dx = (u_center[:, 1:] - u_center[:, :-1])[1:-1,:] / dx  # Gradient of u in x-direction
du_dy = (u_center[1:, :] - u_center[:-1, :])[1:,1:] / dy  # Gradient of u in y-direction
dv_dx = (v_center[:, 1:] - v_center[:, :-1])[1:,1:] / dx  # Gradient of v in x-direction
dv_dy = (v_center[1:, :] - v_center[:-1, :])[:,1:-1] / dy  # Gradient of v in y-direction

print(du_dx)

L = 2  # Example: coarse-graining factor
L_values = [2, 4, 8, 16, 32, 64]
L_values = [2, 4, 8, 16, 32, 64]
deformation_means = []

N = np.shape(du_dx)[0]
deformations_L = []

for v in range(len(L_values)):
    L = L_values[v]
    coarse_defos = []
    counter=0
    du_dx_moy, du_dy_moy, dv_dx_moy, dv_dy_moy = 0, 0, 0, 0
    for i in range(np.shape(du_dx)[0]):
        for j in range(np.shape(du_dx)[1]):
            if ((i)%(L/2) == 0) and ((j)%(L/2) == 0):
                coarse_du_dx = du_dx[i:i+L, j:j+L]
                if np.shape(coarse_du_dx)==(L,L):
                    counter+=1
                    print(coarse_du_dx)
                    du_dx_sum = np.nansum(coarse_du_dx)
                    du_dx_bool_sum = np.nansum(np.array(coarse_du_dx, dtype=bool))
                    if du_dx_bool_sum != 0:
                        print('sum', du_dx_sum)
                        print('bool sum',du_dx_bool_sum)
                        du_dx_moy = du_dx_sum/du_dx_bool_sum
                        print('moy', du_dx_moy)
                
                coarse_du_dy = du_dy[i:i+L, j:j+L]
                du_dy_sum = np.nansum(coarse_du_dy)
                du_dy_bool_sum = np.nansum(np.array(coarse_du_dy, dtype=bool))
                if du_dy_bool_sum != 0:
                    print('bool sum',du_dy_bool_sum)
                    du_dy_moy = du_dy_sum/du_dy_bool_sum
                
                coarse_dv_dx = dv_dx[i:i+L, j:j+L]
                dv_dx_sum = np.nansum(coarse_dv_dx)
                dv_dx_bool_sum = np.nansum(np.array(coarse_dv_dx, dtype=bool))
                if dv_dx_bool_sum != 0:
                    print('bool sum', dv_dx_bool_sum)
                    dv_dx_moy = dv_dx_sum/dv_dx_bool_sum
                
                coarse_dv_dy = dv_dy[i:i+L, j:j+L]
                dv_dy_sum = np.nansum(coarse_dv_dy)
                dv_dy_bool_sum = np.nansum(np.array(coarse_dv_dy, dtype=bool))
                if dv_dy_bool_sum != 0:
                    print('bool sum', dv_dy_bool_sum)
                    dv_dy_moy = dv_dy_sum/dv_dy_bool_sum
                    #print('hi', dv_dy_moy)
                
                
                divergence = du_dx_moy + dv_dy_moy
                shear = du_dy_moy - dv_dx_moy
                deformation = np.sqrt(divergence**2 + shear**2)
                coarse_defos = np.append(coarse_defos, deformation)
    
    
    print(v)
    print("count", counter)
    #print(coarse_defos)
    print(np.nanmean(coarse_defos))
    deformations_L= np.append(deformations_L, np.nanmean(coarse_defos))
    print(deformations_L)
    
plt.rcParams.update({'font.size': 16})
with plt.style.context(['science', 'no-latex']):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('Spatial scaling')
    ax.grid(True, which='both')
    ax.scatter(L_values, deformations_L, c='tab:orange',s=60, alpha=1, edgecolors="k", zorder=1000)
    ax.set_xlabel('Spatial scale (nu)')
    ax.set_ylabel('$\\langle\\epsilon_{tot}\\rangle$')
    ax.set_xscale("log")  # Set logarithmic scale on the x-axis
    ax.set_yscale("log")  # Set logarithmic scale on the y-axis
    fig.savefig("Spatial_Scaling2.png")  # Save the figure
    plt.close()
#plt.loglog(L_values, deformation_means)
#%%
"""
dy_sparse = create_sparse_double_matrix_dydx(4, 1, 1)

# Convert the sparse matrix to a dense matrix for visualization
dense_matrix = dy_sparse.toarray()

determinant = det(dense_matrix)
print(f"Determinant: {determinant}")

# Create a color map: 1 -> blue, -1 -> red, 0 -> white
colors = np.zeros(dense_matrix.shape)
colors[dense_matrix == 1] = 1  # Mark 1's as blue
colors[dense_matrix == -1] = -1  # Mark -1's as red

# Plot the matrix using imshow
plt.figure(figsize=(6, 6))
plt.imshow(colors, cmap='bwr', origin='upper', interpolation='nearest')
plt.colorbar(label="Value")
plt.title(f"Visualization of 1's (blue) and -1's (red) in the Matrix (N={N})")
plt.show()
"""

#%%

'''
# Construct the full F matrix
#pattern = [-1,0 ,-1 ,1, 1, 1, 1,-1 ,0,-1]
#pattern = [1, -1, 1, -1, 1, -1, 1, -1]
#pattern = [-1, -1, -1, 6, -1, -1, -1]
#pattern = [-1, 2, -2, 2, -2, 2, -2, 2, -1]
#pattern = [0, 3, -3, 3, -3, 3, -3, 3, -3, 3, -3, 0]
#pattern = [0,3, 3, 3, 3, 3, 3, 3, 3, 3, 3,0]
#pattern = [2, 2, 2, 2, 2, 2]
#F_grid = np.tile(pattern, (len(pattern), 1))
pattern= (-1)*np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],[2,2,2,2,2,2,2,2,2,2],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[2,2,2,2,2,2,2,2,2,2],[-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
#pattern= (-1)*np.array([[0,0,0,0,0,0,0,0,0,0],[0,-1,-1,-1,-1,-1,-1,-1,-1,0],[0,-1,-1,-1,-1,-1,-1,-1,-1,0],[0,1,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,1,0],[0,-1,-1,-1,-1,-1,-1,-1,-1,0],[0,-1,-1,-1,-1,-1,-1,-1,-1,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
#F_grid = (1/2)*(np.array([[2,2,2,2,2,2], [2,2,2,2,2,2], [2,2,2,2,2,2], [-2,-2,-2,-2,-2,-2], [-2,-2,-2,-2,-2,-2], [-2,-2,-2,-2,-2,-2]]))
#F_grid = np.array([[2, 2, 2, 2, 2, 2], [-2,-2, -2, -2, -2, -2], [2, 2, 2, 2, 2, 2], [-2, -2, -2, -2, -2, -2], [2, 2, 2, 2, 2, 2], [-2, -2, -2, -2,-2,-2]])


N = len(F_grid[0])
N = 30
N2 = N**2

F_grid = np.tile(pattern, (N // pattern.shape[0], N // pattern.shape[1]))



N = 100  # Grid size (500x500)
Lx = 5.0  # Domain length in x-direction
Ly = 5.0  # Domain length in y-direction
dx = Lx / (N-1)
dy = Ly / (N-1)

# Create a shear field (S) with a sharp horizontal shear line in the middle
F_grid = np.zeros((N, N))  # Initialize shear field
mid = N // 2  # Midpoint of the grid
F_grid[mid-10:mid+10, :] = -5  # Sharp negative shear band
F_grid[mid+10:mid+20, :] = 5   
F_grid[mid-20:mid-30, :] = -5  # Sharp negative shear band
F_grid[mid+10:mid+20, :] = 5   
z_grid = np.zeros((N,N))
F = np.vstack([z_grid, F_grid])
u, v = synthetic_shear(F, 1, 1, vel_fig=True, shear_fig=True)

pattern = [0, -1, -1,0,0, -2, -2, 0,0,1, 1, 0]
F_grid = np.tile(pattern, (len(pattern), 1))
N = len(F_grid[0])
N2 = N**2
z_grid = np.zeros((N,N))
F = np.vstack([F_grid, z_grid])
#u, v = synthetic_divergence(F, 1, 1, vel_fig=True, div_fig=True)
'''

"""
def synthetic_shear(S, dx, dy, vel_fig=False, shear_fig=False):

        Function that computes u, v fields based on the shearing field given.

        Args:
            S (np.ndarray): array of size (ny, nx, 2) where each ny,nx component represents a divergence (-) or convergence (+), in x (1) and y (2).
                            shape must be nx = ny
            dx, dy (str): resolution in x and y
            vel_fig (bool): =True if you want the velocities (quivers) figure; default False
            div_fig (bool): =True if you want the divergence figure; default False
            
        Returns:
            u, v (np.ndarray): the velocity fields u (ny, nx+1) and v (ny+1, nx)
            
    

    # Get matrix shape
    N = len(S[0,:])
    N2 = N**2
    Lx=5.0
    Ly=5.0
    dx = Lx / (N-1)
    dy = Ly / (N-1)

    # Flatten the F matrix (2*N2, 1)
    S_flat = S.flatten()
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = create_sparse_matrix_dy(N)
    B_sparse = create_sparse_matrix_dx(N)
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[A_sparse, zero_matrix], 
                       [zero_matrix, B_sparse]])
    
    #B_sparse_csr = B_sparse.tocsr()
    #dense_section = B_sparse_csr[:10, :10].todense()  # Inspect a small section
    #print(dense_section)
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, S_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    
    plt.figure()
    speed = np.sqrt(U_grid**2 + V_grid**2)
    #plt.pcolormesh(speed, cmap=cm.viridis, shading='auto')
    plt.quiver( U_grid, V_grid,color='k')
    plt.colorbar( label="speed")
    plt.title("Computed speeds fields")
    plt.show() 
    #plt.figure()
    #plt.quiver( U_grid, V_grid,color='k')
    #plt.title("Computed speeds fields")
    #plt.show() 
    
    print('Your velocities have been computed in a diverging field!')
    u=U_grid
    v=V_grid
    # Centered finite differences
    zeros_j = np.zeros(len(u[0,:])) # boundary conditions
    u_jp1 = np.append(zeros_j, u[:-1,:]).reshape(u.shape)
    u_jm1 = np.append(u[1:,:], zeros_j).reshape(u.shape)
    
    zeros_i = np.zeros((v.shape[0], 1)) # boundary conditions
    v_ip1 = np.hstack((zeros_i,v[:,:-1]))
    v_im1 = np.hstack((v[:,1:],zeros_i))
    
    dudy = (u_jp1 - u_jm1)/(2*dy)
    dvdx = (v_ip1 - v_im1)/(2*dx)

    shear = dudy + dvdx

    plt.figure()
    plt.title("Recomputed shear field")
    plt.pcolormesh(shear, vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
    
    
    def res_shear(uv):
        
            #Function that computes the residual between given F field and computed F field
            
        
        # Call function that computes the shearing field
        N = int(np.sqrt((len(uv))/2))
        u = uv[:N*N].reshape((N, N))
        v = uv[N*N:].reshape((N, N))
    
        # Compute the gradient components
        # verif the resolution (spacing between values)
        dudx = (u[:, 1:] - u[:, :-1]) / dx
        dudx = np.pad(dudx, ((0, 0), (1, 0)), mode='edge')
        dudy = (u[1:, :] - u[:-1, :]) / dy
        dudy = np.pad(dudy, ((1, 0), (0, 0)), mode='edge')
        dvdx = (v[:, 1:] - v[:, :-1]) / dx
        dvdx = np.pad(dvdx, ((0, 0), (1, 0)), mode='edge')
        dvdy = (v[1:, :] - v[:-1, :]) / dy
        dvdy = np.pad(dvdy, ((1, 0), (0, 0)), mode='edge')

        # Compute the shear
        #shear = np.sqrt((dudx - dvdy)**2 + (dudy + dudx)**2)
        shear = dudy+dvdx
        
        # This is kinda weird, NEED TO CHECK THE MATH to make sure it's legit (I think it might miss the v-component of the F)
        #residual = shear - S[:N, :] 
        residual = shear - S[N:, :] 
        residual_flat = residual.flatten()
        print(np.mean(residual_flat))
        #residual_padded = np.hstack([residual_flat, np.zeros_like(residual_flat)])
        residual_padded = np.hstack([ np.zeros_like(residual_flat),residual_flat])
        return residual_padded
       
    # To remove; test with actually correct the fields    
    #UV = np.array([[1, 1, 1, 1, 1, 1], [-1,-1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1], [-1,-1, -1, -1, -1, -1],[1, 1, 1, 1, 1, 1], [-1,-1, -1, -1, -1, -1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]).flatten()
    #UV = np.zeros_like(UV)
    
    # Optimize the field, and make sure the recomputed and not recomputed ones match    
    UV_opt = optimize.newton_krylov(res_shear, UV+0.0001*np.random.rand(N2*2),method="lgmres", inner_maxiter=20, iter=200,f_tol=1e-4)
        
    U_grid_o = UV_opt[:N2].reshape((N,N))
    V_grid_o = UV_opt[N2:].reshape((N,N))
    
    # Compute the final shear
    # Normalize the u and v
    u = (U_grid_o - np.min(U_grid_o)) / (np.max(U_grid_o) - np.min(U_grid_o))
    v = (V_grid_o - np.min(V_grid_o)) / (np.max(V_grid_o) - np.min(V_grid_o))
    #u = U_grid_o
    #v = V_grid_o
    print(u)
    dudx = (u[:, 1:] - u[:, :-1]) / dx
    dudx = np.pad(dudx, ((0, 0), (1, 0)), mode='edge')
    dudy = (u[1:, :] - u[:-1, :]) / dy
    dudy = np.pad(dudy, ((1, 0), (0, 0)), mode='edge')
    dvdx = (v[:, 1:] - v[:, :-1]) / dx
    dvdx = np.pad(dvdx, ((0, 0), (1, 0)), mode='edge')
    dvdy = (v[1:, :] - v[:-1, :]) / dy
    dvdy = np.pad(dvdy, ((1, 0), (0, 0)), mode='edge')
    #shear = np.sqrt((dudx - dvdy)**2 + (dudy + dudx)**2)
    #shear = (shear - np.min(shear)) / (np.max(shear) - np.min(shear))
    shear = dudy + dvdx
    
    if vel_fig == True:
        # Show the U, V fiel in quivers
        plt.figure()
        speed = np.sqrt(U_grid**2 + V_grid**2)
        plt.pcolormesh(speed, cmap=cm.viridis, shading='auto')
        plt.quiver( U_grid, V_grid,color='k')
        plt.colorbar( label="speed")
        plt.title("Initial speeds fields")
        plt.show() 
        
        plt.figure()
        speed = np.sqrt(U_grid_o**2 + V_grid_o**2)
        plt.pcolormesh(speed, cmap=cm.viridis, shading='auto')
        plt.quiver( U_grid_o, V_grid_o,color='k')
        plt.colorbar( label="speed")
        plt.title("Optimized speeds fields")
        plt.show() 
        
    if shear_fig == True:
        # Show the divergence field you prescibed
        plt.figure()
        plt.title("Shear field")
        plt.pcolormesh(S, vmin=-5, vmax=5)
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.title("Recomputed shear field")
        plt.pcolormesh(shear, vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
    U_grid_o = np.hstack([np.zeros((N, 1)), U_grid_o]) # so that the shape is (ny, nx+1)
    V_grid_o = np.vstack([np.zeros((1, N)), V_grid_o]) # so that the shape is (ny+1, nx)
    
    U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
        
    return U_grid_o, V_grid_o
"""

'''
u = np.array([[+1, +1, +1, +1, +1, +1],[-1, -1, -1, -1, -1, -1], [+1, +1, +1, +1, +1, +1],[-1, -1, -1, -1, -1, -1],[+1, +1, +1, +1, +1, +1],[-1, -1, -1, -1, -1, -1]])
v = np.zeros((6,6))
dx, dy =1,1

plt.figure()
plt.quiver( u, v, cmap=cm.viridis)
plt.title("Optimized speeds fields")
plt.show() 

dudx = (u[:, 1:] - u[:, :-1]) / dx
dudx = np.pad(dudx, ((0, 0), (1, 0)), mode='edge')
dudy = (u[1:, :] - u[:-1, :]) / dy
dudy = np.pad(dudy, ((1, 0), (0, 0)), mode='edge')
dvdx = (v[:, 1:] - v[:, :-1]) / dx
dvdx = np.pad(dvdx, ((0, 0), (1, 0)), mode='edge')
dvdy = (v[1:, :] - v[:-1, :]) / dy
dvdy = np.pad(dvdy, ((1, 0), (0, 0)), mode='edge')

# Compute the shear
shear = np.sqrt((dudx - dvdy)**2 + (dudy + dudx)**2)

plt.figure()
plt.title("Recomputed shear field")
plt.pcolormesh(shear)
plt.colorbar()
plt.show()
'''