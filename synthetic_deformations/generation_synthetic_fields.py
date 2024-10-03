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

def create_sparse_matrix_u_j(N):
    # Size of the sparse matrix
    size = N * N

    # Create the diagonals for the top-left to bottom-right direction
    diagonals = [-np.ones(size - N), np.ones(size - N)]
    
    # Offset positions for the diagonals
    offsets = [N, -N]  # N for +1, -N for -1

    # Create the sparse diagonal matrix
    sparse_matrix = diags(diagonals, offsets, shape=(size, size), format='csr')

    return sparse_matrix


def create_sparse_matrix_v_i(N):
    block_count=N
    # Create a single NxN block matrix with the specified diagonal pattern
    diagonals = [-np.ones(N), np.zeros(N), np.ones(N)]  # +1, 0, -1
    offsets = [1, 0, -1]  # +1 on the super diagonal, 0 on the main diagonal, -1 on the sub diagonal

    # Create a single NxN block
    block = diags(diagonals, offsets, shape=(N, N))

    # Create a larger block matrix made of block_count x block_count blocks
    large_matrix = bmat([[block if i == j else np.zeros((N, N)) for j in range(block_count)] for i in range(block_count)])

    return large_matrix



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
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dx
    B_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dy 
    
    diagonals = np.array([-1, 0,1])  # Coefficients for the left and right differences
    offsets = np.array([-1, 0,1])     # Offset positions for the diagonals
    A_sparse = diags(diagonals, offsets, shape=(N2, N2)) / (2 * dx)
    B_sparse = diags(diagonals, offsets, shape=(N2, N2)) / (2 * dy)
    # Get the complete sparse finite differences matrix (2*2N, 2*2N)
    AB_sparse = block_diag([A_sparse, B_sparse])
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))

    print('Your velocities have been computed in a diverging field!')

    # The optimized field is already computed since we have a linear problem
    UV_opt = UV
    
    # Optimized velocity fields
    U_grid_o = UV_opt[:N2].reshape((N,N))
    V_grid_o = UV_opt[N2:].reshape((N,N))
    
    # Compute the final divergence
    #dudx = np.gradient(U_grid_o, axis=1)
    #dudy = np.gradient(U_grid_o, axis=0)
    #dvdx = np.gradient(V_grid_o, axis=1)
    #dvdy = np.gradient(V_grid_o, axis=0)
    u = U_grid_o
    v = V_grid_o
    zeros_j = np.zeros(len(v[0,:])) # boundary conditions
    v_jp1 = np.append(zeros_j, v[:-1,:]).reshape(v.shape)
    v_jm1 = np.append(v[1:,:], zeros_j).reshape(v.shape)
    
    zeros_i = np.zeros(len(u[:,0])) # boundary conditions
    u_ip1 = np.append(zeros_i, u[:-1,:]).reshape(u.shape)
    u_im1 = np.append(u[1:,:], zeros_i).reshape(u.shape)
    
    dudx = (u_ip1 - u_im1)/(2*dy)
    dvdy = (v_jp1 - v_jm1)/(2*dx)
    '''
    dudx = (u[:, 1:] - u[:, :-1]) / dx
    dudx = np.pad(dudx, ((0, 0), (1, 0)), mode='edge')
    dudy = (u[1:, :] - u[:-1, :]) / dy
    dudy = np.pad(dudy, ((1, 0), (0, 0)), mode='edge')
    dvdx = (v[:, 1:] - v[:, :-1]) / dx
    dvdx = np.pad(dvdx, ((0, 0), (1, 0)), mode='edge')
    dvdy = (v[1:, :] - v[:-1, :]) / dy
    dvdy = np.pad(dvdy, ((1, 0), (0, 0)), mode='edge')
    '''
    # Compute the divergence
    div = (dudx + dvdy)
    
    if vel_fig == True:
        # Show the U, V field in quivers
        plt.figure()
        plt.quiver(U_grid_o, V_grid_o, cmap=cm.viridis)
        plt.title("Optimized speeds fields")
        plt.show() 

    
    if div_fig == True:
        # Show the divergence field you prescibed
        plt.figure()
        plt.title("Initial Divergence field")
        plt.pcolormesh(F)
        plt.colorbar()
        plt.show()
        
         # Show the divergence field you prescibed
        plt.figure()
        plt.title("Recomputed Divergence field")
        plt.pcolormesh(div)
        plt.colorbar()
        plt.show()
        
    U_grid_o = np.hstack([np.zeros((N, 1)), U_grid_o]) # so that the shape is (ny, nx+1)
    V_grid_o = np.vstack([np.zeros((1, N)), V_grid_o]) # so that the shape is (ny+1, nx)

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
    Lx=5.0
    Ly=5.0
    dx = Lx / (N-1)
    dy = Ly / (N-1)

    # Flatten the F matrix (2*N2, 1)
    S_flat = S.flatten()
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = create_sparse_matrix_u_j(N)
    B_sparse = create_sparse_matrix_v_i(N)
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
    
    zeros_i = np.zeros(len(v[:,0])) # boundary conditions
    v_ip1 = np.append(zeros_i, v[:-1,:]).reshape(v.shape)
    v_im1 = np.append(v[1:,:], zeros_i).reshape(v.shape)
    
    dudy = (u_jp1 - u_jm1)/(2*dy)
    dvdx = (v_ip1 - v_im1)/(2*dx)
    '''
    dudx = (u[:, 1:] - u[:, :-1]) / dx
    dudx = np.pad(dudx, ((0, 0), (1, 0)), mode='edge')
    dudy = (u[1:, :] - u[:-1, :]) / dy
    dudy = np.pad(dudy, ((1, 0), (0, 0)), mode='edge')
    dvdx = (v[:, 1:] - v[:, :-1]) / dx
    dvdx = np.pad(dvdx, ((0, 0), (1, 0)), mode='edge')
    dvdy = (v[1:, :] - v[:-1, :]) / dy
    dvdy = np.pad(dvdy, ((1, 0), (0, 0)), mode='edge')

    dudx = (u[:, 2:] - u[:, :-2]) / (2 * dx)  # Central difference in x
    dudx = np.pad(dudx, ((0, 0), (1, 1)), mode='edge')  # Pad edges to maintain shape
    dudy = (u[:-2, :] - u[2:, :] ) / (2 * dy)  # Central difference in y
    print(u[2:, :])
    print( u[:-2, :])
    dudy = np.pad(dudy, ((1, 1), (0, 0)), mode='edge')  # Pad edges to maintain shape
    # Calculate central differences for v
    dvdx = (v[:, 2:] - v[:, :-2]) / (2 * dx)  # Central difference in x
    dvdx = np.pad(dvdx, ((0, 0), (1, 1)), mode='edge')  # Pad edges to maintain shape
    dvdy = (v[2:, :] - v[:-2, :]) / (2 * dy)  # Central difference in y
    dvdy = np.pad(dvdy, ((1, 1), (0, 0)), mode='edge')  # Pad edges to maintain shape
    #shear = np.sqrt((dudx - dvdy)**2 + (dudy + dudx)**2)
    #shear = (shear - np.min(shear)) / (np.max(shear) - np.min(shear))
    '''
    shear = dudy + dvdx

    plt.figure()
    plt.title("Recomputed shear field")
    plt.pcolormesh(shear, vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
    
    """
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
    """
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
    
    # Create and save the fields over time
    current_time = start_date
    for t in range(time_steps):
        # Add a small noize factor
        u_n = u+0.05*np.random.rand(N,N+1)
        v_n = v+0.05*np.random.rand(N+1,N)
    
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

#save_fields(u, v, "60", start_date = datetime(2002, 1, 1), end_date = datetime(2002, 1, 31, 18))

N=30
u=np.ones((N,N))
u[10:20, :] = -1
#u[30:40, :] = -1
#u[50:60, :] = -1
#u[70:80, :] = -1
#u[90:100, :] = -1
v=np.zeros((N,N))

plt.figure()
speed = np.sqrt(u**2 + v**2)
#plt.pcolormesh(speed, cmap=cm.viridis, shading='auto')
plt.quiver( u, v,color='k')
plt.colorbar( label="speed")
plt.title("Initial speeds fields")
plt.show() 

zeros_j = np.zeros(len(u[0,:])) # boundary conditions
u_jp1 = np.append(zeros_j, u[:-1,:]).reshape(u.shape)
u_jm1 = np.append(u[1:,:], zeros_j).reshape(u.shape)
    
zeros_i = np.zeros(len(v[:,0])) # boundary conditions
v_ip1 = np.append(zeros_i, v[:-1,:]).reshape(v.shape)
v_im1 = np.append(v[1:,:], zeros_i).reshape(v.shape)
    
dudy = (u_jp1 - u_jm1)/(2*dy)
dvdx = (v_ip1 - v_im1)/(2*dx)

shear = dudy
plt.figure()
plt.pcolormesh(shear)
plt.colorbar()
plt.title('Initial Shear')
plt.show()

z_grid = np.zeros((N,N))
F = np.vstack([shear, z_grid])
u, v = synthetic_shear(F, 1, 1, vel_fig=True, shear_fig=True)

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