import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
from datetime import datetime, timedelta

from scipy import sparse
from scipy.sparse import diags, block_diag, bmat
from scipy.sparse.linalg import spsolve
from scipy import optimize
from scipy.optimize import KrylovJacobian

def compute_div(uv):
    """
        Function that computes the diverging field from u and v fields
        
        Args:
            uv (np.ndarray): array of size (ny, nx, 2)
            
        Returns:
            div (np.ndarray): array of size (ny, nx)
    """
    # Do some reshaping
    print(len(uv))
    N = int(np.sqrt((len(uv))/2))
    u = uv[:N*N].reshape((N, N))
    v = uv[N*N:].reshape((N, N))
    
    # Compute the gradient components
    # verif the resolution (spacing between values)
    dudx = np.gradient(u, axis=1)
    dudy = np.gradient(u, axis=0)
    dvdx = np.gradient(v, axis=1)
    dvdy = np.gradient(v, axis=0)

    # Compute the divergence
    div = dudx + dvdy
    return div

# Create the divergence fields
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
    
    # Check if all elements of F add up to 0 (if not; it won't solve properly); sould fix this someday
    if np.sum(F) != 0:
        raise SystemExit(
                "Need to have a F field that adds up to 0."
            )
        
    # Flatten the F matrix (2*N2, 1)
    F_flat = F.flatten()
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dx
    B_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dy 
    
    # Get the complete sparse finite differences matrix (2*2N, 2*2N)
    AB_sparse = block_diag([A_sparse, B_sparse])
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))

    print('Your velocities have been computed in a diverging field!')

    def res_div(uv):
        """ 
            Function that computes the residual between given F field and computed F field
        """
        
        # Call function that computes the divergence field
        #div = compute_div(uv).flatten
        
        N = int(np.sqrt((len(uv))/2))
        u = uv[:N*N].reshape((N, N))
        v = uv[N*N:].reshape((N, N))
    
        # Compute the gradient components
        # verif the resolution (spacing between values)
        dudx = np.gradient(u, axis=1)
        dudy = np.gradient(u, axis=0)
        dvdx = np.gradient(v, axis=1)
        dvdy = np.gradient(v, axis=0)

        # Compute the divergence
        div = (dudx + dvdy)
        
        # This is kinda weird, NEED TO CHECK THE MATH to make sure it's legit (I think it might miss the v-component of the F)
        residual = div - F[:N, :] 
        residual_flat = residual.flatten()
        residual_padded = np.hstack([residual_flat, np.zeros_like(residual_flat)])
        
        return residual_padded
        
    # Optimize the field, and make sure the recomputed and not recomputed ones match    
    UV_opt = optimize.newton_krylov(res_div, UV, method='lgmres', inner_maxiter=20, iter=5000, f_tol=6e-5)
        
    U_grid_o = UV_opt[:N2].reshape((N,N))
    V_grid_o = UV_opt[N2:].reshape((N,N))
    
    # Compute the final divergence
    dudx = np.gradient(U_grid_o, axis=1)
    dudy = np.gradient(U_grid_o, axis=0)
    dvdx = np.gradient(V_grid_o, axis=1)
    dvdy = np.gradient(V_grid_o, axis=0)
    # Compute the divergence
    div = (dudx + dvdy)
    
    if vel_fig == True:
        # Show the U, V fiel in quivers
        plt.figure()
        plt.quiver(U_grid, V_grid, cmap=cm.viridis)
        plt.title("Speeds fields")
        plt.show() 
        
        # Show the U, V fiel in quivers
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


# Create the shearing fields
def synthetic_shear(S, dx, dy, vel_fig=False, div_fig=False):
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
    
    # Check if all elements of F add up to 0 (if not; it won't solve properly); sould fix this someday
    if np.sum(S) != 0:
        raise SystemExit(
                "Need to have a F field that adds up to 0."
            )
        
    # Flatten the F matrix (2*N2, 1)
    S_flat = S.flatten()
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dx
    B_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dy 
    C_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dy
    D_sparse = diags([1, -1], [0, -1], shape=(N2, N2)) / dx
    
    # Get the complete sparse finite differences matrix (2*2N, 2*2N)
    ABCD_sparse = bmat([[A_sparse, -B_sparse], 
                        [C_sparse, D_sparse]])
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(ABCD_sparse, S_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    print('Your velocities have been computed in a diverging field!')
    
    def res_shear(uv):
        """ 
            Function that computes the residual between given F field and computed F field
        """
        
        # Call function that computes the shearing field
        N = int(np.sqrt((len(uv))/2))
        u = uv[:N*N].reshape((N, N))
        v = uv[N*N:].reshape((N, N))
    
        # Compute the gradient components
        # verif the resolution (spacing between values)
        dudx = np.gradient(u, axis=1)
        dudy = np.gradient(u, axis=0)
        dvdx = np.gradient(v, axis=1)
        dvdy = np.gradient(v, axis=0)

        # Compute the shear
        shear = np.sqrt((dudx - dvdy)**2 + (dudy + dudx)**2)
        print(np.shape(shear))
        print(np.shape(S))
        
        # This is kinda weird, NEED TO CHECK THE MATH to make sure it's legit (I think it might miss the v-component of the F)
        residual = shear - S[:N, :] 
        residual_flat = residual.flatten()
        residual_padded = np.hstack([residual_flat, np.zeros_like(residual_flat)])
        
        return residual_padded
        
    # Optimize the field, and make sure the recomputed and not recomputed ones match    
    UV_opt = optimize.newton_krylov(res_shear, UV, method='lgmres', inner_maxiter=20, iter=5000, f_tol=6e-5)
        
    U_grid_o = UV_opt[:N2].reshape((N,N))
    V_grid_o = UV_opt[N2:].reshape((N,N))
    
    # Compute the final divergence
    dudx = np.gradient(U_grid_o, axis=1)
    dudy = np.gradient(U_grid_o, axis=0)
    dvdx = np.gradient(V_grid_o, axis=1)
    dvdy = np.gradient(V_grid_o, axis=0)
    # Compute the shear
    shear = np.sqrt((dudx - dvdy)**2 + (dudy + dudx)**2)
    
    if vel_fig == True:
        # Show the U, V fiel in quivers
        plt.figure()
        plt.quiver( U_grid, V_grid, cmap=cm.viridis)
        plt.title("Initial speeds fields")
        plt.show() 
        
        plt.figure()
        plt.quiver( U_grid_o, V_grid_o, cmap=cm.viridis)
        plt.title("Optimized speeds fields")
        plt.show() 
        
    if div_fig == True:
        # Show the divergence field you prescibed
        plt.figure()
        plt.title("Shear field")
        plt.pcolormesh(S)
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.title("Recomputed shear field")
        plt.pcolormesh(shear)
        plt.colorbar()
        plt.show()
        
    U_grid_o = np.hstack([np.zeros((N, 1)), U_grid_o]) # so that the shape is (ny, nx+1)
    V_grid_o = np.vstack([np.zeros((1, N)), V_grid_o]) # so that the shape is (ny+1, nx)
        
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
    


# Construct the full F matrix
pattern = [-1,0 ,-1 ,1, 1, 1, 1,-1 ,0,-1]
F_grid = np.tile(pattern, (len(pattern), 1))
N = len(F_grid[0])
N2 = N**2
z_grid = np.zeros((N,N))
F = np.vstack([F_grid, z_grid])

#u, v = synthetic_divergence(F, 1, 1, vel_fig=True, div_fig=True)
u, v = synthetic_shear(F, 1, 1, vel_fig=True, div_fig=True)

#save_fields(u, v, "60", start_date = datetime(2002, 1, 1), end_date = datetime(2002, 1, 31, 18))