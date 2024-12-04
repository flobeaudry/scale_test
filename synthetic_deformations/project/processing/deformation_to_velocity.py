import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import scienceplots
from scipy.sparse import diags, block_diag, bmat,csr_matrix
from utils.helpers import create_sparse_matrix_dx, create_sparse_matrix_dy, create_sparse_double_matrix_dydx, create_sparse_double_matrix_dxdy

def compute_velocity_fields(F, exp_type, name):
    
    if exp_type == "div":
        u, v = synthetic_divergence(F, name)
        
    elif exp_type == "shear":
        u, v = synthetic_shear(F, name)
        
    elif exp_type == "both":
        u, v = synthetic_deformations(F, name)
        
    else:
        raise ValueError(f"Type of deformation '{exp_type}' not found.")

    return (u, v)
        
        
def synthetic_divergence(F, name, dx=1, dy=1, vel_fig=True, div_fig=True):
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
        
    # Flatten the F matrix (2*N2, 1)
    F_flat = F.flatten()
    
    # Define the sparse finite differences matrices (2N, 2N)
    A_sparse = create_sparse_matrix_dx(N)
    B_sparse = create_sparse_matrix_dy(N)
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[A_sparse, zero_matrix], 
                       [zero_matrix, B_sparse]])
    
    A_sparse_csr = A_sparse.tocsr()
    dense_section = A_sparse_csr[:10, :10].todense()  # Inspect a small section
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))

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
    
    # This is if I want to save the data to export and put in Antoine's code ...    
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
    
    # This is to plot velocities and deformations on two different pannels
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        
        # Create a single figure with two panels
        fig, axs = plt.subplots(
            1, 2, 
            figsize=(14, 6), 
            gridspec_kw={'width_ratios': [1, 1]}  # Equal width for both panels
        )

        # Panel 1: Recomputed shear field
        ax1 = axs[1]
        im1 = ax1.pcolormesh(div, cmap=cm.BrBG, vmin=-1, vmax=1)
        fig.colorbar(im1, ax=ax1, orientation='vertical', label="Deformation")
        ax1.set_title("(b) Divergence/Convergence")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Panel 2: Speed field with vectors
        ax2 = axs[0]
        speed = U_grid + V_grid  # Replace with actual calculation
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X , Y = np.meshgrid(x, y)


        im2 = ax2.pcolormesh(X, Y, speed, cmap=cm.RdBu, shading='auto', alpha=0.6, vmin=-2, vmax=2)
        quiver = ax2.quiver(X, Y, U_grid, V_grid, color='k', width=0.004)  # Adjust scale as needed
        fig.colorbar(im2, ax=ax2, orientation='vertical', label="Speed")
        ax2.set_title("(a) Divergence/Convergence Velocities")
        ax2.set_xticks([])
        ax2.set_yticks([])
        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()
     
    # Main fig that gets saved!  
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        speed = U_grid + V_grid
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X , Y = np.meshgrid(x, y)

        im1 = ax.pcolormesh(div, cmap=cm.RdBu, vmin=-1, vmax=1, alpha=0.7)
        quiver = ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.003)  # Adjust scale as needed
        fig.colorbar(im1, ax=ax, orientation='vertical', label="Deformation")
        ax.set_title("(a) Divergence/Convergence")
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  # Adjust the border thickness if needed
        
        plt.tight_layout()
        combined_filename = os.path.join("synthetic_deformations/project/figures", f"{name}_div_velocity.png")
        plt.savefig(combined_filename)
        plt.close(fig)

    return U_grid, V_grid


def synthetic_shear(S, name, dx=1, dy=1, vel_fig=True, shear_fig=True):
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
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, S_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    
    u=U_grid
    v=V_grid
    # Centered finite differences
    zeros_j = np.zeros(len(u[0,:])) # boundary conditions
    #zeros_j = np.ones(len(u[0,:])) # boundary conditions
    #zeros_j = np.ones((v.shape[0], 1))
    u_jp1 = np.append(zeros_j, u[:-1,:]).reshape(u.shape)
    u_jm1 = np.append(u[1:,:], zeros_j).reshape(u.shape)
    
    #zeros_i = np.zeros((v.shape[0], 1)) # boundary conditions
    zeros_i = np.zeros((v.shape[0], 1)) # boundary conditions
    v_ip1 = np.hstack((zeros_i,v[:,:-1]))
    v_im1 = np.hstack((v[:,1:],zeros_i))
    
    dudy = (u_jp1 - u_jm1)/(2*dy)
    dvdx = (v_ip1 - v_im1)/(2*dx)

    shear = dudy + dvdx

    # This is to plot velocities and deformations on two different pannels
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        
        # Create a single figure with two panels
        fig, axs = plt.subplots(
            1, 2, 
            figsize=(14, 6), 
            gridspec_kw={'width_ratios': [1, 1]}  # Equal width for both panels
        )

        # Panel 1: Recomputed shear field
        ax1 = axs[1]
        im1 = ax1.pcolormesh(shear, cmap=cm.BrBG, vmin=-1, vmax=1)
        fig.colorbar(im1, ax=ax1, orientation='vertical', label="Deformation")
        ax1.set_title("(d) Shear")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Panel 2: Speed field with vectors
        ax2 = axs[0]
        speed = U_grid + V_grid  # Replace with actual calculation
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X , Y = np.meshgrid(x, y)


        im2 = ax2.pcolormesh(X, Y, speed, cmap=cm.RdBu, shading='auto', alpha=0.6, vmin=-2, vmax=2)
        quiver = ax2.quiver(X, Y, U_grid, V_grid, color='k', width=0.004)  # Adjust scale as needed
        fig.colorbar(im2, ax=ax2, orientation='vertical', label="Speed")
        ax2.set_title("(c) Shear Velocities")
        ax2.set_xticks([])
        ax2.set_yticks([])
        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()
    
    # Main figure to save
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        
        # Create a single figure with two panels
        fig, ax = plt.subplots(
            figsize=(8, 6)  # Equal width for both panels
        )

        speed = U_grid + V_grid
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X , Y = np.meshgrid(x, y)

        im1 = ax.pcolormesh(shear, cmap=cm.RdBu, vmin=-1, vmax=1, alpha=0.7)
        quiver = ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.003)  # Adjust scale as needed
        fig.colorbar(im1, ax=ax, orientation='vertical', label="Deformation")
        ax.set_title("(b) Shear")
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  # Adjust the border thickness if needed        
        
        plt.tight_layout()
        combined_filename = os.path.join("synthetic_deformations/project/figures", f"{name}_shear_velocity.png")
        plt.savefig(combined_filename)
        plt.close(fig)
    
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
    
    U_grid_o = U_grid
    V_grid_o = V_grid
        
    return U_grid_o, V_grid_o

def synthetic_deformations(F, name, dx=1, dy=1, vel_fig=False, shear_fig=False):
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

    # This is to plot velocities and deformations on two different pannels
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        
        # Create a single figure with two panels
        fig, axs = plt.subplots(
            1, 2, 
            figsize=(14, 6), 
            gridspec_kw={'width_ratios': [1, 1]}  # Equal width for both panels
        )

        # Panel 1: Recomputed shear field
        ax1 = axs[1]
        im1 = ax1.pcolormesh(defo, cmap=cm.BrBG, vmin=-1, vmax=1)
        fig.colorbar(im1, ax=ax1, orientation='vertical', label="Deformation")
        ax1.set_title("(f) Divergence/Convergence + Shear")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Panel 2: Speed field with vectors
        ax2 = axs[0]
        speed = U_grid + V_grid  # Replace with actual calculation
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X , Y = np.meshgrid(x, y)


        im2 = ax2.pcolormesh(X, Y, speed, cmap=cm.RdBu, shading='auto', alpha=0.6, vmin=-2, vmax=2)
        quiver = ax2.quiver(X, Y, U_grid, V_grid, color='k', width=0.002)  # Adjust scale as needed
        fig.colorbar(im2, ax=ax2, orientation='vertical', label="Speed")
        ax2.set_title("(e) Divergence/Convergence + Shear Velocities")
        ax2.set_xticks([])
        ax2.set_yticks([])
        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()
        
    
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        
        # Create a single figure with two panels
        fig, ax = plt.subplots(
            figsize=(8, 6)  # Equal width for both panels
        )

        speed = U_grid + V_grid  # Replace with actual calculation
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X , Y = np.meshgrid(x, y)

        im1 = ax.pcolormesh(defo, cmap=cm.RdBu, vmin=-1, vmax=1, alpha=0.7)
        quiver = ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.004)  # Adjust scale as needed
        fig.colorbar(im1, ax=ax, orientation='vertical', label="Deformation")
        ax.set_title("(c) Divergence/Convergence + Shear")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  # Adjust the border thickness if needed
                    
        # Adjust layout for clarity
        plt.tight_layout()
        combined_filename = os.path.join("synthetic_deformations/project/figures", f"{name}_both_velocity.png")
        plt.savefig(combined_filename)
        plt.close()
    
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
     
    U_grid_o = U_grid   
    V_grid_o = V_grid
    
    return U_grid_o, V_grid_o
