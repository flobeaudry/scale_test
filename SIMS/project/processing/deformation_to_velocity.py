import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import scienceplots
from scipy.optimize import newton_krylov
from scipy.sparse import diags, block_diag, bmat,csr_matrix, lil_matrix, csc_matrix
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from utils.helpers import create_sparse_matrix_dx, create_sparse_matrix_dy, create_sparse_double_matrix_dydx, create_sparse_double_matrix_dxdy, create_sparse_matrix_dudy,create_sparse_matrix_dvdx, create_sparse_matrix_dy_shear, create_sparse_matrix_dx_shear
from utils.velocity_gradients_calc import calc_du_dx, calc_du_dy, calc_dv_dx, calc_dv_dy


def compute_velocity_fields(F, exp_type, name, color='k'):
    
    if exp_type == "div":
        u, v, F_recomp, u_noise, v_noise = synthetic_divergence(F, name, color)
        
    elif exp_type == "shear":
        u, v, F_recomp, u_noise, v_noise = synthetic_shear(F, name, color)
        
    elif exp_type == "both":
        u, v = synthetic_deformations(F, name)
        
    else:
        raise ValueError(f"Type of deformation '{exp_type}' not found.")

    return (u, v, F_recomp, u_noise, v_noise)
        
        
def synthetic_divergence(F, name, color, dx=1, dy=1, vel_fig=True, div_fig=True):
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
    zero_matrix = lil_matrix((N2, N2))
    A_sparse = create_sparse_matrix_dx(N).tolil()  # Efficient construction
    B_sparse = create_sparse_matrix_dy(N).tolil()
    AB_sparse = bmat([[A_sparse, zero_matrix], [zero_matrix, B_sparse]], format="lil")
    AB_sparse = AB_sparse.tocsr()

    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    print("Old N", N)
    
    # To only analyse the top right quadrant
    mid_x = U_grid.shape[1] // 2
    mid_y = V_grid.shape[0] // 2
    fraction = 1 # Adjust this value as needed
    reduced_x = int(mid_x + (U_grid.shape[1] - mid_x) * fraction)
    reduced_y = int(mid_y + (V_grid.shape[0] - mid_y) * fraction)
    U_grid = U_grid[mid_y:reduced_y, mid_x:reduced_x]
    V_grid = V_grid[mid_y:reduced_y, mid_x:reduced_x]
    N = len(U_grid[0,:])
    N2 = N**2
        
    
    # Centered finite differences
    u = U_grid
    v = V_grid

    # I think could remove; useless
    noise_u = np.zeros((N,N))
    noise_v = np.zeros((N,N))
        
    # Compute the divergence
    dudx = calc_du_dx(u, dx)
    # Pad with zeros to add with dvdy   
    zeros_i = np.zeros((u.shape[0], 1))
    dudx = np.hstack((zeros_i, dudx)) 
    

    dvdy = calc_dv_dy (v, dy)
    # Pad with zeros to match
    zeros_j = np.zeros(len(v[0,:]))
    dvdy = np.vstack((zeros_j, dvdy))
    
    div = dudx + dvdy
    shear = np.zeros((N,N))
    
    return U_grid, V_grid, div, noise_u, noise_v


def synthetic_shear(S, name, color, dx=1, dy=1, vel_fig=True, shear_fig=True):
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
    zero_matrix = lil_matrix((N2, N2))
    B_sparse = create_sparse_matrix_dy_shear(N).tolil()
    A_sparse = create_sparse_matrix_dx(N).tolil()  # for div, wrong !!
    AB_sparse = bmat([[A_sparse, zero_matrix], [zero_matrix, B_sparse]], format="lil")
    AB_sparse = AB_sparse.tocsr()

    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, S_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    u=U_grid
    v=V_grid
    
    # Don't deal with noise yet lol -- don't think we'll ever here !
    noise_u = np.zeros((N,N))
    noise_v = np.zeros((N,N))
    
    div = np.zeros((N,N))

    dvdx = calc_dv_dx (v, dx)
    dudy = calc_du_dy (u, dy)
    
    dudx = calc_du_dx (u, dx)
    dvdy = calc_dv_dy (v, dy)
    #shear = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)
    shear = dudy + dvdx
   
    return U_grid, V_grid, shear, noise_u, noise_v

def synthetic_deformations_together_ok(F, name, dx=1, dy=1, vel_fig=False, shear_fig=False):
    # TOGETHER, kind of works?
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
    A_sparse = create_sparse_matrix_dx(N)
    B_sparse = create_sparse_matrix_dy(N)
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[A_sparse, A_sparse/2], 
                       [B_sparse/2, B_sparse]])
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    
    u=U_grid
    v=V_grid

    # Compute the divergence
    u_i = u[:, :-1]
    u_ip1 = u[:, 1:]
    dudx = (u_ip1 - u_i)/dx
    # Pad with zeros to add with dvdy   
    zeros_i = np.zeros((u.shape[0], 1))
    dudx = np.hstack((zeros_i, dudx)) 
    
    v_j = v[:-1, :]
    v_jp1 = v[1:, :]
    dvdy = (v_jp1 - v_j)/dy
    # Pad with zeros to match
    zeros_j = np.zeros(len(v[0,:]))
    dvdy = np.vstack((zeros_j, dvdy))
    
    div = dudx + dvdy
    
    v_i = v[:, :-1]
    v_ip1 = v[:, 1:]
    dvdx = (v_ip1 - v_i)/dx
    # Pad with zeros to add with dvdy   
    zeros_i = np.zeros((v.shape[0], 1))
    dvdx = np.hstack((zeros_i, dvdx)) 
    
    u_j = u[:-1, :]
    u_jp1 = u[1:, :]
    dudy = (u_jp1 - u_j)/dy
    # Pad with zeros to match
    zeros_j = np.zeros(len(u[0,:]))
    dudy = np.vstack((zeros_j, dudy))

    shear = dudy + dvdx

    #defo = np.sqrt((dudx + dvdy)**2 + (dudy + dvdx)**2)
    defo = div + shear/2
    
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
        quiver = ax2.quiver(X, Y, U_grid, V_grid, color='k', width=0.004, scale = 30)  # Adjust scale as needed
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
        combined_filename = os.path.join("SIMS/project/figures", f"{name}_both_velocity.png")
        plt.savefig(combined_filename)
        plt.close()
        
        print(f"Deformations figure saved: {name}")
        
    
    
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
     
    U_grid_o = U_grid   
    V_grid_o = V_grid
    
    return U_grid_o, V_grid_o

def synthetic_deformations(F, name, dx=1, dy=1, vel_fig=False, shear_fig=False):
# SEPARATED
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
    A_sparse = create_sparse_matrix_dx(N)
    B_sparse = create_sparse_matrix_dy(N)
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[A_sparse, zero_matrix, zero_matrix, zero_matrix], 
                       [zero_matrix, B_sparse/2, zero_matrix, zero_matrix],
                       [zero_matrix, zero_matrix, B_sparse, zero_matrix],
                       [zero_matrix, zero_matrix, zero_matrix, A_sparse/2]])
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, F_flat)
    #U_grid = (UV[:N2]).reshape((N,N))
    #V_grid = np.zeros((N, N))
    U_grid = (UV[:N2]+UV[N2:2*N2]).reshape((N,N))
    V_grid = (UV[2*N2:3*N2]+UV[3*N2:]).reshape((N,N))
    
    u=U_grid
    v=V_grid

    # Compute the divergence
    u_i = u[:, :-1]
    u_ip1 = u[:, 1:]
    dudx = (u_ip1 - u_i)/dx
    # Pad with zeros to add with dvdy   
    zeros_i = np.zeros((u.shape[0], 1))
    dudx = np.hstack((zeros_i, dudx)) 
    
    v_j = v[:-1, :]
    v_jp1 = v[1:, :]
    dvdy = (v_jp1 - v_j)/dy
    # Pad with zeros to match
    zeros_j = np.zeros(len(v[0,:]))
    dvdy = np.vstack((zeros_j, dvdy))
    
    div = dudx + dvdy
    
    v_i = v[:, :-1]
    v_ip1 = v[:, 1:]
    dvdx = (v_ip1 - v_i)/dx
    # Pad with zeros to add with dvdy   
    zeros_i = np.zeros((v.shape[0], 1))
    dvdx = np.hstack((zeros_i, dvdx)) 
    
    u_j = u[:-1, :]
    u_jp1 = u[1:, :]
    dudy = (u_jp1 - u_j)/dy
    # Pad with zeros to match
    zeros_j = np.zeros(len(u[0,:]))
    dudy = np.vstack((zeros_j, dudy))

    shear = dudy + dvdx

    #defo = np.sqrt((dudx + dvdy)**2 + (dudy + dvdx)**2)
    defo = div + shear/2
    
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
        quiver = ax2.quiver(X, Y, U_grid, V_grid, color='k', width=0.004, scale = 30)  # Adjust scale as needed
        fig.colorbar(im2, ax=ax2, orientation='vertical', label="Speed")
        ax2.set_title("(e) Divergence/Convergence + Shear Velocities")
        ax2.set_xticks([])
        ax2.set_yticks([])
        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()
        
    shear = F[N:2*N] + F[3*N:]
    div = F[:N] + F[2*N : 3*N]
    
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['hatch.linewidth'] = 1
    with plt.style.context(['science', 'no-latex']):
        
        # Create a single figure with two panels
        fig, ax = plt.subplots(
            figsize=(9, 6)  # Equal width for both panels
        )

        speed = U_grid + V_grid  # Replace with actual calculation
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X , Y = np.meshgrid(x, y)

        # Add conditions for divergence and shear
        div_pos = (div > 0).astype(int)
        div_neg = (div < 0).astype(int)
        shear_pos = (shear > 0).astype(int)
        shear_neg = (shear < 0).astype(int)

        # Apply hatching patterns for each condition
        for condition, hatch, line_color, label in zip(
            [div_pos, div_neg, shear_pos, shear_neg],
            ['', '', '', ''],  # Hatching styles
            ['darkseagreen', 'steelblue', 'darksalmon', 'darkorange'],  # Colors for hatching lines
            ['Convergence', 'Divergence', 'Shear +', 'Shear -']  # Labels
        ):
            # Simulate hatches using colored contour lines
            ax.contour(
                X, Y, condition,
                levels=[0.5],  # Single-level contour
                colors='none',  # Set the desired hatch color
                linestyles='solid',
                linewidths=1.5,  # Adjust thickness for better visibility
                alpha=0.8
            )
            ## Add background fill color for the selected condition
            #ax.contourf(
            #    X, Y, condition,
            #    levels=[0.5, 1.5],
            #    colors=[line_color],  # Background fill color (blue/red)
            #    alpha=0.6  # Adjust transparency
            #)
            # Add dense hatching patterns using contourf
            ax.contourf(
                X, Y, condition,
                levels=[0.5, 1.5],
                colors=[line_color], 
                hatches=[hatch],
                alpha=0.5
            )
            
        # Add quivers for vector field
        ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.004, scale=30, alpha=1)
    
        # Add legend for hatching patterns
        legend_elements = [
            Patch(edgecolor='black', facecolor='darkseagreen', hatch='',alpha=0.5, label='Divergence'),
            Patch(edgecolor='black', facecolor='steelblue', hatch='',alpha=0.5, label='Convergence'),
            Patch(edgecolor='k', facecolor='darksalmon', hatch='',alpha=0.5, label='Shear +'),
            Patch(edgecolor='k', facecolor='darkorange', hatch='',alpha=0.5, label='Shear -'),
            Patch(edgecolor='white', facecolor='white', label=''),
            Line2D([0], [0], color='k', lw=2, label='Velocities', marker='>', markersize=8)
            
        ]
        
        quiver_key = ax.quiverkey(
            quiver, X=0.8, Y=1.05, U=1, label='Velocities', labelpos='E', coordinates='axes'
        )
        legend = ax.legend(handles=legend_elements, loc="upper left",bbox_to_anchor=(1, 1), fontsize=14, title="Deformations", frameon=True, facecolor='white', edgecolor='k', framealpha=0.9)
    
        ax.set_title("(c) Divergence/Convergence + Shear")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  # Adjust the border thickness if needed
                    
        # Adjust layout for clarity
        plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures", f"Separated_{name}_both_velocity.png")
        plt.savefig(combined_filename)
        plt.close()
        
        print(f"Deformations figure saved: {name}")

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
        quiver = ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.004, scale=30 )  # Adjust scale as needed
        fig.colorbar(im1, ax=ax, orientation='vertical', label="Deformation")
        ax.set_title("(c) Divergence/Convergence + Shear")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  # Adjust the border thickness if needed
                    
        # Adjust layout for clarity
        plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures", f"{name}_both_velocity.png")
        plt.savefig(combined_filename)
        plt.close()
        
        print(f"Deformations figure saved: {name}")
    
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
     
    U_grid_o = U_grid   
    V_grid_o = V_grid
    
    return U_grid_o, V_grid_o

def synthetic_deformations_okold_alltogether(F, name, dx=1, dy=1, vel_fig=False, shear_fig=False):
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
    """
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
    """
    # Compute the divergence
    u_i = u[:, :-1]
    u_ip1 = u[:, 1:]
    dudx = (u_ip1 - u_i)/dx
    # Pad with zeros to add with dvdy   
    zeros_i = np.zeros((u.shape[0], 1))
    dudx = np.hstack((zeros_i, dudx)) 
    
    v_j = v[:-1, :]
    v_jp1 = v[1:, :]
    dvdy = (v_jp1 - v_j)/dy
    # Pad with zeros to match
    zeros_j = np.zeros(len(v[0,:]))
    dvdy = np.vstack((zeros_j, dvdy))
    
    div = dudx + dvdy
    
    v_i = v[:, :-1]
    v_ip1 = v[:, 1:]
    dvdx = (v_ip1 - v_i)/dx
    # Pad with zeros to add with dvdy   
    zeros_i = np.zeros((v.shape[0], 1))
    dvdx = np.hstack((zeros_i, dvdx)) 
    
    u_j = u[:-1, :]
    u_jp1 = u[1:, :]
    dudy = (u_jp1 - u_j)/dy
    # Pad with zeros to match
    zeros_j = np.zeros(len(u[0,:]))
    dudy = np.vstack((zeros_j, dudy))

    shear = dudy + dvdx

    #defo = np.sqrt((dudx + dvdy)**2 + (dudy + dvdx)**2)
    defo = div + shear/2
    
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
        quiver = ax2.quiver(X, Y, U_grid, V_grid, color='k', width=0.004, scale = 30)  # Adjust scale as needed
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
        combined_filename = os.path.join("SIMS/project/figures", f"{name}_both_velocity.png")
        plt.savefig(combined_filename)
        plt.close()
        
        print(f"Deformations figure saved: {name}")
    
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
     
    U_grid_o = U_grid   
    V_grid_o = V_grid
    
    return U_grid_o, V_grid_o



def synthetic_deformations_bad_complicated(F, name, dx=1, dy=1, vel_fig=False, shear_fig=False, tol=1e-6):
    """
    Function that computes u, v fields based on the deformations (div/conv + shear) field given.
    Args:
        F (np.ndarray): array of size (ny, nx, 2) representing divergence (-) or convergence (+).
        dx, dy (float): resolution in x and y.
        vel_fig (bool): True to display velocities (quivers) figure.
        shear_fig (bool): True to display shear figure.
        tol (float): tolerance for the nonlinear solver.
    Returns:
        u, v (np.ndarray): the velocity fields u (ny, nx+1) and v (ny+1, nx).
    """

    # Get matrix shape
    N = len(F[0, :])
    N2 = N ** 2

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
    U_grid = UV[:N2].reshape((N, N))
    V_grid = UV[N2:].reshape((N, N))
    
    def recompute_defo(UV):
        # Unpack UV
        U_grid = UV[:N2].reshape((N, N))
        V_grid = UV[N2:].reshape((N, N))

        # Compute centered finite differences
        zeros_j = np.zeros(len(U_grid[0, :]))  # boundary conditions
        u_jp1 = np.append(zeros_j, U_grid[:-1, :]).reshape(U_grid.shape)
        u_jm1 = np.append(U_grid[1:, :], zeros_j).reshape(U_grid.shape)
        zeros_i = np.zeros((V_grid.shape[0], 1))  # boundary conditions
        v_ip1 = np.hstack((zeros_i, V_grid[:, :-1]))
        v_im1 = np.hstack((V_grid[:, 1:], zeros_i))
        dudy = (u_jp1 - u_jm1) / (2 * dy)
        dvdx = (v_ip1 - v_im1) / (2 * dx)
        
        zeros_j = np.zeros(len(V_grid[0, :]))  # boundary conditions
        v_jp1 = np.append(zeros_j, V_grid[:-1, :]).reshape(V_grid.shape)
        v_jm1 = np.append(V_grid[1:, :], zeros_j).reshape(V_grid.shape)
        zeros_i = np.zeros((U_grid.shape[0], 1))  # boundary conditions
        u_ip1 = np.hstack((zeros_i, U_grid[:, :-1]))
        u_im1 = np.hstack((U_grid[:, 1:], zeros_i))
        dudx = (u_ip1 - u_im1) / (2 * dy)
        dvdy = (v_jp1 - v_jm1) / (2 * dx)
        
        # Recomputed deformation field
        #defo = np.sqrt((dudx + dvdy) ** 2 + (dudy + dvdx) ** 2)
        defo = dudx + dvdy + dudy + dvdx
    
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
            combined_filename = os.path.join("SIMS/project/figures", f"{name}_both_velocity.png")
            plt.savefig(combined_filename)
            plt.close()
        
            print(f"Deformations figure saved: {name}")
     
        U_grid_o = U_grid   
        V_grid_o = V_grid
    
        return defo

    def residual(UV):
        """Residual function for Newton-Krylov."""
        defo = recompute_defo(UV)
        F_reshaped = F.reshape((2, 26, 26)) 
        return (defo - F_reshaped).flatten()

    # Use the linear solution as the initial guess
    UV_initial = UV.flatten()

    # Solve the nonlinear problem using Newton-Krylov
    UV_solution = newton_krylov(residual, UV_initial, f_tol=tol)

    # Reshape the solution back into U and V grids
    U_final = UV_solution[:N2].reshape((N, N))
    V_final = UV_solution[N2:].reshape((N, N))

    return U_final, V_final