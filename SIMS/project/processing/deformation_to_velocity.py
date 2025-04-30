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
from utils.helpers import create_sparse_matrix_dx, create_sparse_matrix_dy, create_sparse_double_matrix_dydx, create_sparse_double_matrix_dxdy, create_sparse_matrix_dudy,create_sparse_matrix_dvdx

def compute_velocity_fields(F, exp_type, name, color='k'):
    
    if exp_type == "div":
        u, v, F_recomp = synthetic_divergence(F, name, color)
        
    elif exp_type == "shear":
        u, v = synthetic_shear(F, name)
        
    elif exp_type == "both":
        u, v = synthetic_deformations(F, name)
        
    else:
        raise ValueError(f"Type of deformation '{exp_type}' not found.")

    return (u, v, F_recomp)
        
        
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
        
    # RANDOM SHUFFLING 
    #np.random.shuffle(F)
    
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
    
    # Centered finite differences
    u = U_grid
    v = V_grid
    
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
    
    # Compute the shear
    # du/dy
    u_j = u[:-1, :]
    u_jp1 = u[1:, :]
    dudy = (u_jp1 - u_j) / dy
    # Pad with zeros to match dimensions
    zeros_j = np.zeros((1, u.shape[1]))
    dudy = np.vstack((dudy, zeros_j))
    # dv/dx
    v_i = v[:, :-1]
    v_ip1 = v[:, 1:]
    dvdx = (v_ip1 - v_i) / dx
    # Pad with zeros to match dimensions
    zeros_i = np.zeros((v.shape[0], 1))
    dvdx = np.hstack((dvdx, zeros_i))
    
    #shear = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)
    shear = np.zeros((N,N))
    
    #div = np.sqrt(div**2 + shear**2)
    #div = dudy + dvdx
    
    #div = F[N:,:]
    #div = F[:N,:]
    
    # This is if I want to save the data to export and put in Antoine's code ...    
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
    
    ## This is to plot velocities and deformations on two different pannels
    #plt.rcParams.update({'font.size': 16})
    #with plt.style.context(['science', 'no-latex']):
    #    
    #    # Create a single figure with two panels
    #    fig, axs = plt.subplots(
    #        1, 2, 
    #        figsize=(14, 6), 
    #        gridspec_kw={'width_ratios': [1, 1]}  # Equal width for both panels
    #    )
    #
    #    # Panel 1: Recomputed shear field
    #    ax1 = axs[1]
    #    im1 = ax1.pcolormesh(div, cmap=cm.BrBG, vmin=-1, vmax=1)
    #    fig.colorbar(im1, ax=ax1, orientation='vertical', label="Deformation")
    #    ax1.set_title("(b) Divergence/Convergence")
    #    ax1.set_xticks([])
    #    ax1.set_yticks([])
    #    
    #    # Panel 2: Speed field with vectors
    #    ax2 = axs[0]
    #    speed = U_grid + V_grid  # Replace with actual calculation
    #    x = np.arange(U_grid.shape[1])
    #    y = np.arange(V_grid.shape[0])
    #    X , Y = np.meshgrid(x, y)
    #
    #
    #    im2 = ax2.pcolormesh(X, Y, speed, cmap=cm.RdBu, shading='auto', alpha=0.6, vmin=-2, vmax=2)
    #    quiver = ax2.quiver(X, Y, U_grid, V_grid, color='k', width=0.004)  # Adjust scale as needed
    #    fig.colorbar(im2, ax=ax2, orientation='vertical', label="Speed")
    #    ax2.set_title("(a) Divergence/Convergence Velocities")
    #    ax2.set_xticks([])
    #    ax2.set_yticks([])
    #    # Adjust layout for clarity
    #    plt.tight_layout()
    #    plt.show()
     
    # Main fig that gets saved!  
    #plt.rcParams.update({'font.size': 16})
    #with plt.style.context(['science', 'no-latex']):
    #    
    #    fig, ax = plt.subplots(
    #        figsize=(8, 6)
    #    )
    #
    #    speed = U_grid + V_grid
    #    x = np.arange(U_grid.shape[1])
    #    y = np.arange(V_grid.shape[0])
    #    X , Y = np.meshgrid(x, y)
    #
    #    im1 = ax.pcolormesh(div, cmap=cm.RdBu, vmin=-1, vmax=1, alpha=0.7)
    #    quiver = ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.004, scale = 30)  # Adjust scale as needed
    #    fig.colorbar(im1, ax=ax, orientation='vertical', label="Deformation")
    #    ax.set_title("(a) Divergence/Convergence")
    #    ax.set_xticks([])
    #    ax.set_yticks([])
    #    
    #    for spine in ax.spines.values():
    #        spine.set_edgecolor('black')
    #        spine.set_linewidth(2)  # Adjust the border thickness if needed
    #    
    #    plt.tight_layout()
    #    combined_filename = os.path.join("SIMS/project/figures", f"{name}_div_velocity.png")
    #    plt.savefig(combined_filename)
    #    plt.close(fig)
        
    #plt.rcParams.update({'font.size': 16})
    #plt.rcParams['hatch.linewidth'] = 1
    #with plt.style.context(['science', 'no-latex']):
    #    
    #    # Create a single figure with two panels
    #    fig, ax = plt.subplots(
    #        figsize=(9, 6)  # Equal width for both panels
    #    )
    #
    #    speed = U_grid + V_grid  # Replace with actual calculation
    #    x = np.arange(U_grid.shape[1])
    #    y = np.arange(V_grid.shape[0])
    #    X , Y = np.meshgrid(x, y)
    #    X_shifted = X 
    #    
    #    #div_shifted = np.roll(div, shift=-1, axis=1)
    #    #div_shifted[:, -1] = div[:, -1]
    #    #div = div_shifted
    #    
    #
    #
    #    # Add conditions for divergence and shear
    #    div_pos = (div > 0).astype(int)
    #    div_neg = (div < 0).astype(int)
    #    shear_pos = (shear > 0).astype(int)
    #    shear_neg = (shear < 0).astype(int)
    #
    #    # Apply hatching patterns for each condition
    #    for condition, hatch, line_color, label in zip(
    #        [div_pos, div_neg, shear_pos, shear_neg],
    #        ['', '', '', ''],  # Hatching styles
    #        ['darkseagreen', 'steelblue', 'darksalmon', 'darkorange'],  # Colors for hatching lines
    #        ['Convergence', 'Divergence', 'Shear +', 'Shear -']  # Labels
    #    ):
    #        # Simulate hatches using colored contour lines
    #        #ax.contour(
    #        #    X_shifted, Y, condition,
    #        #    levels=[0.5],  # Single-level contour
    #        #    colors='none',  # Set the desired hatch color
    #        #    linestyles='solid',
    #        #    linewidths=1.5,  # Adjust thickness for better visibility
    #        #    alpha=0.8
    #        #)
    #        
    #        
    #        ## Add background fill color for the selected condition
    #        #ax.contourf(
    #        #    X, Y, condition,
    #        #    levels=[0.5, 1.5],
    #        #    colors=[line_color],  # Background fill color (blue/red)
    #        #    alpha=0.6  # Adjust transparency
    #        #)
    #        # Add dense hatching patterns using contourf
    #        ax.contourf(
    #            X_shifted, Y, condition,
    #            levels=[0.5, 1.5],
    #            colors=[line_color], 
    #            hatches=[hatch],
    #            alpha=0.5
    #        )
    #        
    #    # Add quivers for vector field
    #    ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.005, scale=20, alpha=1)
    
    #    # Add legend for hatching patterns
    #    legend_elements = [
    #        Patch(edgecolor='black', facecolor='darkseagreen', hatch='',alpha=0.5, label='Divergence'),
    #        Patch(edgecolor='black', facecolor='steelblue', hatch='',alpha=0.5, label='Convergence'),
    #        Patch(edgecolor='k', facecolor='darksalmon', hatch='',alpha=0.5, label='Shear +'),
    #        Patch(edgecolor='k', facecolor='darkorange', hatch='',alpha=0.5, label='Shear -'),
    #        Patch(edgecolor='white', facecolor='white', label=''),
    #        Line2D([0], [0], color='k', lw=2, label='Velocities', marker='>', markersize=8)
    #        
    #    ]
    #    
    #    quiver_key = ax.quiverkey(
    #        quiver, X=0.8, Y=1.05, U=1, label='Velocities', labelpos='E', coordinates='axes'
    #    )
    #    legend = ax.legend(handles=legend_elements, loc="upper left",bbox_to_anchor=(1, 1), fontsize=14, title="Deformations", frameon=True, facecolor='white', edgecolor='k', framealpha=0.9)
    #
    #    ax.set_title(f"{name}", fontweight ="bold", color=color)
    #    ax.set_xticks([])
    #    ax.set_yticks([])

    #    for spine in ax.spines.values():
    #        spine.set_edgecolor('black')
    #        spine.set_linewidth(2)  # Adjust the border thickness if needed
    #                
    #    # Adjust layout for clarity
    #    plt.tight_layout()
    #    combined_filename = os.path.join("SIMS/project/figures", f"Separated_{name}_div_velocity.png")
    #    plt.savefig(combined_filename)
    #    plt.close(fig)
    #
    #    print(f"Divergence figure saved: {name}")
        
    '''    
    with plt.style.context(['science', 'no-latex']):
        # Create a single figure with one panel
        fig, ax = plt.subplots(figsize=(9, 10))

        # Create a mesh grid for the vector field
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X, Y = np.meshgrid(x, y)

        # Calculate speed for better visualization (optional, if needed)
        speed = np.sqrt(U_grid**2 + V_grid**2)

        # Use pcolormesh for background color representation
        ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', alpha=0.6, vmin = -1, vmax = 1)
    
        # Plot quivers for velocity field
        ax.quiver(X, Y, U_grid, V_grid, color='k', linewidth=5,width=0.1, scale=10, scale_units='xy',alpha=0.8)

        # Add a color bar for the pcolormesh
        cbar = plt.colorbar(ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', vmin = -1, vmax = 1), ax=ax, orientation='horizontal')
        cbar.set_label('Divergence', fontsize=18)

        # Set plot title and remove ticks
        #ax.set_title(f"{name}", fontweight="bold", color="black")
        ax.set_title(f"{name}", fontweight ="extra bold", color=color,  family='sans-serif')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a legend for the quivers
        quiver_key = ax.quiverkey(
            ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.003, scale=40, alpha=1),
            X=0.85, Y=1.025, U=1, label='Velocities',labelpos='E', coordinates='axes' , fontproperties={'size': 18}
        )


        # Adjust spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Save the figure
        plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures", f"2Separated_{name}_div_velocity.png")
        plt.savefig(combined_filename)
        plt.close(fig)

    print(f"Divergence figure saved: {name}")
    '''    
    """
    with plt.style.context(['science', 'no-latex']):
        # Create a single figure with one panel
        fig, ax = plt.subplots(figsize=(9, 10))

        # Create a mesh grid for the vector field
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X, Y = np.meshgrid(x, y)

        # Calculate speed for better visualization (optional, if needed)
        speed = np.sqrt(U_grid**2 + V_grid**2)

        # Use pcolormesh for background color representation
        ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', alpha=0.6, vmin = -1, vmax = 1)
    
        # Add a color bar for the pcolormesh
        cbar = plt.colorbar(ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', vmin = -1, vmax = 1), ax=ax, orientation='horizontal')
        cbar.set_label('Divergence', fontsize=18)

        # Set plot title and remove ticks
        #ax.set_title(f"{name}", fontweight="bold", color="black")
        ax.set_title(f"{name}", fontweight ="extra bold", color=color,  family='sans-serif')
        ax.set_xticks([])
        ax.set_yticks([])



        # Adjust spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Save the figure
        plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures", f"no_quivs_{name}_div_velocity.png")
        plt.savefig(combined_filename)
        plt.close(fig)

    print(f"Divergence figure saved no quivers: {name}")
    """
    return U_grid, V_grid, div


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
    #A_sparse = create_sparse_matrix_dudy(N)
    B_sparse = create_sparse_matrix_dx(N)
    #B_sparse = create_sparse_matrix_dvdx(N)
    
    zero_matrix = csr_matrix((N2, N2)) 
    AB_sparse = bmat([[A_sparse, zero_matrix], 
                       [zero_matrix, B_sparse]])
    
    B_sparse_csr = B_sparse.tocsr()
    dense_section = B_sparse_csr[:10, :10].todense()  # Inspect a small section
    
    # Compute the u and v field by solving the linear system (2*2N, 1)
    UV = spsolve(AB_sparse, S_flat)
    U_grid = UV[:N2].reshape((N,N))
    V_grid = UV[N2:].reshape((N,N))
    
    # Test to reshape the u and v values on the grid!!
    Ny, Nx = U_grid.shape
    # Redistribute U: Average in x-direction for interior points, preserve edges
    U_corrected = 0.5 * (U_grid[:, :-1] + U_grid[:, 1:])  # Average adjacent x values
    U_corrected = np.hstack((U_grid[:, 0:1], U_corrected, U_grid[:, -1:]))  # Preserve edges
    # Redistribute V: Average in y-direction for interior points, preserve edges
    V_corrected = 0.5 * (V_grid[:-1, :] + V_grid[1:, :])  # Average adjacent y values
    V_corrected = np.vstack((V_grid[0:1, :], V_corrected, V_grid[-1:, :]))  # Preserve edges
    # To ensure consistent dimensions with the C-grid:
    U_corrected = U_corrected[:, :-1]  # Drop the extra row at the bottom
    V_corrected = V_corrected[:-1, :]  # Drop the extra column on the right
    #U_grid = U_corrected
    #V_grid = V_corrected
    
    u=U_grid
    v=V_grid
    """
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
    """
    
    
    """
    # WORKED ON DEC 9th !!
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
    """
    
    # Get the shape of the input array
    Ny, Nx = u.shape

    # Initialize dudy with zeros
    dudy = np.zeros_like(u)

    # Loop over the interior points (avoid boundaries)
    for i in range(Ny - 1):  # Loop over rows (i, i+1)
        for j in range(1, Nx - 1):  # Loop over columns (j-1, j+1)
            # Apply the scheme
            dudy[i, j] = (
                u[i+1, j] + u[i+1, j+1]  # ui,j+1 and ui+1,j+1
                - u[i-1, j] - u[i-1, j+1]  # ui,j-1 and ui+1,j-1
            ) / (4 * dy)
            print(dudy[i, j])
            
    dvdx = np.zeros_like(dudy)
    shear = dudy + dvdx
    print(u)
    print(shear)
    #shear = (S[N:,:] + S[:N, :])/2

    div = np.zeros((N,N))

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
        quiver = ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.004, scale = 30)  # Adjust scale as needed
        fig.colorbar(im1, ax=ax, orientation='vertical', label="Deformation")
        ax.set_title("(b) Shear")
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  # Adjust the border thickness if needed        
        
        plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures", f"{name}_shear_velocity.png")
        plt.savefig(combined_filename)
        plt.close(fig)
        
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
        ax.quiver(X, Y, U_grid, V_grid, color='k', width=0.005, scale=20, alpha=1)
    
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
    
        ax.set_title("(b) Shear")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  # Adjust the border thickness if needed
                    
        # Adjust layout for clarity
        plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures", f"Separated_{name}_shear_velocity.png")
        plt.savefig(combined_filename)
        plt.close(fig)
    
    print(f"Shear figure saved: {name}")
    #U_grid_o = np.hstack([np.zeros((N, 1)), U_grid]) # so that the shape is (ny, nx+1)
    #V_grid_o = np.vstack([np.zeros((1, N)), V_grid]) # so that the shape is (ny+1, nx)
    
    U_grid_o = U_grid
    V_grid_o = V_grid
        
    return U_grid_o, V_grid_o

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