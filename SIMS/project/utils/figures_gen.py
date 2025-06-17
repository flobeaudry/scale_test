import numpy as np
import matplotlib.pyplot as plt
import os
from utils.velocity_gradients_calc import calc_div, calc_shear, calc_shear_components, calc_tot_defo, calc_du_dx, calc_du_dy, calc_dv_dx, calc_dv_dy


def fig_velocity_defo(U_grid, V_grid, div, name, color, top_right_quadrant = True):

    if top_right_quadrant == True:
        # slice the arrays to only retain top-smoll section quadrants to plot
        mid_x = U_grid.shape[1] // 4
        mid_y = V_grid.shape[0] // 4
        U_grid  = U_grid[mid_y:, mid_x:]
        V_grid = V_grid[mid_y:, mid_x:]
        div = div[mid_y:, mid_x:]
    
    div = calc_div(U_grid, V_grid)
    U_grid = U_grid[1:-1, 1:-1]
    V_grid = V_grid[1:-1, 1:-1]
    
    with plt.style.context(['science', 'no-latex']):
            # Create a single figure with one panel
            fig, ax = plt.subplots(figsize=(3, 4))

            # Create a mesh grid for the vector field
            x = np.arange(U_grid.shape[1])
            y = np.arange(V_grid.shape[0])
            X, Y = np.meshgrid(x, y)

            # Calculate speed for better visualization (optional, if needed)
            speed = np.sqrt(U_grid**2 + V_grid**2)

            # Use pcolormesh for background color representation
            ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', alpha=0.6, vmin = -1, vmax = 1)
    
            # Plot quivers for velocity field
            ax.quiver(X, Y, U_grid, V_grid, color='k', linewidth=1,width=0.002, scale=50, scale_units='xy',alpha=0.8)
           
            # Add a color bar for the pcolormesh
            cbar = plt.colorbar(ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto'), ax=ax, orientation='horizontal')
            cbar.set_label('Divergence', fontsize=18)

            # Set plot title and remove ticks
            ax.set_title(f"{name}", fontweight ="extra bold", color=color,  family='sans-serif')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add a legend for the quivers
            skip = 10
            skip = 1
            x_downsampled = x[::skip]
            y_downsampled = y[::skip]
            X_down, Y_down = np.meshgrid(x_downsampled, y_downsampled)
            quiver_key = ax.quiverkey(
                ax.quiver(X_down, Y_down, U_grid[::skip, ::skip], V_grid[::skip, ::skip], color='k', linewidth=2, width=0.0099, scale=0.9, scale_units='xy', alpha=0.8),
                X=0.85, Y=1.025, U=1, label='',labelpos='E', coordinates='axes' , fontproperties={'size': 18}
            )
            
            # Adjust spines
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            # Save the figure
            plt.tight_layout()
            combined_filename = os.path.join("SIMS/project/figures/divergence", f"{name}_div_velocity.png")
            plt.savefig(combined_filename, dpi=300)
            plt.close(fig)

    print(f"Divergence figure saved: {name}")
    
    # shear
    div = calc_shear_components(U_grid, V_grid)
    #div = calc_du_dy(U_grid, 1)
    #div = div[1:-1, 1:-1]
    U_grid = U_grid[1:-1, 1:-1]
    V_grid = V_grid[1:-1, 1:-1]
    
    with plt.style.context(['science', 'no-latex']):
            # Create a single figure with one panel
            fig, ax = plt.subplots(figsize=(3, 4))

            # Create a mesh grid for the vector field
            x = np.arange(U_grid.shape[1])
            y = np.arange(V_grid.shape[0])
            X, Y = np.meshgrid(x, y)

            # Calculate speed for better visualization (optional, if needed)
            speed = np.sqrt(U_grid**2 + V_grid**2)

            # Use pcolormesh for background color representation
            ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', alpha=0.6, vmin = -1, vmax = 1)
    
            # Plot quivers for velocity field
            ax.quiver(X, Y, U_grid, V_grid, color='k', linewidth=1,width=0.002, scale=50, scale_units='xy',alpha=0.8)
           
            # Add a color bar for the pcolormesh
            cbar = plt.colorbar(ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto'), ax=ax, orientation='horizontal')
            cbar.set_label('Divergence', fontsize=18)

            # Set plot title and remove ticks
            ax.set_title(f"{name}", fontweight ="extra bold", color=color,  family='sans-serif')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add a legend for the quivers
            skip = 10
            skip = 1
            x_downsampled = x[::skip]
            y_downsampled = y[::skip]
            X_down, Y_down = np.meshgrid(x_downsampled, y_downsampled)
            quiver_key = ax.quiverkey(
                ax.quiver(X_down, Y_down, U_grid[::skip, ::skip], V_grid[::skip, ::skip], color='k', linewidth=2, width=0.0099, scale=0.9, scale_units='xy', alpha=0.8),
                X=0.85, Y=1.025, U=1, label='',labelpos='E', coordinates='axes' , fontproperties={'size': 18}
            )

            # Adjust spines
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            # Save the figure
            plt.tight_layout()
            combined_filename = os.path.join("SIMS/project/figures/shear/shear_components", f"{name}_shear_velocity.png")
            plt.savefig(combined_filename, dpi=300)
            plt.close(fig)
            
            
            
            
    # velocity gradient components
    du_dx = calc_du_dx(U_grid, dx=1)
    du_dy = calc_du_dy(U_grid, dy=1)
    dv_dx = calc_dv_dx(V_grid, dx=1)
    dv_dy = calc_dv_dy(V_grid, dy=1)
    du_dx = du_dx[1:-1, 1:-1]
    du_dy = du_dy[1:-1, 1:-1]
    dv_dx = dv_dx[1:-1, 1:-1]
    dv_dy = dv_dy[1:-1, 1:-1]
    U_grid = U_grid[1:-1, 1:-1]
    V_grid = V_grid[1:-1, 1:-1]
    components = {
        "du/dx": du_dx,
        "dv/dy": dv_dy,
        "du/dy": du_dy,
        "dv/dx": dv_dx
    }

    # Create mesh grid
    x = np.arange(U_grid.shape[1])
    y = np.arange(V_grid.shape[0])
    X, Y = np.meshgrid(x, y)

    # Plot
    with plt.style.context(['science', 'no-latex']):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i, (label, data) in enumerate(components.items()):
            ax = axes[i]
            pcm = ax.pcolormesh(X, Y, data, cmap='coolwarm', shading='auto', vmin=-0.2, vmax=0.2,alpha=0.9)
            ax.quiver(X, Y, U_grid, V_grid, color='k', linewidth=1, width=0.002, scale=1, scale_units='xy', alpha=0.8)
            ax.set_title(label, fontweight="bold", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

        # Shared colorbar
        cbar = fig.colorbar(pcm, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.03, pad=0.08)
        cbar.set_label('Velocity gradients', fontsize=14)

        #plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures/velocity_gradients", f"{name}_velo_gradients_components.png")
        plt.savefig(combined_filename, dpi=300)
        plt.close(fig)
        
    return


def fig_defo_new(U_grid, V_grid, div, name, color, top_right_quadrant = False):
    
    if top_right_quadrant == True:
        # slice the arrays to only retain top-right quadrants to plot
        mid_x = U_grid.shape[1] // 2
        mid_y = V_grid.shape[0] // 2
        fraction = 1 # Adjust this value as needed
        reduced_x = int(mid_x + (U_grid.shape[1] - mid_x) * fraction)
        reduced_y = int(mid_y + (V_grid.shape[0] - mid_y) * fraction)
        U_grid = U_grid[mid_y:reduced_y, mid_x:reduced_x]
        V_grid = V_grid[mid_y:reduced_y, mid_x:reduced_x]
        div = div[mid_y:reduced_y, mid_x:reduced_x]
    
    U_grid = U_grid[1:-1, 1:-1]
    V_grid = V_grid[1:-1, 1:-1]
           
    with plt.style.context(['science', 'no-latex']):
        # Create a single figure with one panel
        fig, ax = plt.subplots(figsize=(4, 4))

        # Create a mesh grid for the vector field
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X, Y = np.meshgrid(x, y)
        print('HERE',np.shape(X))
        
        cmap = ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='nearest', alpha=0.6, vmin = -0.1, vmax = 0.1)
        
        ax.set_title(f"{name}", fontweight ="extra bold", color=color,  family='sans-serif', fontsize=23)
        ax.set_xticks([])
        ax.set_yticks([])

        # Adjust spines
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Save the figure
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tight_layout()
        combined_filename = os.path.join("SIMS/project/figures/deformation/", f"grid_{name}_div_velocity.png")
        plt.savefig(combined_filename, dpi=300)
        plt.close(fig)

    print(f"Divergence figure saved no quivers: {name}")
    
    return