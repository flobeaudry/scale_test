import numpy as np
import matplotlib.pyplot as plt
import os


def fig_velocity_defo(U_grid, V_grid, div, name, color, top_right_quadrant = True):
    #print(np.shape(np.where(U_grid != 0)))
    #print(np.shape(U_grid))
    #print(np.shape(np.where(V_grid != 0)))
    
    #print(np.mean(div))
    #print(np.sum(div))
    
    if top_right_quadrant == True:
        # slice the arrays to only retain top-right quadrants to plot
        mid_x = U_grid.shape[1] // 2
        mid_y = V_grid.shape[0] // 2
        U_grid  = U_grid[mid_y:, mid_x:]
        V_grid = V_grid[mid_y:, mid_x:]
        div = div[mid_y:, mid_x:]

    with plt.style.context(['science', 'no-latex']):
            # Create a single figure with one panel
            #fig, ax = plt.subplots(figsize=(9, 10))
            fig, ax = plt.subplots(figsize=(3, 4))


            # Create a mesh grid for the vector field
            x = np.arange(U_grid.shape[1])
            y = np.arange(V_grid.shape[0])
            X, Y = np.meshgrid(x, y)

            # Calculate speed for better visualization (optional, if needed)
            speed = np.sqrt(U_grid**2 + V_grid**2)

            # Use pcolormesh for background color representation
            ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', alpha=0.6, vmin = -1, vmax = 1)
            #ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', alpha=0.6, vmin = -2, vmax = 2)
    
            # Plot quivers for velocity field
            #ax.quiver(X, Y, U_grid, V_grid, color='k', linewidth=1,width=0.002, scale=50, scale_units='xy',alpha=0.8)
            #ax.quiver(X_down, Y_down, U_grid[::10, ::10], V_grid[::10, ::10], color='r', linewidth=1, width=0.002, scale=50, scale_units='xy', alpha=0.8)

            # Add a color bar for the pcolormesh
            cbar = plt.colorbar(ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', vmin = -1, vmax = 1), ax=ax, orientation='horizontal')
            #cbar = plt.colorbar(ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', vmin = -2, vmax = 2), ax=ax, orientation='horizontal')
            cbar.set_label('Divergence', fontsize=18)

            # Set plot title and remove ticks
            #ax.set_title(f"{name}", fontweight="bold", color="black")
            ax.set_title(f"{name}", fontweight ="extra bold", color=color,  family='sans-serif')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add a legend for the quivers
            skip = 10
            x_downsampled = x[::skip]
            y_downsampled = y[::skip]
            X_down, Y_down = np.meshgrid(x_downsampled, y_downsampled)
            quiver_key = ax.quiverkey(
                ax.quiver(X_down, Y_down, U_grid[::skip, ::skip], V_grid[::skip, ::skip], color='k', linewidth=2, width=0.0099, scale=0.9, scale_units='xy', alpha=0.8),
                X=0.85, Y=1.025, U=1, label='',labelpos='E', coordinates='axes' , fontproperties={'size': 18}
            )
            
            #print(U_grid[::skip, ::skip])
            #print(V_grid[1::skip, 1::skip])
            
            print(U_grid)
            print(V_grid)

            # Adjust spines
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            # Save the figure
            plt.tight_layout()
            combined_filename = os.path.join("SIMS/project/figures", f"quivs_Separated_{name}_div_velocity.png")
            plt.savefig(combined_filename, dpi=300)
            plt.close(fig)

    print(f"Divergence figure saved: {name}")
        
    return
        
def fig_defo(U_grid, V_grid, div, name, color, top_right_quadrant = True):
    
    if top_right_quadrant == True:
        # slice the arrays to only retain top-right quadrants to plot
        mid_x = U_grid.shape[1] // 2
        mid_y = V_grid.shape[0] // 2
        U_grid  = U_grid[mid_y:, mid_x:]
        V_grid = V_grid[mid_y:, mid_x:]
        div = div[mid_y:, mid_x:]
    
           
    with plt.style.context(['science', 'no-latex']):
        # Create a single figure with one panel
        #fig, ax = plt.subplots(figsize=(9, 10))
        fig, ax = plt.subplots(figsize=(4, 5))

        # Create a mesh grid for the vector field
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X, Y = np.meshgrid(x, y)

        # Calculate speed for better visualization (optional, if needed)
        speed = np.sqrt(U_grid**2 + V_grid**2)

        # Use pcolormesh for background color representation
        ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', alpha=0.6, vmin = -0.1, vmax = 0.1)
    
        print('size', np.shape(np.where(div != 0)))
        
        # Add a color bar for the pcolormesh
        cbar = plt.colorbar(ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='auto', vmin = -0.1, vmax = 0.1), ax=ax, orientation='horizontal')
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
        plt.savefig(combined_filename, dpi=300)
        plt.close(fig)

    print(f"Divergence figure saved no quivers: {name}")
    
    return


def fig_defo_new(U_grid, V_grid, div, name, color, top_right_quadrant = True):
    
    if top_right_quadrant == True:
        # slice the arrays to only retain top-right quadrants to plot
        mid_x = U_grid.shape[1] // 2
        mid_y = V_grid.shape[0] // 2
        fraction = 0.5 # Adjust this value as needed
        reduced_x = int(mid_x + (U_grid.shape[1] - mid_x) * fraction)
        reduced_y = int(mid_y + (V_grid.shape[0] - mid_y) * fraction)
        U_grid = U_grid[mid_y:reduced_y, mid_x:reduced_x]
        V_grid = V_grid[mid_y:reduced_y, mid_x:reduced_x]
        div = div[mid_y:reduced_y, mid_x:reduced_x]
    
           
    with plt.style.context(['science', 'no-latex']):
        # Create a single figure with one panel
        #fig, ax = plt.subplots(figsize=(9, 10))
        fig, ax = plt.subplots(figsize=(4, 4))

        # Create a mesh grid for the vector field
        x = np.arange(U_grid.shape[1])
        y = np.arange(V_grid.shape[0])
        X, Y = np.meshgrid(x, y)
        print('HERE',np.shape(X))
        
        # Use pcolormesh for background color representation
        #cmap = ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='nearest', alpha=0.6, vmin = -0.1, vmax = 0.1, edgecolors='black')
        #cmap = ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='nearest', alpha=0.6, vmin = -0.1, vmax = 0.1)
        cmap = ax.pcolormesh(X, Y, div, cmap='coolwarm', shading='nearest', alpha=0.6, edgecolors='black', vmin=-1, vmax=1)
        
        # Add a color bar for the pcolormesh
        #cbar = plt.colorbar(cmap, ax=ax, orientation='horizontal')
        #cbar.set_label('Divergence', fontsize=18)
        
        # Set plot title and remove ticks
        #ax.set_title(f"{name}", fontweight="bold", color="black")
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
        combined_filename = os.path.join("SIMS/project/figures", f"grid_{name}_div_velocity.png")
        plt.savefig(combined_filename, dpi=300)
        plt.close(fig)

    print(f"Divergence figure saved no quivers: {name}")
    
    return