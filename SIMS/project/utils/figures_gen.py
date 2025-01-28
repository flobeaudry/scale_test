import numpy as np
import matplotlib.pyplot as plt
import os


def fig_velocity_defo(U_grid, V_grid, div, name, color):

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
        
    return
        
def fig_defo(U_grid, V_grid, div, name, color):
           
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
    
    return