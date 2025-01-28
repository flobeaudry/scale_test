import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def scale_and_coarse(u, v, L_values, dx, dy):
    """
    If on an Arakawa C-grid, need to obtain the velocities at the center of each cell
                ____o____
                |q_{i,j}|
        u_{i,j} o   o   o       where q is A, h, T, p
                |       |
                ----o----
                 v_{i,j}
    """

    #u_center = 0.5 * (u[:, :-1] + u[:, 1:])  # Average along x
    #v_center = 0.5 * (v[:-1, :] + v[1:, :])  # Average along y

    # Compute the velocity gradients based on a centered-difference scheme
    du_dx = (u[:, 1:] - u[:, :-1]) / dx  # Gradient of u in x-direction
    du_dy = (u[1:, :] - u[:-1, :]) / dy  # Gradient of u in y-direction
    dv_dx = (v[:, 1:] - v[:, :-1])/ dx  # Gradient of v in x-direction
    dv_dy = (v[1:, :] - v[:-1, :])/ dy  # Gradient of v in y-direction
    
    # Compute the velocity gradients based on a centered-difference scheme
    #du_dx = (u_center[:, 1:] - u_center[:, :-1])[1:-1,:] / dx  # Gradient of u in x-direction
    #du_dy = (u_center[1:, :] - u_center[:-1, :])[1:,1:] / dy  # Gradient of u in y-direction
    #dv_dx = (v_center[:, 1:] - v_center[:, :-1])[1:,1:] / dx  # Gradient of v in x-direction
    #dv_dy = (v_center[1:, :] - v_center[:-1, :])[:,1:-1] / dy  # Gradient of v in y-direction
    
    # Initialise things
    deformations_L = []
    
    # Main loop
    for L in L_values:
        step = L // 2
        coarse_defos = []
        
        for i in range(0, du_dx.shape[0]- L + 1, step):
            for j in range(0, du_dx.shape[1] - L + 1, step):
                block_du_dx = du_dx[i:i + L, j:j + L]
                block_du_dy = du_dy[i:i + L, j:j + L]
                block_dv_dx = dv_dx[i:i + L, j:j + L]
                block_dv_dy = dv_dy[i:i + L, j:j + L]
                
                du_dx_moy = np.nanmean(block_du_dx)
                du_dy_moy = np.nanmean(block_du_dy)
                dv_dx_moy = np.nanmean(block_dv_dx)
                dv_dy_moy = np.nanmean(block_dv_dy)
                
                divergence = du_dx_moy + dv_dy_moy
                shear = du_dy_moy - dv_dx_moy
                deformation = np.sqrt(divergence**2 + shear**2)
                
                coarse_defos.append(deformation)
                
        deformations_L.append(np.nanmean(coarse_defos))
    
    return deformations_L


def scaling_parameters(deformations_L, L_values):

    log_L_values = np.log(L_values)
    log_deformations = np.log(deformations_L)
    slope, intercept, _, _, _ = linregress(log_L_values, log_deformations)
     
    return(intercept, slope)
    
    
def scaling_figure(deformations, L_values, intercepts, slopes, names, colors):
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title('Spatial scaling')
        ax.grid(True, which='both')

        # Collect slope information for the legend
        legend_elements = []
    
        for i in range(len(deformations)):

            # Scatter plot and regression line
            ax.scatter(L_values, deformations[i], c=colors[i], s=60, alpha=1, edgecolors="k", zorder=1000)
            ax.plot(L_values, np.exp(intercepts[i]) * L_values**slopes[i], c=colors[i], linewidth=1.5,linestyle='-', zorder=500)

            #slopes_print = -1*slopes
            # Add slope value to the legend
            legend_elements.append((names[i] + f': {slopes[i]:.2f}',colors[i]))

        # Custom legend with only colored numbers
        legend_labels = [f'{text}' for text, _ in legend_elements]
        legend_colors = [color for _, color in legend_elements]
        legend_title = '$\\beta$'

        # Add text outside the plot as the legend
        ax.text(1.25, 0.85, legend_title, transform=ax.transAxes, fontsize=16, ha='center', va='center', fontweight='1000')
        for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
            ax.text(1.25, 0.85 - (i + 1.05) * 0.07, label, transform=ax.transAxes, fontsize=12, ha='center', va='center', color=color, weight='bold', family='sans-serif')

        # Finalize plot
        ax.set_xlabel('Spatial scale (nu)')
        ax.set_ylabel('$\\langle\\epsilon_{tot}\\rangle$')
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_ylim([1e-2, 1e0])
        
        # Save and show plot
        file_name = "SIMS/project/figures/Spatial_scaling_with_regression.png"
        fig.savefig(file_name, bbox_inches='tight')  # Adjust bounding box for custom annotations
        plt.close()





def scaling_fig(experiments, L_values):
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title('Spatial scaling')
        ax.grid(True, which='both')

        # Collect slope information for the legend
        legend_elements = []
    
        for exp in experiments:
            # Perform scaling
            deformations_L = exp['deformations']

            # Perform linear regression in log-log space
            log_L_values = np.log(L_values)
            log_deformations = np.log(deformations_L)
            slope, intercept, _, _, _ = linregress(log_L_values, log_deformations)

            # Scatter plot and regression line
            ax.scatter(L_values, deformations_L, c=exp['color'], s=60, alpha=1, edgecolors="k", zorder=1000)
            ax.plot(L_values, np.exp(intercept) * L_values**slope, c=exp['color'], linewidth=1.5,linestyle='-', zorder=500)

            # Add slope value to the legend
            #legend_elements.append((f'{slope:.2f}', exp['color']))
            legend_elements.append((exp['name'] + f': {slope:.2f}', exp['color']))

        # Custom legend with only colored numbers
        legend_labels = [f'{text}' for text, _ in legend_elements]
        legend_colors = [color for _, color in legend_elements]
        legend_title = '$\\beta$'

        # Add text outside the plot as the legend
        ax.text(1.15, 0.85, legend_title, transform=ax.transAxes, fontsize=16, ha='center', va='center', fontweight='1000')
        for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
            ax.text(1.15, 0.85 - (i + 1.05) * 0.07, label, transform=ax.transAxes, fontsize=14, ha='center', va='center', color=color, weight='bold', family='sans-serif')

        # Finalize plot
        ax.set_xlabel('Spatial scale (nu)')
        ax.set_ylabel('$\\langle\\epsilon_{tot}\\rangle$')
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Save and show plot
        file_name = "Spatial_scaling_with_regression.png"
        fig.savefig(file_name, bbox_inches='tight')  # Adjust bounding box for custom annotations
        plt.show()
