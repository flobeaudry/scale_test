import numpy as np
import os
import cmocean
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.stats import linregress

def scale_and_coarse(u, v, L_values, dx, dy,c="c0"):
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
    
    du_dx = du_dx[1:,:]
    du_dy = du_dy[:,1:]
    dv_dx = dv_dx[1:,:]
    dv_dy = dv_dy[:,1:]

    #np.random.shuffle(du_dx)
    #np.random.shuffle(du_dy)
    #np.random.shuffle(dv_dx)
    #np.random.shuffle(dv_dy)
    
    if c=="rgps" or c == "sim":
        if c=="rgps":
            file_path = "SIMS/project/rgps_export.npy"
            #file_path = "SIMS/project/sim_export.npy"
        elif c=="sim":
            file_path = "SIMS/project/sim_export2.npy"
            
        data = np.load(file_path)

        day_stamp = 0
        du_dx = data[:,:,day_stamp,0]
        du_dy = data[:,:,day_stamp,1]
        dv_dx = data[:,:,day_stamp,2]
        dv_dy = data[:,:,day_stamp,3]
        
        
        #du_dx = abs(du_dx)
        #du_dy = abs(du_dy)
        #dv_dx = abs(dv_dx)
        #dv_dy = abs(dv_dy)
        
        """
        flat_dudx = du_dx[~np.isnan(du_dx)].flatten()
        np.random.shuffle(flat_dudx)
        du_dx[~np.isnan(du_dx)] = flat_dudx

        flat_dudy = du_dy[~np.isnan(du_dy)].flatten()
        np.random.shuffle(flat_dudy)
        du_dy[~np.isnan(du_dy)] = flat_dudy

        flat_dvdx = dv_dx[~np.isnan(dv_dx)].flatten()
        np.random.shuffle(flat_dvdx)
        dv_dx[~np.isnan(dv_dx)] = flat_dvdx

        flat_dvdy = dv_dy[~np.isnan(dv_dy)].flatten()
        np.random.shuffle(flat_dvdy)
        dv_dy[~np.isnan(dv_dy)] = flat_dvdy
        """
        
        #du_dx = np.nanmean(data[:, :, :, 0], axis=2)
        #du_dy = np.nanmean(data[:, :, :, 1], axis=2)
        #dv_dx = np.nanmean(data[:, :, :, 2], axis=2)
        #dv_dy = np.nanmean(data[:, :, :, 3], axis=2)
        
        #np.random.shuffle(du_dx)
        #np.random.shuffle(du_dy)
        #np.random.shuffle(dv_dx)
        #np.random.shuffle(dv_dy)
        
        div = du_dx + dv_dy
        #sh = du_dy - dv_dx
        sh = np.sqrt((du_dx - dv_dy)**2 + (du_dy + dv_dx)**2)
        #div[np.abs(div) < 5e-3] = np.nan
        #sh[np.abs(sh) < 5e-3] = np.nan
        
        defo = np.sqrt(div**2 + sh**2)
        
        #flat_defo = defo.flatten()
        #np.random.shuffle(flat_defo)
        #defo = flat_defo.reshape(defo.shape)
        #defo=sh
        #defo = du_dy + dv_dx
        #defo = div
        
        deps = np.load("SIMS/project/rgps_deps.npy")
        
        #defo = deps
        
        #plt.figure(figsize=(6, 6))  # Adjust the size as needed
        #plt.imshow(defo, cmap=cmocean.cm.thermal, vmax=0.1)  # 'viridis' is a popular colormap, but there are many options
        #plt.colorbar(label="Deformation rate [d]") 

        
        plt.rcParams.update({'font.size': 16})
        with plt.style.context(['science', 'no-latex']):
            fig, ax = plt.subplots(figsize=(6, 7))
            #plt.imshow(defo, cmap=cmocean.cm.thermal, vmax=0.1) 
            
            #ax.set_facecolor("lightgray")
            #plt.imshow(defo, cmap="cubehelix_r", vmax=0.1) 
            
            plt.imshow(defo[30:-30, 10:-50], cmap="coolwarm",vmin=-0.1, vmax=0.1) 
            
            plt.colorbar(label="Deformation rate [days$^{-1}$]", shrink=0.6) 
        
            fig.patch.set_linewidth(2) 
            
            ax.set_xticks([])
            ax.set_yticks([])
        
            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            
            if c=="rgps":
                ax.set_title("obs", fontweight ="extra bold", color="k", fontsize=18, family='sans-serif')
                combined_filename = os.path.join("SIMS/project/figures", f"RGPS.png")
            elif c=="sim":
                ax.set_title("model", fontweight ="extra bold", color="lightseagreen", fontsize=18, family='sans-serif')
                combined_filename = os.path.join("SIMS/project/figures", f"SIM.png")
                
            plt.savefig(combined_filename, dpi=300)        
            plt.close()

        
        
        
    # Compute the velocity gradients based on a centered-difference scheme
    #du_dx = (u_center[:, 1:] - u_center[:, :-1])[1:-1,:] / dx  # Gradient of u in x-direction
    #du_dy = (u_center[1:, :] - u_center[:-1, :])[1:,1:] / dy  # Gradient of u in y-direction
    #dv_dx = (v_center[:, 1:] - v_center[:, :-1])[1:,1:] / dx  # Gradient of v in x-direction
    #dv_dy = (v_center[1:, :] - v_center[:-1, :])[:,1:-1] / dy  # Gradient of v in y-direction
    
    # Initialise things
    deformations_L = []
    deformations_Long = []
    
    # Main loop
    for L in L_values:
        step = L // 2
        coarse_defos = []
        bool_defos = [] # counter to know how many grid points are not nans in a box
        bool_shape = []
        
        if L == 1: # no coarse graining is needed here
            du_dx_moy = du_dx
            du_dy_moy = du_dy
            dv_dx_moy = dv_dx
            dv_dy_moy = dv_dy
            divergence = du_dx_moy + dv_dy_moy
            if c == 'rgps':
                shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
            else : shear = np.sqrt((du_dx_moy - dv_dy_moy)**2)
                                   
            #shear = du_dy_moy + dv_dx_moy
            deformation = np.sqrt(divergence**2 + shear**2)
            coarse_defos.append(deformation)
                 
        else:    
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
                
                    #du_dx_moy = np.nanmean(abs(block_du_dx))
                    #du_dy_moy = np.nanmean(abs(block_du_dy))
                    #dv_dx_moy = np.nanmean(abs(block_dv_dx))
                    #dv_dy_moy = np.nanmean(abs(block_dv_dy))
                
                    divergence = du_dx_moy + dv_dy_moy
                    #shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
                    #shear = du_dy_moy + dv_dx_moy
                    shear = np.sqrt((du_dx_moy - dv_dy_moy)**2)
                    real_shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
                    
                    if np.shape(block_du_dx)!= np.shape(block_dv_dy):
                        divergence_bool, shear_bool = 0, 0
                    
                    else:
                        divergence_bool = block_du_dx + block_dv_dy
                        real_shear_bool = np.sqrt((block_du_dx - block_dv_dy)**2 + (block_du_dy + block_dv_dx)**2) #good one
                        shear_bool = np.sqrt((block_du_dx - block_dv_dy)**2) # use for div only
                
                    if c == 'c0':
                        #deformation = divergence
                        deformation = np.sqrt(divergence**2 + shear**2)
                        deformation_bool = np.sqrt(divergence_bool**2 + shear_bool**2)
                        #deformation = np.sqrt(divergence**2)
                        #deformation = np.sqrt(shear**2)
                        
                    elif c == 'rgps':
                        deformation = np.sqrt(divergence**2 + real_shear**2)
                        deformation_bool = np.sqrt(divergence_bool**2 + real_shear_bool**2)
                        
                    else :
                        deformation = np.sqrt(divergence**2 + shear**2)
                        deformation_bool = np.sqrt(divergence_bool**2 + shear_bool**2)
                        #deformation = np.sqrt(divergence**2)
                        #deformation = np.sqrt(shear**2)
                
                    bool_defos.append(np.sum(~np.isnan(deformation_bool))) #count the non nan values
                    bool_shape.append(deformation_bool.size)
                    coarse_defos.append(deformation)
        
        #print("SHAPE",np.shape(coarse_defos))
            coarse_defos = np.where(
                    np.array(bool_defos) < L** 2 // 2, np.nan, np.array(coarse_defos),
                )  
        #coarse_defos = np.where(
        #        np.array(bool_defos) < 1, np.nan, np.array(coarse_defos),
        #    )      
        #print("SHAPE2",np.shape(coarse_defos))
        #print(bool_defos)
        #print(bool_shape)
        
        deformations_L.append(np.nanmean(coarse_defos))
        deformations_Long.append((coarse_defos))

        #print(10/L)
        #print(np.nanmean(coarse_defos))
        #print((np.nanmean(coarse_defos))/(10/L))
        #deformations_L.append(np.nanmean(coarse_defos)/(10*0.1/L))
        
    return deformations_L, deformations_Long


def scaling_parameters(deformations_L, L_values):

    log_L_values = np.log(L_values)
    log_deformations = np.log(deformations_L)
    slope, intercept, r_value, _, _ = linregress(log_L_values, log_deformations)
    r_squared = r_value**2
    
    return(intercept, slope, r_squared)
    
    
def scaling_figure(deformations, L_values, intercepts, slopes, r2s, names, colors, r_threshold=0.85, linestyle="-"):
    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        size_text = 18
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title('Spatial scaling', fontsize=size_text)
        ax.grid(True, which='both')
        
        fig.patch.set_linewidth(2) 

        # Collect slope information for the legend
        legend_elements = []
    
        for i in range(len(deformations)):
            if colors[i] == "black":
                ax.scatter(np.array(L_values)*10, deformations[i], c=colors[i], marker='^',s=60, alpha=1, edgecolors="k", zorder=1000)
                if r2s[i] > r_threshold:
                    #ax.plot(np.array(L_values)*10, np.exp(intercepts[i]) * L_values**-slopes[i], c=colors[i], linewidth=1.5,linestyle='--', zorder=500)
                    ax.plot(np.array(L_values)*10, np.exp(intercepts[i]) * L_values**-slopes[i], c=colors[i], linewidth=1.5,linestyle=':', zorder=500)
            else:
                # Scatter plot and regression line
                print("HI",slopes)
                ax.scatter(np.array(L_values)*10, deformations[i], c=colors[i], s=60, alpha=1, edgecolors="k", zorder=1000)
                if r2s[i] > r_threshold:
                    ax.plot(np.array(L_values)*10, np.exp(intercepts[i]) * L_values**-slopes[i], c=colors[i], linewidth=1.5,linestyle=linestyle, zorder=500)

            #slopes_print = -1*slopes
            # Add slope value to the legend
            if r2s[i] > r_threshold:
                #legend_elements.append((names[i] + f': {slopes[i]:.2f}',colors[i]))
                if slopes[i]<0.01:
                    #legend_elements.append((f'{abs(slopes[i]):.2f}',colors[i]))
                    legend_elements.append((names[i],colors[i]))
                else:
                    #legend_elements.append((f'{slopes[i]:.2f}',colors[i]))
                    legend_elements.append((names[i],colors[i]))
                    
            else:
                #legend_elements.append((names[i],colors[i]))
                #legend_elements.append(("-",colors[i]))
                legend_elements.append((names[i],colors[i]))
                
            #legend_elements.append((f'{slopes[i]:.2f}',colors[i]))


        #slope = -np.log(0.009/0.05)/np.log(640/20)
        #ax.plot(np.array([L_values[0],L_values[-1]])*10, np.array([0.05,0.009]), c='k', linestyle='--')
        #legend_elements.append((f'{slope:.2f}',"k"))
        # Custom legend with only colored numbers
        legend_labels = [f'{text}' for text, _ in legend_elements]
        legend_colors = [color for _, color in legend_elements]
        #legend_title = '$\\beta$'
        legend_title = "Experiment"

        # Add text outside the plot as the legend
        pos_x = 1.27
        #pos_x = 1.25
        ax.text(pos_x, 0.85, legend_title, transform=ax.transAxes, fontsize=size_text, ha='center', va='center', fontweight='1000')
        for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
            ax.text(pos_x, 0.85 - (i + 1.05) * 0.07, label, transform=ax.transAxes, fontsize=size_text-5, ha='center', va='center', color=color, weight='bold', family='sans-serif')

        # Finalize plot
        ax.set_xlabel('Spatial scale (km)', fontsize=size_text)
        #ax.set_ylabel('$\\langle\\epsilon_{tot}\\rangle$')
        ax.set_ylabel('$\\langle\\dot{\\epsilon}_{tot}\\rangle$', fontsize=size_text+3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        #ax.set_ylim([1e-1, 1e0])
        #ax.set_ylim([5e-1, 5e0])
        
        #ax.set_ylim([1e-3, 1e0]) # good one !
        ax.set_ylim([5e-4, 1e0])
        #ax.set_ylim([5e-3, 1e0]) # good one !
        
        #ax.set_ylim([8e-3, 2.1e-1])
        
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Save and show plot
        file_name = "SIMS/project/figures/Spatial_scaling_with_regression.png"
        fig.savefig(file_name, bbox_inches='tight', dpi=300)  # Adjust bounding box for custom annotations
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
        #ax.set_ylabel('$\\langle\\epsilon_{I}\\rangle$')
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Save and show plot
        file_name = "Spatial_scaling_with_regression.png"
        fig.savefig(file_name, bbox_inches='tight')  # Adjust bounding box for custom annotations
        plt.show()
