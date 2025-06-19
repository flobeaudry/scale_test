import numpy as np
import os
import cmocean
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.stats import linregress
from matplotlib.colors import LogNorm
import ruptures as rpt
import pickle
from ruptures.base import BaseCost
from ruptures.detection import Pelt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from utils.velocity_gradients_calc import calc_du_dx, calc_du_dy, calc_dv_dx, calc_dv_dy


def scale_and_coarse(u, v, u_noise, v_noise, L_values, dx, dy,c="c0", rgps='', scaling_on = "all"):
    """
    If on an Arakawa C-grid, need to obtain the velocities at the center of each cell
                ____o____
                |q_{i,j}|
        u_{i,j} o   o   o       where q is A, h, T, p
                |       |
                ----o----
                 v_{i,j}
    """
    
    # get velocity gradients
    du_dx = calc_du_dx(u, dx)[2:-2, 2:-2]
    dv_dy = calc_dv_dy(v, dy)[2:-2, 2:-2]
    dv_dx = calc_dv_dx(v, dx)[2:-2, 2:-2]
    du_dy = calc_du_dy(u, dy)[2:-2, 2:-2]
    
    print(np.shape(du_dy))
    # pad dudx and dvdy
    #du_dx = np.pad(du_dx, ((0, 0), (1, 0)), mode='constant')  # 1 col on the left
    #dv_dy = np.pad(dv_dy, ((0, 1), (0, 0)), mode='constant')  # 1 row at the bottom
    print("NON-ZERO wheres")
    print("du dx", np.where(du_dx != 0))
    print("dv dy", np.where(dv_dy != 0))
    print("dv dx", np.where(dv_dx != 0))
    print("du dy", np.where(du_dy != 0))
    
    u = u[2:-2, 2:-2]
    v = v[2:-2, 2:-2]
    
    if c == "vel_gradients":
        print("You directly specified the velocity gradients!")
        du_dx, dv_dy, du_dy, dv_dx = np.split(u, 4, axis=0)
    
    
    # define the noise on the velocity gradients !!
    if c=='err' or c=='weighted' or c =="weighted2":
        du_dx_noise = (np.random.randn(*np.shape(du_dx))*np.max(du_dx)/10)
        du_dy_noise = (np.random.randn(*np.shape(du_dy))*np.max(du_dy)/10)
        dv_dx_noise = (np.random.randn(*np.shape(dv_dx))*np.max(dv_dx)/10)
        dv_dy_noise = (np.random.randn(*np.shape(dv_dy))*np.max(dv_dy)/10)

        # add noise to velocity gradients
        du_dx = du_dx + du_dx_noise
        du_dy = du_dy + du_dy_noise
        dv_dx = dv_dx + dv_dx_noise
        dv_dy = dv_dy + dv_dy_noise
        
        # weighting the velocity gradients before the scaling
        if c == "weighted2":
            # do the weighting BEFORE anything else
            du_dx = (du_dx * abs(du_dx/du_dx_noise))/np.mean(abs(du_dx/du_dx_noise))
            du_dy = (du_dy * abs(du_dy/du_dy_noise))/np.mean(abs(du_dy/du_dy_noise))
            dv_dx = (dv_dx * abs(dv_dx/dv_dx_noise))/np.mean(abs(dv_dx/dv_dx_noise))
            dv_dy = (dv_dy * abs(dv_dy/dv_dy_noise))/np.mean(abs(dv_dy/dv_dy_noise))

            du_dx = np.nan_to_num(du_dx)
            du_dy = np.nan_to_num(du_dy)
            dv_dx = np.nan_to_num(dv_dx)
            dv_dy = np.nan_to_num(dv_dy)
            
    # if no noise, set the noise to zero
    else: 
        du_dx_noise = np.zeros_like(du_dx)
        du_dy_noise = np.zeros_like(du_dy)
        dv_dx_noise = np.zeros_like(dv_dx)
        dv_dy_noise = np.zeros_like(dv_dy)
    
    
    
    
    # load velocity gradients for model or rgps data
    if c=="rgps" or c == "sim":
        if c=="rgps":
            file_path = "SIMS/project/utils/rgps_export.npy"
        elif c=="sim":
            file_path = "SIMS/project/utils/sim_export2.npy"
            
        data = np.load(file_path)

        day_stamp = 0
        du_dx = data[:,:,day_stamp,0]
        du_dy = data[:,:,day_stamp,1]
        dv_dx = data[:,:,day_stamp,2]
        dv_dy = data[:,:,day_stamp,3]
        
        # set the noise to almost zero (in the gradients)
        du_dx_noise = np.zeros_like(du_dx)*1e-7
        du_dy_noise = np.zeros_like(du_dy)*1e-7
        dv_dx_noise = np.zeros_like(dv_dx)*1e-7
        dv_dy_noise = np.zeros_like(dv_dy)*1e-7
       
        
        # divergence tests for RGPS (diff conditions; to combine with the "scaling_on")
        if rgps == "div_pos":
            # Only positive divergence
            du_dx = np.where(np.isnan(du_dx), np.nan, np.where(du_dx >= 0, du_dx, 0))
            dv_dy = np.where(np.isnan(dv_dy), np.nan, np.where(dv_dy >= 0, dv_dy, 0))
        if rgps == "div_neg":
            # Only neg div
            du_dx = np.where(np.isnan(du_dx), np.nan, np.where(du_dx <= 0, du_dx, 0))
            dv_dy = np.where(np.isnan(dv_dy), np.nan, np.where(dv_dy <= 0, dv_dy, 0))
        if rgps == "div_abs":
            du_dx = abs(du_dx)
            dv_dy = abs(dv_dy)
        if rgps == "div":
            du_dx_p = np.where(np.isnan(du_dx), np.nan, np.where(du_dx >= 0, du_dx, 0))
            dv_dy_p = np.where(np.isnan(dv_dy), np.nan, np.where(dv_dy >= 0, dv_dy, 0))
            
            du_dx_n = np.where(np.isnan(du_dx), np.nan, np.where(du_dx <= 0, du_dx, 0))
            dv_dy_n = np.where(np.isnan(dv_dy), np.nan, np.where(dv_dy <= 0, dv_dy, 0))
            
            du_dx = du_dx_p + du_dx_n
            dv_dy = dv_dy_p + dv_dy_n

        # get RGPS defo
        div = du_dx + dv_dy
        #sh = du_dy - dv_dx
        sh = np.sqrt((du_dx - dv_dy)**2 + (du_dy + dv_dx)**2)
        defo = np.sqrt(div**2 + sh**2)

        # --------------------------------------  Thresholds with RGPS   ---------------------------------------------------
        #PDF and find threshold
        #div_max = np.nanmax(abs(defo))
        div_max = 0.1
        # Define the thresholds
        threshold1 = 0.1* div_max
        threshold2 = 0.2* div_max
        threshold3 = 0.3* div_max
        
        if rgps == "t1":
            threshold = threshold1
            color_th = 'xkcd:greenish'
            
        if rgps == "t2":
            threshold = threshold2
            color_th = 'xkcd:bluish purple'
            
        if rgps == "t3":
            threshold = threshold3
            color_th = 'xkcd:red orange'
        
        else: color_th = "k"
        
        if rgps == 't1' or rgps == 't2' or rgps == 't3':
            # Figure for PDF
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.hist(abs(defo.flatten()), bins=100, density=True, alpha=0.7, color='k')
            ax.axvline(threshold1, c='xkcd:greenish', linestyle="--", label=f"{threshold1:.2f}")
            ax.axvline(threshold2, c='xkcd:bluish purple', linestyle="--", label=f"{threshold2:.2f}")
            ax.axvline(threshold3, c='xkcd:red orange', linestyle="--", label=f"{threshold3:.2f}")
            #ax.set_title("PDF of RGPS divergence", fontsize=15, family='sans-serif')
            ax.set_xlim(0,0.2)
            ax.legend()
            plt.xlabel('Divergence')
            plt.ylabel('Probability Density')
            plt.grid(True, zorder=0)
            combined_filename = os.path.join("SIMS/project/figures", f"PDF.png")
            plt.savefig(combined_filename, dpi=300)        
            plt.close()
        
            #apply the threshold
            defo = np.where(abs(defo) >= threshold, div_max, 0)
            
            
            '''
            # to save the ratios of divergence to no-deformation grid cells:
            thresholds_to_pickle = [threshold1, threshold2, threshold3]
            # for t1:
            du_dx_t1 = np.where(np.isnan(du_dx), np.nan, np.where(abs(du_dx) >= threshold1, 1, 0))
            dv_dy_t1 = np.where(np.isnan(dv_dy), np.nan, np.where(abs(dv_dy) >= threshold1, 1, 0))
            #div_t1 = du_dx_t1 + dv_dy_t1
            div_t1 = np.nansum(np.dstack((du_dx_t1,dv_dy_t1)),2)
            num_ones_t1 = np.nansum(div_t1)
            num_zeros_t1 = div_t1.size - num_ones_t1
            ratio_1 = num_ones_t1 / num_zeros_t1
            # for t2:
            du_dx_t2 = np.where(np.isnan(du_dx), np.nan, np.where(abs(du_dx) >= threshold2, 1, 0))
            dv_dy_t2 = np.where(np.isnan(dv_dy), np.nan, np.where(abs(dv_dy) >= threshold2, 1, 0))
            #div_t2 = du_dx_t2 + dv_dy_t2
            div_t2 = np.nansum(np.dstack((du_dx_t2,dv_dy_t2)),2)
            num_ones_t2 = np.nansum(div_t2)
            num_zeros_t2 = div_t2.size - num_ones_t2
            ratio_2 = num_ones_t2 / num_zeros_t2
            # for t3:
            du_dx_t3 = np.where(np.isnan(du_dx), np.nan, np.where(abs(du_dx) >= threshold3, 1, 0))
            dv_dy_t3 = np.where(np.isnan(dv_dy), np.nan, np.where(abs(dv_dy) >= threshold3, 1, 0))
            #div_t3 = du_dx_t3 + dv_dy_t3
            div_t3 = np.nansum(np.dstack((du_dx_t3,dv_dy_t3)),2)
            num_ones_t3 = np.nansum(div_t3)
            num_zeros_t3 = div_t3.size - num_ones_t3
            ratio_3 = num_ones_t3 / num_zeros_t3
            #save to pickle
            ratios_to_pickle = [ratio_1, ratio_2, ratio_3]
            print('RATIOS!!', ratios_to_pickle)
            print('thresholds!!', thresholds_to_pickle)
            data = {'thresholds': thresholds_to_pickle, 'ratios': ratios_to_pickle}
            with open('SIMS/project/utils/rgps_div_ratios.pkl', 'wb') as f:
                pickle.dump(data, f)
            '''
            
            # Apply the thresholds on the divergence only 
            du_dx = np.where(np.isnan(du_dx), np.nan, np.where(abs(du_dx) >= threshold, div_max, 0))
            dv_dy = np.where(np.isnan(dv_dy), np.nan, np.where(abs(dv_dy) >= threshold, div_max, 0))
            du_dy = np.zeros_like(du_dy)
            dv_dx = np.zeros_like(dv_dx)
            
            # recalculate the deformation
            div = du_dx + dv_dy
            defo = div
        
        if rgps == "div" or rgps == 'div_abs' or rgps == "div_pos" or rgps == "div_neg":
            defo = div
            du_dy = np.zeros_like(du_dy)
            dv_dx = np.zeros_like(dv_dx)
        
        
        # load the already calculated deformations from Antoine's code (don't use; just for validation)      
        deps = np.load("SIMS/project/utils/rgps_deps.npy")
        # defo = deps

        
        plt.rcParams.update({'font.size': 16})
        with plt.style.context(['science', 'no-latex']):
            fig, ax = plt.subplots(figsize=(6, 7))
            plt.imshow(defo[30:-30, 10:-50], cmap="coolwarm",vmin=-0.05, vmax=0.05) 
            plt.colorbar(label="Deformation rate [days$^{-1}$]", shrink=0.6) 
            fig.patch.set_linewidth(2) 
            
            ax.set_xticks([])
            ax.set_yticks([])
        
            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            
            if c=="rgps":
                ax.set_title("RGPS div", fontweight ="extra bold", color=color_th, fontsize=18, family='sans-serif')
                combined_filename = os.path.join("SIMS/project/figures", f"RGPS.png")
            elif c=="sim":
                ax.set_title("model", fontweight ="extra bold", color="lightseagreen", fontsize=18, family='sans-serif')
                combined_filename = os.path.join("SIMS/project/figures", f"SIM.png")
                
            plt.savefig(combined_filename, dpi=300)        
            plt.close()

        
        
    
    if scaling_on != "all":
        print(scaling_on)
        if scaling_on == "du_dx":
            print("hi?")
            dv_dy, du_dy, dv_dx = np.zeros_like(du_dx), np.zeros_like(du_dx), np.zeros_like(du_dx)
            #dv_dy, du_dy, dv_dx = np.ones_like(du_dx)*np.NaN, np.ones_like(du_dx)*np.NaN, np.ones_like(du_dx)*np.NaN
        
        if scaling_on == "du_dy":
            dv_dy, du_dx, dv_dx = np.zeros_like(du_dx), np.zeros_like(du_dx), np.zeros_like(du_dx)
            
        if scaling_on == "dv_dx":
            dv_dy, du_dy, du_dx = np.zeros_like(du_dx), np.zeros_like(du_dx), np.zeros_like(du_dx)
        
        if scaling_on == "dv_dy":
            du_dy, du_dx, dv_dx = np.zeros_like(du_dx), np.zeros_like(du_dx), np.zeros_like(du_dx)
            
        if scaling_on == "div":
            du_dy, dv_dx = np.zeros_like(du_dx), np.zeros_like(du_dx)
        
        if scaling_on == "shear":
            dv_dy, du_dx = np.zeros_like(du_dx), np.zeros_like(du_dx)
            
        if scaling_on == "shuffle":
            # add this one here?
            # to randomize the velocoty gradients (without shuffling the nans)  
            #"""
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
            #"""
        
        #else:
        #    raise ValueError("Scaling condition unrecognized.")
    
    
    
    # -----------------------------------   Scaling   -----------------------------------------------------
    print("NON-ZERO before scaling")
    print("du dx", np.where(du_dx != 0))
    print("dv dy", np.where(dv_dy != 0))
    print("dv dx", np.where(dv_dx != 0))
    print("du dy", np.where(du_dy != 0))
    
    # Figure for PDF
    #fig, ax = plt.subplots(figsize=(6, 4))
    #plt.hist(du_dx[du_dx != 0].flatten(), bins=100, density=True, alpha=0.7, color='k')
    #combined_filename = os.path.join("SIMS/project/figures", f"PDF.png")
    #plt.savefig(combined_filename, dpi=300)        
    #plt.close()
    
    # Initialise things
    deformations_L = []
    deformations_Long = []
    
    if c == "c0":
        print(c)
        U_grid = u
        V_grid = v
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
            combined_filename = os.path.join("SIMS/project/figures/tests", f"velo_gradients_components.png")
            plt.savefig(combined_filename, dpi=300)
            plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=20) 

    # Main loop
    for L in L_values:
        step = L // 2
        coarse_defos = []
        coarse_defos_noise = []
        bool_defos = [] # counter to know how many grid points are not nans in a box
        bool_shape = []
        
        if L == 1: # no coarse graining is needed here
            du_dx_moy = du_dx
            du_dy_moy = du_dy
            dv_dx_moy = dv_dx
            dv_dy_moy = dv_dy
            
            du_dx_moy_noise = du_dx_noise
            du_dy_moy_noise = du_dy_noise
            dv_dx_moy_noise = dv_dx_noise
            dv_dy_moy_noise = dv_dy_noise
                    
            divergence = du_dx_moy + dv_dy_moy
            divergence_noise = du_dx_moy_noise + dv_dy_moy_noise
            
            shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
            #shear = (du_dy_moy + dv_dx_moy)
            shear_noise = np.sqrt((du_dx_moy_noise - dv_dy_moy_noise)**2 + (du_dy_moy_noise + dv_dx_moy_noise)**2)

            deformation = np.sqrt(divergence**2 + shear**2)
            deformation_noise = np.sqrt(divergence_noise**2 + shear_noise**2)
            
            #deformation = np.abs(du_dx_moy) # !!!!!!! test

            coarse_defos.append(deformation)
            coarse_defos_noise.append(deformation/deformation_noise) # for weighting
        
        # Coarse-graining         
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
                    
                    
                    block_du_dx_noise = du_dx_noise[i:i + L, j:j + L]
                    block_du_dy_noise = du_dy_noise[i:i + L, j:j + L]
                    block_dv_dx_noise = dv_dx_noise[i:i + L, j:j + L]
                    block_dv_dy_noise = dv_dy_noise[i:i + L, j:j + L]
                
                    du_dx_moy_noise = np.nanmean(block_du_dx_noise)
                    du_dy_moy_noise = np.nanmean(block_du_dy_noise)
                    dv_dx_moy_noise = np.nanmean(block_dv_dx_noise)
                    dv_dy_moy_noise = np.nanmean(block_dv_dy_noise)
                    
                
                    divergence = du_dx_moy + dv_dy_moy
                    divergence_noise = du_dx_moy_noise + dv_dy_moy_noise
                    
                    real_shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
                    #real_shear = (du_dy_moy + dv_dx_moy)
                    real_shear_noise = np.sqrt((du_dx_moy_noise - dv_dy_moy_noise)**2 + (du_dy_moy_noise + dv_dx_moy_noise)**2)
                    
                    if np.shape(block_du_dx)!= np.shape(block_dv_dy):
                        divergence_bool, shear_bool = 0, 0
                    
                    else:
                        divergence_bool = block_du_dx + block_dv_dy
                        real_shear_bool = np.sqrt((block_du_dx - block_dv_dy)**2 + (block_du_dy + block_dv_dx)**2) #good one
                        shear_bool = np.sqrt((block_du_dx - block_dv_dy)**2) # use for div only
                        
                        dudx_bool = np.abs(block_du_dx)


                    deformation = np.sqrt(divergence**2 + real_shear**2)
                    #deformation = np.abs(du_dx_moy) # !!!!!!! test
                    deformation_bool = np.sqrt(divergence_bool**2 + real_shear_bool**2)
                    #deformation_bool = np.abs(dudx_bool) # !!!!!!! test
                    deformation_noise = np.sqrt(divergence_noise**2 + real_shear_noise**2)
       
                
                    bool_defos.append(np.sum(~np.isnan(deformation_bool))) #count the non nan values
                    bool_shape.append(deformation_bool.size)

                    coarse_defos.append(deformation)
                    coarse_defos_noise.append(deformation/deformation_noise)
        

            coarse_defos = np.where(
                    np.array(bool_defos) < L** 2 // 2, np.nan, np.array(coarse_defos),
                )  
            coarse_defos_noise = np.where(
                    np.array(bool_defos) < L** 2 // 2, np.nan, np.array(coarse_defos_noise),
                )  
    
        
        if c == "weighted":
            x = L + np.random.uniform(-0.3, 0.3, size=np.array(coarse_defos).shape)
            sc = ax.scatter(x, coarse_defos, c=coarse_defos_noise,alpha=0.7, cmap=cmap, norm=LogNorm(vmin=1, vmax = 1000), s=6)
            mask = ~np.isnan(coarse_defos) & ~np.isnan(coarse_defos_noise)
            a= np.sum(np.array(coarse_defos)[mask] * np.array(coarse_defos_noise)[mask]) / np.sum(np.array(coarse_defos_noise)[mask])
            deformations_L.append(a)
            
        else:
            deformations_L.append(np.nanmean(coarse_defos))
        
        deformations_Long.append((coarse_defos))
    
    if c == "weighted":
        print("scaling snr fig !!")
        ax.set_xlabel('L')
        ax.set_ylabel('Deformation')
        ax.set_title('Deformations colored by SNR')
        ax.set_xscale('log')
        ax.set_yscale("log")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('SNR')
        file_name = "SIMS/project/figures/spatial_scaling_SNR.png"
        fig.savefig(file_name, bbox_inches='tight', dpi=300)  # Adjust bounding box for custom annotations
    plt.close()
        
    return deformations_L, deformations_Long

class CostSlope(BaseCost):
    model = "slope"

    def fit(self, signal):
        self.signal = signal
        return self

    def error(self, start, end):
        x = self.signal[start:end, 0]
        y = self.signal[start:end, 1]
        if len(x) < 2:
            return np.inf  # can't compute slope
        slope, _, r_value, _, _ = linregress(x, y)
        print(1 - r_value**2)
        return (1 - r_value**2)  # lower is better (closer to linear)
    

def scaling_segments(deformations_L, L_values, name = "exp", max_breaks=2):
    
    """
    Function to find scaling exponent (beta), the intercept and the r^2 of different segments (scaling steps)
    """
    
    segment = []
    
    log_L = np.log(L_values)
    log_def = np.log(deformations_L)
    
    if name == "rgps":
        slope, intercept, r, _, _ = linregress(log_L, log_def)
        return [{
            "start": 0,
            "end": len(log_L),
            "intercept": intercept,
            "slope": slope,
            "r_squared": r**2,
            "x_range": (L_values[0], L_values[-1])
        }]
        
    # Fail-safe for NaNs
    if np.any(np.isnan(log_def)):
        print("NaNs detected in deformation data. Returning single regression segment.")
        nan_where = np.argwhere(np.isnan(log_def))
        log_def = np.delete(log_def, nan_where)
        log_L = np.delete(log_L, nan_where)
        slope, intercept, r, _, _ = linregress(log_L, log_def)
        return [{
            "start": 0,
            "end": len(log_L),
            "intercept": intercept,
            "slope": slope,
            "r_squared": r**2,
            "x_range": (L_values[0], L_values[-1])
        }]
    
    log_def_smooth = gaussian_filter1d(log_def, sigma=0.3)
    
    dy = log_def[:-1] - log_def[1:]
    ddy = dy[:-1] - dy[1:]

    peaks = np.where((np.abs(ddy) > np.mean(np.abs(ddy))) & (np.abs(ddy) > np.std(np.abs(ddy))))[0]
    peaks = peaks+1
    # Add endpoints for segmentation
    breakpoints = [0] + peaks.tolist() + [len(log_L)-1]
    breakpoints = sorted(list(set(breakpoints)))

    
    for i in range(len(breakpoints)-1):
        start, end = breakpoints[i], breakpoints[i+1]+1  # include endpoint
        x_seg = log_L[start:end]
        y_seg = log_def[start:end]
        slope, intercept, r, _, _ = linregress(x_seg, y_seg)
        y_fit = slope * x_seg + intercept
        r_squared = r**2
    
        segment.append({
            "start": start,
            "end": end,
            "intercept": intercept,
            "slope": slope,
            "r_squared": r_squared,
            "x_range": (L_values[start], L_values[end-1])
        })
    
    return segment
    
    
def scaling_figure(deformations, L_values, segments, names, colors, markers, r_threshold=0.3, linestyle="-"):

    plt.rcParams.update({'font.size': 16})
    with plt.style.context(['science', 'no-latex']):
        size_text = 18
        fig, ax = plt.subplots(figsize=(7, 5))
        #ax.set_title('Spatial scaling', fontsize=size_text)
        ax.grid(True, which='both')
        
        fig.patch.set_linewidth(2) 

        # Collect slope information for the legend
        legend_elements = []
    
        for i in range(len(deformations)):
            if colors[i] == "black" or colors[i] == "grey" or colors[i] == "red" or colors[i] == "blue" or colors[i] == "xkcd:greenish" or colors[i] == "xkcd:bluish purple" or colors[i] == "xkcd:red orange":
                ax.scatter(np.array(L_values)*10, deformations[i], c=colors[i], marker='^',s=60, alpha=1, edgecolors="k", zorder=1000)
                markers[i] = '^'
               
                # for the multiple slopes (scaling_sgments fcn)
                experiment_segments = segments[i]
                for seg in experiment_segments:
                    if colors[i] != "black":
                        L_fit = np.linspace(seg["x_range"][0], seg["x_range"][1], 100)
                        fit = np.exp(seg["intercept"]) * L_fit**seg["slope"]
                        ax.plot(L_fit * 10, fit, c=colors[i], linewidth=1.5, linestyle=':', zorder=500)
        
            else:
                if markers[i]=="s":
                    # Scatter plot and regression line
                    ax.scatter(np.array(L_values)*10, deformations[i], c=colors[i], marker=markers[i], s=40, alpha=1, edgecolors="k", zorder=1000)
                    # for the multiple slopes (scaling_sgments fcn)
                    experiment_segments = segments[i]
                    for seg in experiment_segments:
                        L_fit = np.linspace(seg["x_range"][0], seg["x_range"][1], 100)
                        fit = np.exp(seg["intercept"]) * L_fit**seg["slope"]
                        ax.plot(L_fit * 10, fit, c=colors[i], linewidth=1.5, linestyle=':', zorder=500)
                else:
                    # Scatter plot and regression line
                    ax.scatter(np.array(L_values)*10, deformations[i], c=colors[i], marker=markers[i], s=60, alpha=1, edgecolors="k", zorder=1000)
                    experiment_segments = segments[i]
                    for seg in experiment_segments:
                        L_fit = np.linspace(seg["x_range"][0], seg["x_range"][1], 100)
                        fit = np.exp(seg["intercept"]) * L_fit**seg["slope"]
                        ax.plot(L_fit * 10, fit, c=colors[i], linewidth=1.5, linestyle=linestyle, zorder=500)
             
            legend_elements.append((names[i],colors[i], markers[i]))       

        # Custom legend with only colored numbers
        legend_labels = [f'{text}' for text, _,_ in legend_elements]
        legend_colors = [color for _, color, _ in legend_elements]
        legend_markers = [marker for _,_, marker in legend_elements]
        legend_title = " "

        
        x0 = 1.05  # leftmost position for marker (adjust as needed)
        text_offset = 0.03  # distance between marker and text

        ax.text(x0, 0.95, legend_title, transform=ax.transAxes,
                fontsize=size_text, ha='left', va='center', fontweight='1000')

        for i, (label, color, marker) in enumerate(zip(legend_labels, legend_colors, legend_markers)):
            y = 0.85 - (i + 1.05) * 0.07

            # scatter marker
            ax.scatter([x0], [y], transform=ax.transAxes,
                    marker=marker, edgecolors='k', color=color,
                    s=40, clip_on=False)

            # label text just to the right of the marker
            ax.text(x0 + text_offset, y, label, transform=ax.transAxes,
                    fontsize=size_text-5, ha='left', va='center', color=color,
                    weight='bold', family='sans-serif')
            
        # Finalize plot
        ax.set_xlabel('Spatial scale (km)', fontsize=size_text)
        #ax.set_ylabel('$\\langle\\epsilon_{tot}\\rangle$')
        ax.set_ylabel('$\\langle\\dot{\\epsilon}_{tot}\\rangle$', fontsize=size_text+3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        #ax.set_ylim([1e-3, 1e0]) # good one !

        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Save and show plot
        file_name = "SIMS/project/figures/Spatial_scaling_with_regression.png"
        fig.savefig(file_name, bbox_inches='tight', dpi=300)  # Adjust bounding box for custom annotations
        plt.close()