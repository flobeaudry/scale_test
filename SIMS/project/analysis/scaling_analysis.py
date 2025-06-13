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


def scale_and_coarse(u, v, u_noise, v_noise, L_values, dx, dy,c="c0", rgps=''):
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
    #du_dy = (u[1:, :] - u[:-1, :]) / dy  # Gradient of u in y-direction
    #dv_dx = (v[:, 1:] - v[:, :-1])/ dx  # Gradient of v in x-direction
    dv_dy = (v[1:, :] - v[:-1, :])/ dy  # Gradient of v in y-direction
    
    print("du_dx non zero", np.shape(np.where(du_dx != 0)))
    print("dv_dy non zero", np.shape(np.where(dv_dy != 0)))
    
    # Try other approach with shear
    #u_ip1jp1 = u[1: ,1:]
    #u_ijp1 = u[1:, :-1]
    #u_ip1jm1 = np.vstack((u[:-2, 1:], u[0, 1:]))
    #u_ijm1 = np.vstack((u[:-2 ,:-1], u[0, 1:]))

    #du_dy = (u_ip1jp1 + u_ijp1 - u_ip1jm1 - u_ijm1) / (2 * dy)
    
    ## dudy_diag ~ from diagonals of u
    #u_ip1jp1 = u[1:, 1:]
    #u_ijp1   = u[:-1, 1:]
    #u_ip1jm1 = u[1:, :-1]
    #u_ijm1   = u[:-1, :-1]
    #du_dy = (u_ip1jp1 + u_ijp1 - u_ip1jm1 - u_ijm1) / (2 * dy)
    # dvdx_diag ~ from diagonals of v
    #v_ip1jp1 = v[1:, 1:]
    #v_ip1    = v[1:, :-1]
    #v_im1jp1 = v[:-1, 1:]
    #v_im1    = v[:-1, :-1]
    #dv_dx = (v_ip1jp1 + v_ip1 - v_im1jp1 - v_im1) / (2 * dx)
    
    
    
    # dvdx with periodic in y (vertical), zero in x (horizontal)
    v_ip1jp1 = np.zeros_like(v)
    v_ip1    = np.zeros_like(v)
    v_im1jp1 = np.zeros_like(v)
    v_im1    = np.zeros_like(v)

    # roll in x (axis=1) for periodicity
    # bottom bndy = Drichlet, and left and right are periodic !
    v_ip1jp1[1:-1, :] = np.roll(v, -1, axis=1)[2:, :]   # v[i+1, j+1]
    v_ip1[1:-1, :]    = np.roll(v, -1, axis=1)[1:-1, :] # v[i+1, j]
    v_im1jp1[1:-1, :] = np.roll(v, 1, axis=1)[2:, :]    # v[i-1, j+1]
    v_im1[1:-1, :]    = np.roll(v, 1, axis=1)[1:-1, :]  # v[i-1, j]

    dv_dx = (v_ip1jp1 + v_ip1 - v_im1jp1 - v_im1) / (2 * dx) # remove comment for shear too!
    print("dv_dx non zero", np.shape(np.where(dv_dx != 0)))
    #dv_dx = np.zeros_like(dv_dy)

    #dv_dx = dv_dx[1:,1:]
    
    # FOR THE SHEAR (need to do but should automatize)
    #dv_dx = dv_dx[:-2,2:]
    #du_dx = np.zeros_like(dv_dx)
    #du_dy = np.zeros_like(dv_dx)
    #dv_dy = np.zeros_like(dv_dx)
    
    du_dx = du_dx[1:,:]
    print('HERE',np.shape(du_dx))
    #du_dy = du_dy[:,1:]
    #dv_dx = dv_dx[1:,:]
    dv_dy = dv_dy[:,1:]
    
    du_dy = np.zeros_like(du_dx)
    dv_dx = np.zeros_like(du_dx)
    
    # Noise !!
    #du_dx_noise = (u_noise[:, 1:] - u_noise[:, :-1]) / dx  # Gradient of u in x-direction
    #du_dy_noise = (u_noise[1:, :] - u_noise[:-1, :]) / dy  # Gradient of u in y-direction
    #dv_dx_noise = (v_noise[:, 1:] - v_noise[:, :-1])/ dx  # Gradient of v in x-direction
    #dv_dy_noise = (v_noise[1:, :] - v_noise[:-1, :])/ dy  # Gradient of v in y-direction
    
    #du_dx_noise = du_dx_noise[1:,:]
    #du_dy_noise = du_dy_noise[:,1:]
    #dv_dx_noise = dv_dx_noise[1:,:]
    #dv_dy_noise = dv_dy_noise[:,1:]
    
    # define the noise on the velocity gradients !!
    #print(c)
    if c=='err' or c=='weighted' or c =="weighted2":
        du_dx_noise = (np.random.randn(*np.shape(du_dx))*np.max(du_dx)/10)
        du_dy_noise = (np.random.randn(*np.shape(du_dy))*np.max(du_dy)/10)
        dv_dx_noise = (np.random.randn(*np.shape(dv_dx))*np.max(dv_dx)/10)
        dv_dy_noise = (np.random.randn(*np.shape(dv_dy))*np.max(dv_dy)/10)
        
        #print('du_dx no noise:',du_dx)
        
        du_dx = du_dx + du_dx_noise
        du_dy = du_dy + du_dy_noise
        dv_dx = dv_dx + dv_dx_noise
        dv_dy = dv_dy + dv_dy_noise
        
        #print('du_dx noise', du_dx)
        
        if c == "weighted2":
            #print('dvdy', dv_dy)
            # do the weighting BEFORE anything else
            du_dx = (du_dx * abs(du_dx/du_dx_noise))/np.mean(abs(du_dx/du_dx_noise))
            du_dy = (du_dy * abs(du_dy/du_dy_noise))/np.mean(abs(du_dy/du_dy_noise))
            dv_dx = (dv_dx * abs(dv_dx/dv_dx_noise))/np.mean(abs(dv_dx/dv_dx_noise))
            dv_dy = (dv_dy * abs(dv_dy/dv_dy_noise))/np.mean(abs(dv_dy/dv_dy_noise))
            
            #du_dx = (du_dx * abs(du_dx/du_dx_noise))
            #du_dy = (du_dy * abs(du_dy/du_dy_noise))
            #dv_dx = (dv_dx * abs(dv_dx/dv_dx_noise))
            #dv_dy = (dv_dy * abs(dv_dy/dv_dy_noise))

            du_dx = np.nan_to_num(du_dx)
            du_dy = np.nan_to_num(du_dy)
            dv_dx = np.nan_to_num(dv_dx)
            dv_dy = np.nan_to_num(dv_dy)
            #print("DUDX", du_dx)
            #print("DVDY", dv_dy)
    else: 
        du_dx_noise = np.zeros_like(du_dx)
        du_dy_noise = np.zeros_like(du_dy)
        dv_dx_noise = np.zeros_like(dv_dx)
        dv_dy_noise = np.zeros_like(dv_dy)
    
    # Actually getting the signal to noise ratio !
    
    
    
    
    # pad to have good shapes
    #du_dx = np.pad(du_dx, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    #du_dy = np.pad(du_dy, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    #dv_dx = np.pad(dv_dx, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    #dv_dy = np.pad(dv_dy, ((0, 1), (0, 0)), mode='constant', constant_values=0) 
    
    np.random.shuffle(du_dx)
    np.random.shuffle(du_dy)
    np.random.shuffle(dv_dx)
    np.random.shuffle(dv_dy)
    
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
        
        # set the noise to zero (in the gradients)
        #du_dx_noise = np.ones_like(du_dx)*1e-7
        #du_dy_noise = np.ones_like(du_dy)*1e-7
        #dv_dx_noise = np.ones_like(dv_dx)*1e-7
        #dv_dy_noise = np.ones_like(dv_dy)*1e-7
        du_dx_noise = np.zeros_like(du_dx)*1e-7
        du_dy_noise = np.zeros_like(du_dy)*1e-7
        dv_dx_noise = np.zeros_like(dv_dx)*1e-7
        dv_dy_noise = np.zeros_like(dv_dy)*1e-7
        
        
        
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
        
        
        # ONLY DIV
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
            
        #du_dy = np.zeros_like(du_dy)#!!!!!!!!!!!
        #dv_dx = np.zeros_like(dv_dx) #!!!!!!!!!!!
    
        
        
        #du_dx = du_dx2 + du_dx3
        #dv_dy = dv_dy2 + dv_dy3
        #du_dx = np.zeros_like(du_dx)
        #dv_dy = np.zeros_like(dv_dy)
        
        
        div = du_dx + dv_dy
        #sh = du_dy - dv_dx
        sh = np.sqrt((du_dx - dv_dy)**2 + (du_dy + dv_dx)**2)
        #div[np.abs(div) < 5e-3] = np.nan
        #sh[np.abs(sh) < 5e-3] = np.nan
        
        defo = np.sqrt(div**2 + sh**2)

        
        #PDF and find threshold
        div_max = np.nanmax(abs(defo))
        div_max = 0.1
        #print("MAXXX", div_max)
        threshold = 0.3* div_max
        #print("threshold", threshold)
        
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
            
            
            #du_dx = np.where(abs(du_dx) >= threshold, div_max, 0)
            du_dx = np.where(np.isnan(du_dx), np.nan, np.where(abs(du_dx) >= threshold, div_max, 0))
            #du_dy = np.zeros_like(du_dy)
            #dv_dx = np.zeros_like(dv_dx)
            #du_dy = np.where(abs(du_dy) >= threshold, div_max, 0)
            #dv_dx = np.where(abs(dv_dx) >= threshold, div_max, 0)
            #dv_dy = np.where(abs(dv_dy) >= threshold, div_max, 0)
            dv_dy = np.where(np.isnan(dv_dy), np.nan, np.where(abs(dv_dy) >= threshold, div_max, 0))
            #dv_dy = np.zeros_like(dv_dy)
        
            div = du_dx + dv_dy
            defo = div
            
            du_dy = np.zeros_like(du_dy)
            dv_dx = np.zeros_like(dv_dx)
        
        if rgps == "div" or rgps == 'div_abs' or rgps == "div_pos" or rgps == "div_neg":
            defo = div
            du_dy = np.zeros_like(du_dy)
            dv_dx = np.zeros_like(dv_dx)
        

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
            
            #plt.imshow(defo[30:-30, 10:-50], cmap="coolwarm",vmin=-0.1, vmax=0.1) 
            #plt.imshow(defo[30:-30, 10:-50], cmap="coolwarm",vmin=-div_max, vmax=div_max) 
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
                #x.set_title("RGPS div", fontweight ="extra bold", color='k', fontsize=18, family='sans-serif')
                ax.set_title("RGPS div", fontweight ="extra bold", color=color_th, fontsize=18, family='sans-serif')
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
            divergence = du_dx_moy + dv_dy_moy
            
            #print("du_dx_moy", du_dx_moy)
            #print("du_dy_moy", du_dy_moy)
            #print("dv_dx_moy", dv_dx_moy)
            #print("dv_dy_moy", dv_dy_moy)
            
            du_dx_moy_noise = du_dx_noise
            du_dy_moy_noise = du_dy_noise
            dv_dx_moy_noise = dv_dx_noise
            dv_dy_moy_noise = dv_dy_noise
            divergence_noise = du_dx_moy_noise + dv_dy_moy_noise
                    
            if c == 'rgps':
                shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
                shear_noise = np.sqrt((du_dx_moy_noise - dv_dy_moy_noise)**2 + (du_dy_moy_noise + dv_dx_moy_noise)**2)
            else : 
                #shear = np.sqrt((du_dx_moy - dv_dy_moy)**2)
                shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
                shear_noise = np.sqrt((du_dx_moy_noise - dv_dy_moy_noise)**2 + (du_dy_moy_noise + dv_dx_moy_noise)**2)
                                   
            #shear = du_dy_moy + dv_dx_moy
            deformation = np.sqrt(divergence**2 + shear**2)
            deformation_noise = np.sqrt(divergence_noise**2 + shear_noise**2)
            
            #print("defo", deformation)
            #print("defomation_noise", deformation_noise)
            #print('defo/noise', deformation/deformation_noise)
            #coarse_defos.append(deformation) # GOOD ONE FOR NO WEIGHTING
            
            if c == "weighted":
                #deformation_noise = (np.random.randn(np.shape(deformation))*0.01)
                coarse_defos.append(deformation) # FOR WEIGHTING
                
            else: 
                coarse_defos.append(deformation) # FOR WEIGHTING
                
            coarse_defos_noise.append(deformation/deformation_noise) # FOR WEIGHTING
                 
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
                    
                        
                    # New weighting 
                    #du_dx_moy = np.nanmean(block_du_dx*block_du_dx_noise)
                    #du_dy_moy = np.nanmean(block_du_dy*block_du_dy_noise)
                    #dv_dx_moy = np.nanmean(block_dv_dx*block_dv_dx_noise)
                    #dv_dy_moy = np.nanmean(block_dv_dy*block_dv_dy_noise)
                
                    #du_dx_moy = np.nanmean(abs(block_du_dx))
                    #du_dy_moy = np.nanmean(abs(block_du_dy))
                    #dv_dx_moy = np.nanmean(abs(block_dv_dx))
                    #dv_dy_moy = np.nanmean(abs(block_dv_dy))
                
                    divergence = du_dx_moy + dv_dy_moy
                    divergence_noise = du_dx_moy_noise + dv_dy_moy_noise
                    #shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
                    #shear = du_dy_moy + dv_dx_moy
                    shear = np.sqrt((du_dx_moy - dv_dy_moy)**2)
                    real_shear = np.sqrt((du_dx_moy - dv_dy_moy)**2 + (du_dy_moy + dv_dx_moy)**2)
                    real_shear_noise = np.sqrt((du_dx_moy_noise - dv_dy_moy_noise)**2 + (du_dy_moy_noise + dv_dx_moy_noise)**2)
                    
                    if np.shape(block_du_dx)!= np.shape(block_dv_dy):
                        divergence_bool, shear_bool = 0, 0
                    
                    else:
                        divergence_bool = block_du_dx + block_dv_dy
                        real_shear_bool = np.sqrt((block_du_dx - block_dv_dy)**2 + (block_du_dy + block_dv_dx)**2) #good one
                        shear_bool = np.sqrt((block_du_dx - block_dv_dy)**2) # use for div only
                
                    if c == 'c0':
                        #deformation = divergence
                        
                        #deformation = np.sqrt(divergence**2 + shear**2) # for only div noise!!!
                        #deformation_bool = np.sqrt(divergence_bool**2 + shear_bool**2)
                        
                        deformation = np.sqrt(divergence**2 + real_shear**2)
                        deformation_bool = np.sqrt(divergence_bool**2 + real_shear_bool**2)
                        
                        deformation_noise = np.sqrt(divergence_noise**2 + real_shear_noise**2)
                        
                        #deformation = np.sqrt(divergence**2)
                        #deformation = np.sqrt(shear**2)
                        
                    elif c == 'rgps':
                        deformation = np.sqrt(divergence**2 + real_shear**2)
                        deformation_bool = np.sqrt(divergence_bool**2 + real_shear_bool**2)
                        
                        deformation_noise = np.sqrt(divergence_noise**2 + real_shear_noise**2)
                        
                    else :
                        #deformation = np.sqrt(divergence**2 + shear**2) # for only div noise!!!
                        #deformation_bool = np.sqrt(divergence_bool**2 + shear_bool**2)
                        
                        deformation = np.sqrt(divergence**2 + real_shear**2)
                        deformation_bool = np.sqrt(divergence_bool**2 + real_shear_bool**2)
                        
                        deformation_noise = np.sqrt(divergence_noise**2 + real_shear_noise**2)
                        
                        #deformation = np.sqrt(divergence**2)
                        #deformation = np.sqrt(shear**2)
                
                    bool_defos.append(np.sum(~np.isnan(deformation_bool))) #count the non nan values
                    bool_shape.append(deformation_bool.size)
                    #coarse_defos.append(deformation) # GOOD ONE FOR NO WEIGHTING
                    
                    # For the weighting by the SNR
                    #coarse_defos.append(deformation)
                    
                    if c == 'weighted':
                        #coarse_defos.append(deformation*deformation_noise)
                        coarse_defos.append(deformation)
                
                        #deformation_noise = (np.random.randn(np.shape(deformation))*0.01)
                        
                    else:
                        coarse_defos.append(deformation)
                        
                    coarse_defos_noise.append(deformation/deformation_noise)
        
        #print("SHAPE",np.shape(coarse_defos))
            coarse_defos = np.where(
                    np.array(bool_defos) < L** 2 // 2, np.nan, np.array(coarse_defos),
                )  
            coarse_defos_noise = np.where(
                    np.array(bool_defos) < L** 2 // 2, np.nan, np.array(coarse_defos_noise),
                )  
    
        #coarse_defos = np.where(
        #        np.array(bool_defos) < 1, np.nan, np.array(coarse_defos),
        #    )      
        #print("SHAPE2",np.shape(coarse_defos))
        #print(bool_defos)
        #print(bool_shape)
        
        #print(coarse_defos_noise)
        #print("omax", coarse_defos) 
        #print('noise', np.array(coarse_defos)/np.array(coarse_defos_noise))
        
        if c == "weighted":
            #print("weighting")
            #deformations_L.append(np.nansum(coarse_defos)/np.nansum(coarse_defos_noise))
            x = L + np.random.uniform(-0.3, 0.3, size=np.array(coarse_defos).shape)
            sc = ax.scatter(x, coarse_defos, c=coarse_defos_noise,alpha=0.7, cmap=cmap, norm=LogNorm(vmin=1, vmax = 1000), s=6)
            #print('max', max(coarse_defos_noise))
            #print('mean', np.mean(coarse_defos_noise))
            
            
            #a = np.nansum(np.array(coarse_defos)*np.array(coarse_defos_noise))/np.nansum(coarse_defos_noise)
            mask = ~np.isnan(coarse_defos) & ~np.isnan(coarse_defos_noise)
            a= np.sum(np.array(coarse_defos)[mask] * np.array(coarse_defos_noise)[mask]) / np.sum(np.array(coarse_defos_noise)[mask])
            deformations_L.append(a)
        else:
            deformations_L.append(np.nanmean(coarse_defos))
        
        deformations_Long.append((coarse_defos))

        #print(10/L)
        #print(np.nanmean(coarse_defos))
        #print((np.nanmean(coarse_defos))/(10/L))
        #deformations_L.append(np.nanmean(coarse_defos)/(10*0.1/L))
    
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


def scaling_parameters(deformations_L, L_values):

    """
    Function to find scaling exponent (beta), the intercept and the r^2
    """
    
    log_L_values = np.log(L_values)
    log_deformations = np.log(deformations_L)
    slope, intercept, r_value, _, _ = linregress(log_L_values, log_deformations)
    r_squared = r_value**2
    
    return(intercept, slope, r_squared)

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
        #nan_where = np.argwhere(np.isnan(log_def))
        #log_def = np.delete(log_def, nan_where)
        #log_L = np.delete(log_L, nan_where)
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
    
    #dy = np.gradient(log_def_smooth)
    dy = log_def[:-1] - log_def[1:]
    ddy = dy[:-1] - dy[1:]
    #ddy = np.gradient(dy)
    
    print("DDYYYYY", ddy)
    #peaks, _ = find_peaks(np.abs(ddy), height=0.15)  # adjust threshold (height) as needed
    peaks = np.where((np.abs(ddy) > np.mean(np.abs(ddy))) & (np.abs(ddy) > np.std(np.abs(ddy))))[0]
    peaks = peaks+1
    print(peaks)
    # Add endpoints for segmentation
    breakpoints = [0] + peaks.tolist() + [len(log_L)-1]
    breakpoints = sorted(list(set(breakpoints)))
    print(breakpoints)
    
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
    
    
    """
    
    signal = np.column_stack((log_L, log_def))
    
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
    
    # Check for flat signal
    print(log_def)
    if np.allclose(log_def, log_def[1], atol=1e-2):
        print("Signal too flat — returning full regression over one segment.")
        slope, intercept, r, _, _ = linregress(log_L, log_def)
        return [{
            "start": 0,
            "end": len(log_L),
            "intercept": intercept,
            "slope": slope,
            "r_squared": r**2,
            "x_range": (L_values[0], L_values[-1])
        }]

    # Detect inflexion points in scaling
    max_possible_breaks = (len(signal) - 1) // 2  # ensures at least 2 pts/segment
    if max_possible_breaks < 1:
        print("Too few points to segment — returning full regression.")
        slope, intercept, r, _, _ = linregress(log_L, log_def)
        return [{
            "start": 0,
            "end": len(log_L),
            "intercept": intercept,
            "slope": slope,
            "r_squared": r**2,
            "x_range": (L_values[0], L_values[-1])
        }]

    max_breaks = min(max_breaks, max_possible_breaks)
    print(f"Trying Dynp with n={len(signal)}, n_bkps={max_breaks}")
    cost = CostSlope().fit(signal)   # fit your custom cost function on the signal
    algo = Pelt(custom_cost=True, min_size=1).fit(signal)  # tell Pelt to use a custom cost
    algo.cost = cost         
    algo = rpt.Pelt(model="l1").fit(signal[:, 1])
    breakpoints = algo.predict(pen=10)
    #algo = rpt.KernelCPD(kernel='linear').fit(signal)
    #breakpoints = algo.predict(n_bkps=max_breaks)
    #breakpoints = algo.predict(pen=3)
    
    
    segment = []
    start = 0
    for end in breakpoints:
        x = log_L[start:end]
        y = log_def[start:end]
        slope, intercept, r, _, _ = linregress(x,y)
        r_squared = r**2
        segment.append({
            "start": start,
            "end": end,
            "intercept": intercept,
            "slope": slope,
            "r_squared": r_squared,
            "x_range": (L_values[start], L_values[end-1])
        })
        start = end
        """ 
    return segment
    
    
def scaling_figure(deformations, L_values, segments, names, colors, markers, r_threshold=0.3, linestyle="-"):
#def scaling_figure(deformations, L_values, intercepts, slopes, r2s, names, colors, markers, r_threshold=0.3, linestyle="-"):
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
                """ for the single slope (scaling_parameters fcn)
                if r2s[i] > r_threshold:
                    #ax.plot(np.array(L_values)*10, np.exp(intercepts[i]) * L_values**-slopes[i], c=colors[i], linewidth=1.5,linestyle='--', zorder=500)
                    ax.plot(np.array(L_values)*10, np.exp(intercepts[i]) * L_values**-slopes[i], c=colors[i], linewidth=1.5,linestyle=':', zorder=500)
                """
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
                    #ax.scatter(np.array(L_values)*10, deformations[i], c=colors[i], marker=markers[i], s=50, alpha=1, edgecolors=colors[i], zorder=1000)
                    """
                    if r2s[i] > r_threshold:
                        ax.plot(np.array(L_values)*10, np.exp(intercepts[i]) * L_values**-slopes[i], c=colors[i], linewidth=1.5,linestyle=':', zorder=500)
                    """
                    # for the multiple slopes (scaling_sgments fcn)
                    experiment_segments = segments[i]
                    for seg in experiment_segments:
                        L_fit = np.linspace(seg["x_range"][0], seg["x_range"][1], 100)
                        fit = np.exp(seg["intercept"]) * L_fit**seg["slope"]
                        ax.plot(L_fit * 10, fit, c=colors[i], linewidth=1.5, linestyle=':', zorder=500)
                else:
                    # Scatter plot and regression line
                    ax.scatter(np.array(L_values)*10, deformations[i], c=colors[i], marker=markers[i], s=60, alpha=1, edgecolors="k", zorder=1000)
                    #if r2s[i] > r_threshold or (slopes[i] > -0.1 and slopes[i] < 0.1):
                    #    ax.plot(np.array(L_values)*10, np.exp(intercepts[i]) * L_values**-slopes[i], c=colors[i], linewidth=1.5,linestyle=linestyle, zorder=500)
                    # for the multiple slopes (scaling_sgments fcn)
                    experiment_segments = segments[i]
                    for seg in experiment_segments:
                        L_fit = np.linspace(seg["x_range"][0], seg["x_range"][1], 100)
                        fit = np.exp(seg["intercept"]) * L_fit**seg["slope"]
                        ax.plot(L_fit * 10, fit, c=colors[i], linewidth=1.5, linestyle=linestyle, zorder=500)
             
            legend_elements.append((names[i],colors[i], markers[i]))       
            #slopes_print = -1*slopes
            # Add slope value to the legend
            #if r2s[i] > r_threshold and r2s[i]<1:
            #    print("ICI", r2s[i])
            #    #legend_elements.append((names[i] + f': {slopes[i]:.2f}',colors[i]))
            #    if slopes[i]<0.01:
            #        #legend_elements.append((f'{abs(slopes[i]):.2f}',colors[i]))
            #        legend_elements.append((names[i],colors[i], markers[i]))
            #    else:
            #        #legend_elements.append((f'{slopes[i]:.2f}',colors[i]))
            #        legend_elements.append((names[i],colors[i], markers[i]))
                    
            #else:
                #legend_elements.append((names[i],colors[i]))
                #legend_elements.append(("-",colors[i]))
            #    legend_elements.append((names[i],colors[i], markers[i]))
                
            #legend_elements.append((f'{slopes[i]:.2f}',colors[i]))


        #slope = -np.log(0.009/0.05)/np.log(640/20)
        #ax.plot(np.array([L_values[0],L_values[-1]])*10, np.array([0.05,0.009]), c='k', linestyle='--')
        #legend_elements.append((f'{slope:.2f}',"k"))
        # Custom legend with only colored numbers
        legend_labels = [f'{text}' for text, _,_ in legend_elements]
        legend_colors = [color for _, color, _ in legend_elements]
        legend_markers = [marker for _,_, marker in legend_elements]
        #legend_title = '$\\beta$'
        #legend_title = "Experiment"
        legend_title = " "

        """
        # Add text outside the plot as the legend
        #pos_x = 1.27
        pos_x = 1.2
        #pos_x = 1.4
        #pos_x = 1.25
        ax.text(pos_x, 0.95, legend_title, transform=ax.transAxes, fontsize=size_text, ha='left', va='center', fontweight='1000')
        for i, (label, color, marker) in enumerate(zip(legend_labels, legend_colors, legend_markers)):
            y = 0.85 - (i + 1.05) * 0.07
            ax.text(pos_x, y, label, transform=ax.transAxes, fontsize=size_text-5, ha='center', va='center', color=color, weight='bold', family='sans-serif')
            ax.scatter([pos_x - 0.35], [y], transform=ax.transAxes, marker=marker, edgecolors='k',
               color=color, s=40, clip_on=False)
        """
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

        #ax.set_ylim([1e-1, 1e0])
        #ax.set_ylim([5e-1, 5e0])
        
        #ax.set_ylim([1e-3, 1e0]) # good one !
        #ax.set_ylim([1e-4, 1e0])
        #ax.set_ylim([5e-3, 1e0]) # good one !
        
        #ax.set_ylim([1e-3, 1e-1])
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
