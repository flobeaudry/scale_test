import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from experiments.define_experiments import get_experiment
from processing.deformation_to_velocity import compute_velocity_fields
from analysis.scaling_analysis import scale_and_coarse, scaling_parameters, scaling_figure
from utils.figures_gen import fig_velocity_defo, fig_defo

# Define experiments
experiment_names = [
    
    #"control",
    #"irregular spacing",
    ##"control",
    #"narrow spacing",
    #"irregular intensity",
    #"irregular domain",    
    #"errors", 
    
    "control +-",
    "irregular spacing +-",
    #"control +-",
    "narrow spacing +-",
    "irregular intensity +-",
    "irregular domain +-",    
    "errors +-",
    
    #"DIV+_oneline",
    
    #"DIV+constant",
    #"DIV+increase",
    
    #"DIV+",
    #"DIV+45",
    #"DIV+density",
    #"DIV+frequency",
    #"DIV+intensity",
    #"DIV+domain",
    #"DIV+errors",
    
    #"errors",
    #"errors_c1",
    #"errors_randn",
    #"errors_speckle",
    
    #"DIV+-",
    #"DIV+-45",
    #"DIV+-density",
    ##"DIV+-density_c1",
    #"DIV+-frequency",
    ##"DIV+-frequency_c1",
    #"DIV+-intensity",
    ##"DIV+-intensity_c1",
    #"DIV+-domain",
    #"DIV+-errors",
    
    
    #"DIVs",
    ##"DIVs45",
    #"DIVsfrequency",
    #"DIVsintensity",
    #"DIVsdomain",
    #"DIVserrors",
    
    
    #"DIV+RAMPlin",
    #"DIV+-RAMPlin",
    #"DIV+RAMPsin",
    #"DIV+-RAMPsin",
    
    #"DIV+-RAMPsin_pino2",
    
    
    ##"Divergence_control",
    ##"Divergence_control_div",
    ##"Divergence_conv_control",
    ##"Divergence_smallangle",
    #"Divergence_conv_control_4",
    #"Divergence_control_half",
    ##"Divergence_spectrum",
    ##"Divergence_spectrum_int",
    ##"Divergence_spectrum_full",
    ##"Divergence_spectrum_full_4",
    ##"Divergence_spectrum_full_int", # lines equally spaced, but the div/conv varies randomly with +1 and -1 values
    
    #"Divergence_random",
    #"Divergence_reversed",
    #"Divergence_uneven",
    #"Divergence_uneven_noise",
    #"Divergence_smallangle",
    #"Divergence_control_noise",
    #"Divergence_control_noise_plus",
    
    ###"Divergence_SNR_100",
    ###"Divergence_SNR_10",
    ###"Divergence_SNR_1",
    ###"Divergence_spectrum_full_4",
    ###"Divergence_DSC_100",
    ###"Divergence_DSC_10",
    ###"Divergence_DSC_1",
    
    #"Divergence_intensity",
    #"Divergence_width",
    #"Divergence_divergence",
    #"divergence_with_angle",
    
    
    #"Divergence_angle",
    #"Divergence_density",
    #"Divergence_intensity",
    #"Divergence1",
    #"Divergence2"
    #"Divergence1_1",
    #"Divergence1_2",
    #"Shear0",
    #"Shear1",
    #"Shear1_2",
    #"Shear1_1",
    #"DivShear0"
    ]

new_run = True
figs = True

#L_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
L_values = [2, 4, 8, 16, 32, 64]
dx, dy = 1, 1

if new_run == True:
    
    deformations_tot, intercepts, slopes, names, colors = [], [], [], [], []

    # Loop through experiments

    for name in experiment_names:
        print(f"Processing experiment: {name}")
    
        # Step 1: Fetch experiment
        experiment = get_experiment(name)
        F, exp_type, name, color = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"]

        # Step 2: Compute velocity fields
        u, v, F_recomp = compute_velocity_fields(F, exp_type, name, color)
        print("Velocity fields computed")
    
        # Step 2.2 : Velocity fields figures generation
        if figs == True:
            # Plot velocity fields and recomputed deformations
            fig_velocity_defo(u, v, F_recomp, name, color, top_right_quadrant = True)
            # Plot recomputed deformations
            fig_defo(u, v, F_recomp, name, color, top_right_quadrant = True)
            print("Deformation figures created")


        # Step 3: Perform scaling analysis
        #deformations_L, deformations_Long = scale_and_coarse(u, v, L_values, dx=dx, dy=dy)
        
        if name.endswith("c1"):
            deformations_L, deformations_Long = scale_and_coarse(u, v, L_values, dx=dx, dy=dy, c='c1')
        else:
            deformations_L, deformations_Long = scale_and_coarse(u, v, L_values, dx=dx, dy=dy)

        max_len = max(len(d) for d in deformations_Long)  # Find longest sublist
        deformations_Long_padded = [d + [np.nan] * (max_len - len(d)) for d in deformations_Long]
        deformations_Long = np.array(deformations_Long_padded)
        
        deformations_tot.append(deformations_L)
        print("Scaling analysis done")
        print(color)

        plt.rcParams.update({'font.size': 16})
        with plt.style.context(['science', 'no-latex']):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.grid(True, which='both')
            for i in range(len(L_values)):  
                #x_vals = np.linspace(L_values[i] - 1, L_values[i] + 1, len(deformations_Long[i]))
                x_vals = np.linspace(L_values[i] , L_values[i] , len(deformations_Long[i]))
                ax.scatter(x_vals, deformations_Long[i], color=color, s=60, alpha=1, edgecolors="k", zorder=1000)
        
            ax.set_xlabel('Spatial scale (nu)')
            ax.set_ylabel('$\\epsilon_{tot}$')
            ax.set_xscale("log")
            #plt.yscale("log")
            #plt.ylim( -5,1)
            ax.set_title(f"{name}", fontweight ="extra bold", color=color,  family='sans-serif')
            combined_filename = os.path.join("SIMS/project/figures", f"SCATTER_{name}_scaling.png")
            plt.savefig(combined_filename, dpi=300)

        # Step 4: Generate scaling figure
        intercept, slope = scaling_parameters(deformations_L, L_values)
        intercepts.append(intercept)
        slopes.append(-1*slope)
        names.append(experiment["name"])
        colors.append(experiment["color"])
        #print(slope)
    
        print(f"Finished processing: {name}\n")

    # **Save results to a file for future use**
    results = {
        "deformations_tot": deformations_tot,
        "intercepts": intercepts,
        "slopes": slopes,
        "names": names,
        "colors": colors,
    }

    with open("scaling_results.pkl", "wb") as f:
        pickle.dump(results, f)
        
    print("Results saved successfully")
  
else:
    print('oups')
    
    try:
        with open("scaling_results.pkl", "rb") as f:
            results = pickle.load(f)
            deformations_tot = results["deformations_tot"]
            intercepts = results["intercepts"]
            slopes = results["slopes"]
            names = results["names"]
            colors = results["colors"]
        print("Loaded precomputed results.")
        
    except FileNotFoundError:
        print("No precomputed results found. Set `new_run = True` to generate results.")
 

scaling_figure(deformations_tot, L_values, intercepts, slopes, names, colors)
print("Scaling figure created")
    
    