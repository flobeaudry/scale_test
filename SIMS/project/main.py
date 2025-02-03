import pickle
from experiments.define_experiments import get_experiment
from processing.deformation_to_velocity import compute_velocity_fields
from analysis.scaling_analysis import scale_and_coarse, scaling_parameters, scaling_figure
from utils.figures_gen import fig_velocity_defo, fig_defo

# Define experiments
experiment_names = [
    ##"Divergence_control",
    ##"Divergence_control_div",
    ##"Divergence_conv_control",
    ##"Divergence_smallangle",
    #"Divergence_conv_control_4",
    #"Divergence_control_half",
    ##"Divergence_spectrum",
    ##"Divergence_spectrum_int",
    ##"Divergence_spectrum_full",
    "Divergence_spectrum_full_4",
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
            fig_velocity_defo(u, v, F_recomp, name, color)
            # Plot recomputed deformations
            fig_defo(u, v, F_recomp, name, color)
            print("Deformation figures created")

        # Step 3: Perform scaling analysis
        deformations_L = scale_and_coarse(u, v, L_values, dx=dx, dy=dy)
        deformations_tot.append(deformations_L)
        print("Scaling analysis done")

        # Step 4: Generate scaling figure
        intercept, slope = scaling_parameters(deformations_L, L_values)
        intercepts.append(intercept)
        slopes.append(-1*slope)
        names.append(experiment["name"])
        colors.append(experiment["color"])
        print(slope)
    
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
    
    