import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from experiments.define_experiments import get_experiment
from processing.deformation_to_velocity import compute_velocity_fields
from analysis.scaling_analysis import scale_and_coarse, scaling_segments, scaling_figure, scaling_figure_new
from utils.figures_gen import fig_velocity_defo, fig_defo_new, fig_defo_gradients

# Define experiments
experiment_names = [
    
    "axial_strain",
    "pure_shear_strain",

    #"control",
    #"narrow spacing",
    #"exp",
    #"irregular spacing",
    #"irregular intensity",
    #"irregular domain",
    #"errors",
    
    #"control_div_shear",
    #"45_angle",
     
    #"control_diamonds_angled",
    #"control_diamonds",
    
    #"control err", #########
    #"control err weighted",
    #"control err weighted2",
    
    #"control err",
    #"narrow spacing err",
    #"irregular spacing err",
    #"irregular intensity err",

    
    
    #"control +-", #yes
    #"narrow spacing +-", #yes
    #"exp +-", #naur
    #"irregular spacing +-", #yes
    #"irregular intensity +-", #yes ######
    #"errors +-", # later
    #"irregular domain +-",
    
    #"control +- err",
    #"narrow spacing +- err",
    #"narrow spacing +- err weighted",
    #"irregular spacing +- err",
    #"irregular intensity +- err", #########
    #"irregular intensity +- err weighted",
    #"irregular intensity +- err weighted2",
    
    
    #"irregular spacing",
    ##"control",
    #"narrow spacing",
    #"irregular intensity",
    #"irregular domain",    
    #"errors", 
    
    #"sin+",
    #"exp",
    #"exp +-",
    

    #"sin01",
    #"ksin01",

    #"sin-0.51",
    #"ksin-0.51",
    #"ksin-11", #this one !!
    #"sin-11", #this one !!
    
    
    #"sin01 err",
    #"ksin01 err",

    #"sin-0.51 err",
    #"ksin-0.51 err",
    #"sin-11 err",
    #"ksin-11 err",
    
    #"err",
    #"err +-",
    
    #'control_decay',
    #"cantor",
    #"fractal_tree",
    #"fractal_tree_abs",
    #"radial_tree",
    #"sierpinski",
    #"koch",
    #"fractal_shuffle",
    #"fractal+error",
    #"weierstrass",
    #"weierstrass_pos",
    
    #"fractalline01",
    #"fractalline11",
    #"fractalline10",
    
    #"control +-",
    #"irregular spacing +-",
    ##"control +-",
    #"narrow spacing +-",
    #"irregular intensity +-",
    #"irregular domain +-",    
    #"errors +-",
    #"irregular intensity errors +-",
    #"exp +-",
    #"lin",
    
    #"onlyerrors",
    #"onlyerrors_speckle",
    #"errors_speckle",
    
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
#new_run = "vel_gradients"
#new_run = "RGPS"
#new_run = "RGPS_threshold"
save_exp = False # Do you want to save the experiment you are running ?
figs = True # For deformation figures (i.e. seeing the deformations)

max_breaks  = 2
RGPS_scaling = True

#L_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
L_values = [1,2, 4, 8, 16, 32, 64, 128, 256, 512]
#L_values = [1, 2, 4, 8, 16, 32]
dx, dy = 1, 1
#dx, dy = 0.1, 0.1

if new_run == True:
     
    #deformations_tot, intercepts, slopes, r2s, names, colors, markers = [], [], [], [], [], [], []
    deformations_tot, segments, names, colors, markers = [], [], [], [], []

    # Loop through experiments

    for name in experiment_names:
        print(f"Processing experiment: {name}")
    
        # Step 1: Fetch experiment
        experiment = get_experiment(name)
        F, exp_type, name, color, marker, scaling_on = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"], experiment["marker"], experiment["scaling_on"]

        # Step 2: Compute velocity fields
        u, v, F_recomp, u_noise, v_noise = compute_velocity_fields(F, exp_type, name, color)
        print("Velocity fields computed")
    
        # Step 2.2 : Velocity fields figures generation
        if figs == True:
            # Plot velocity fields and recomputed deformations
            fig_velocity_defo(u, v, F_recomp, name, color, top_right_quadrant = True)
            # Plot recomputed deformations

            fig_defo_new(u, v, F_recomp, name, color, top_right_quadrant = False)
            print("Deformation figures created")


        # Step 3: Perform scaling analysis
        #deformations_L, deformations_Long = scale_and_coarse(u, v, L_values, dx=dx, dy=dy)
        
        if name.endswith("c1"):
            deformations_L, deformations_Long = scale_and_coarse(u, v, F, u_noise, v_noise, L_values, dx=dx, dy=dy, c='c1',  scaling_on = scaling_on )
            
        elif name.endswith("err"):
            deformations_L, deformations_Long = scale_and_coarse(u, v, F, u_noise, v_noise, L_values, dx=dx, dy=dy, c='err',  scaling_on = scaling_on )
            
        elif name.endswith("weighted"):
            deformations_L, deformations_Long = scale_and_coarse(u, v, F,  u_noise, v_noise, L_values, dx=dx, dy=dy, c='weighted', scaling_on = scaling_on )
        elif name.endswith("weighted2"):
            deformations_L, deformations_Long = scale_and_coarse(u, v, F, u_noise, v_noise, L_values, dx=dx, dy=dy, c='weighted2', scaling_on = scaling_on )
            
        else:
            deformations_L, deformations_Long = scale_and_coarse(u, v, F, u_noise, v_noise, L_values, dx=dx, dy=dy, scaling_on = scaling_on )

        deformations_tot.append(deformations_L)
        print("Scaling analysis done")
        print(color)

       
        
        # Step 4: Generate scaling figure
        segment = scaling_segments(deformations_L, L_values, max_breaks = max_breaks)
        segments.append(segment)
        names.append(experiment["name"])
        colors.append(experiment["color"])
        markers.append(experiment["marker"])
    
        print(f"Finished processing: {name}\n")
        
        if save_exp == True:
            results = {
                "deformations_L": deformations_L,
                "segment": segment,
                "name": name,
                "color": color,
                "marker": marker,
            }

            safe_name = name.strip().replace(" ", "").replace(":", "").replace("/", "")
            with open(f"SIMS/project/results/scaling_{safe_name}.pkl", "wb") as f:
                pickle.dump(results, f)
        
            print("Results saved successfully")
        
        
    if RGPS_scaling == True:    
        
        # RGPS
        experiment = get_experiment("control")
        F, exp_type, name, color, marker = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"], experiment["marker"]
        # Step 2: Compute velocity fields
        u, v, F_recomp, u_noise, v_noise = compute_velocity_fields(F, exp_type, name, color)

        #name, color =  "RGPS", "black"
        name, color =  "RGPS", "grey"
        #deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', scaling_on = "du_dx")
        #deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', scaling_on = "shuffle")
        deformations_L, deformations_Long = scale_and_coarse(u, v, F, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', scaling_on = "all")
        
        deformations_tot.append(deformations_L)
        # Step 4: Generate scaling figure
        segment = scaling_segments(deformations_L, L_values, name = "rgps")
        segments.append(segment)
        names.append(name)
        colors.append(color)
        markers.append(marker)
    
        results = {
            "deformations_L": deformations_L,
            "segment": segment,
            "name": name,
            "color": color,
            "marker": marker,
        }

        safe_name = name.strip().replace(" ", "").replace(":", "").replace("/", "")
        with open(f"SIMS/project/results/scaling_{safe_name}.pkl", "wb") as f:
            print("DUMPPP")
            pickle.dump(results, f)
    
    # SIM test Antoine
    #name, color =  "SIM", "lightseagreen"
    #deformations_L, deformations_Long = scale_and_coarse(u, v, L_values, dx=dx, dy=dy, c='sim')
    #deformations_tot.append(deformations_L)
    #print("Scaling analysis done")
    #print(color)
    #intercept, slope, r2 = scaling_parameters(deformations_L, L_values)
    #intercepts.append(intercept)
    #slopes.append(-1*slope)
    #r2s.append(r2)
    #names.append(name)
    #colors.append(color)

    print(f"Finished processing: {name}\n")

    with open("scaling_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("Results saved successfully")
    
    
    
elif new_run == "RGPS":
    deformations_tot, segments, names, colors, markers = [], [], [], [], []
        
    # RGPS
    experiment = get_experiment("control")
    F, exp_type, name, color, marker = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"], experiment["marker"]
    u, v, F_recomp, u_noise, v_noise = compute_velocity_fields(F, exp_type, name, color)
    
    # All RGPS cases...
    #scaling_on = "du_dx"
    scaling_on = "all"
    
    # ONLY DIVERGENCE
    name, color =  "div", "black"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="div", scaling_on = scaling_on)
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")
    
    # ABS DIVERGENCE
    name, color =  "|div|", "grey"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="div_abs", scaling_on = scaling_on)
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")
    
    # POS DIVERGENCE
    name, color =  "div>0", "red"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="div_pos", scaling_on = scaling_on)
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")

    '''
    # NEG DIVERGENCE
    name, color =  "div<0", "blue"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="div_neg", scaling_on = scaling_on)
    print("HEREEE")
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")
    '''
    
    print(f"Finished processing: {name}\n")
    
elif new_run == "RGPS_threshold":
    #deformations_tot, intercepts, slopes, r2s, names, colors, markers = [], [], [], [], [], [], []
    deformations_tot, segments, names, colors, markers = [], [], [], [], []
        
    # RGPS
    experiment = get_experiment("control")
    F, exp_type, name, color, marker = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"], experiment["marker"]
    u, v, F_recomp, u_noise, v_noise = compute_velocity_fields(F, exp_type, name, color)
    
    # All RGPS cases...
    
    # ONLY DIVERGENCE
    name, color =  "div", "black"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="div")
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")
    
    # ONLY DIVERGENCE
    name, color =  "|div|", "grey"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="div_abs")
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")
    
    
    # ABS DIVERGENCE
    name, color =  "|div| t1", "xkcd:greenish"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="t1")
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")
    
    # POS DIVERGENCE
    name, color =  "|div| t2", "xkcd:bluish purple"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="t2")
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")

    # NEG DIVERGENCE
    name, color =  "|div| t3", "xkcd:red orange"
    deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', rgps="t3")
    print("HEREEE")
    deformations_tot.append(deformations_L)
    segment = scaling_segments(deformations_L, L_values, name = "rgps")
    segments.append(segment)
    names.append(name)
    colors.append(color)
    markers.append("^")
    
    print(f"Finished processing: {name}\n")
    




elif new_run == "vel_gradients":
     
    #deformations_tot, intercepts, slopes, r2s, names, colors, markers = [], [], [], [], [], [], []
    deformations_tot, segments, names, colors, markers = [], [], [], [], []

    # Loop through experiments

    for name in experiment_names:
        print(f"Processing experiment: {name}")
    
        # Step 1: Fetch experiment
        experiment = get_experiment(name)
        F, exp_type, name, color, marker, scaling_on = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"], experiment["marker"], experiment["scaling_on"]

        du_dx, dv_dy, du_dy, dv_dx = np.split(F, 4, axis=0)
        print('DU_DX',np.shape(du_dx))
        print('DV_DY',np.shape(dv_dy))
        print('DU_DY',np.shape(du_dy))
        print('DV_DX',np.shape(dv_dx))
        fig_defo_gradients(du_dx, dv_dy, du_dy, dv_dx, name, color)
        
        # Step 3: Perform scaling analysis
        deformations_L, deformations_Long = scale_and_coarse(u=F, v=np.zeros((16,16)), u_noise=0, v_noise=0, L_values = L_values, dx=dx, dy=dy, c='vel_gradients',  scaling_on = scaling_on )
       
        deformations_tot.append(deformations_L)
        print("Scaling analysis done")
        print(color)
        
        # Step 4: Generate scaling figure
        segment = scaling_segments(deformations_L, L_values, max_breaks = max_breaks)
        segments.append(segment)
        names.append(experiment["name"])
        colors.append(experiment["color"])
        markers.append(experiment["marker"])
    
        print(f"Finished processing: {name}\n")
        
        if save_exp == True:
            results = {
                "deformations_L": deformations_L,
                "segment": segment,
                "name": name,
                "color": color,
                "marker": marker,
            }

            safe_name = name.strip().replace(" ", "").replace(":", "").replace("/", "")
            with open(f"SIMS/project/results/scaling_{safe_name}.pkl", "wb") as f:
                pickle.dump(results, f)
        
            print("Results saved successfully")
        
        
        
    if RGPS_scaling == True:    
        # RGPS
        experiment = get_experiment("control")
        F, exp_type, name, color, marker = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"], experiment["marker"]
        # Step 2: Compute velocity fields
        u, v, F_recomp, u_noise, v_noise = compute_velocity_fields(F, exp_type, name, color)

        #name, color =  "RGPS", "black"
        name, color =  "RGPS", "grey"
        deformations_L, deformations_Long = scale_and_coarse(u, v, u_noise, v_noise, L_values, dx=dx, dy=dy, c='rgps', scaling_on = "du_dx")

        deformations_tot.append(deformations_L)
        # Step 4: Generate scaling figure
        segment = scaling_segments(deformations_L, L_values, name = "rgps")
        segments.append(segment)
        names.append(name)
        colors.append(color)
        markers.append(marker)
    
        results = {
            "deformations_L": deformations_L,
            "segment": segment,
            "name": name,
            "color": color,
            "marker": marker,
        }

        safe_name = name.strip().replace(" ", "").replace(":", "").replace("/", "")
        with open(f"SIMS/project/results/scaling_{safe_name}.pkl", "wb") as f:
            pickle.dump(results, f)
    
    # SIM test Antoine
    #name, color =  "SIM", "lightseagreen"
    #deformations_L, deformations_Long = scale_and_coarse(u, v, L_values, dx=dx, dy=dy, c='sim')
    #deformations_tot.append(deformations_L)
    #print("Scaling analysis done")
    #print(color)
    #intercept, slope, r2 = scaling_parameters(deformations_L, L_values)
    #intercepts.append(intercept)
    #slopes.append(-1*slope)
    #r2s.append(r2)
    #names.append(name)
    #colors.append(color)

    print(f"Finished processing: {name}\n")

    #with open("scaling_results.pkl", "wb") as f:
    #    pickle.dump(results, f)
    
    print("Results saved successfully")





else:
    print('oups')
    
    names = []
    for name in experiment_names:
        print(f"Processing experiment: {name}")
    
        # Step 1: Fetch experiment
        experiment = get_experiment(name)
        F, exp_type, name, color, marker = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"], experiment["marker"]
        names.append(experiment["name"])
        
    print(names)
    
    try:
        results_folder = "SIMS/project/results"
        deformations_tot = []
        segments = []
        colors = []
        markers = []

        for name in names:
            safe_name = name.strip().replace(" ", "").replace(":", "").replace("/", "")
            filename = f"scaling_{safe_name}.pkl"
            filepath = os.path.join(results_folder, filename)

            with open(filepath, "rb") as f:
                result = pickle.load(f)
                deformations_tot.append(result["deformations_L"])
                segments.append(result["segment"])
                colors.append(result["color"])
                markers.append(result["marker"])
                
        filename = f"scaling_{"obs"}.pkl"
        filepath = os.path.join(results_folder, filename)
        with open(filepath, "rb") as f:
            result = pickle.load(f)
            deformations_tot.append(result["deformations_L"])
            segments.append(result["segment"])
            names.append(result["name"])
            colors.append(result["color"])
            markers.append(result["marker"])

        print("Loaded selected precomputed results.")
        print(deformations_tot)

    except FileNotFoundError as e:
        print(f"Could not find file: {e.filename}")
    


print(deformations_tot)

print(segments)
#scaling_figure(deformations_tot, L_values, segments, names, colors,markers, linestyle="-")
scaling_figure_new(deformations_tot, L_values, segments, names, colors,markers, linestyle="-")
print("Scaling figure created")
    
    