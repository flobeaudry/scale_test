from experiments.define_experiments import get_experiment
from processing.deformation_to_velocity import compute_velocity_fields
from analysis.scaling_analysis import scale_and_coarse, scaling_parameters, scaling_figure

# Define experiments
experiment_names = [
    "Divergence_control",
    #"Divergence_reversed",
    #"Divergence_uneven",
    #"Divergence_uneven_noise",
    #"Divergence_smallangle",
    #"Divergence_control_noise",
    #"Divergence_control_noise_plus",
    "Divergence_SNR_1",
    "Divergence_SNR_10",
    "Divergence_SNR_100",
    #"Divergence_intensity",
    #"Divergence_width",
    #"Divergence_divergence",
    
    
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

L_values = [2, 4, 8, 16, 32]
dx, dy = 1, 1

deformations_tot, intercepts, slopes, names, colors = [], [], [], [], []

# Loop through experiments
for name in experiment_names:
    print(f"Processing experiment: {name}")
    
    # Step 1: Fetch experiment
    experiment = get_experiment(name)
    F, exp_type, name, color = experiment["F"], experiment["exp_type"], experiment["name"], experiment["color"]

    # Step 2: Compute velocity fields
    u, v = compute_velocity_fields(F, exp_type, name, color)
    print("Velocity fields computed")

    # Step 3: Perform scaling analysis
    deformations_L = scale_and_coarse(u, v, L_values, dx=dx, dy=dy)
    deformations_tot.append(deformations_L)
    print("Scaling analysis done")

    # Step 4: Generate scaling figure
    intercept, slope = scaling_parameters(deformations_L, L_values)
    intercepts.append(intercept)
    slopes.append(slope)
    names.append(experiment["name"])
    colors.append(experiment["color"])
    
    print(f"Finished processing: {name}\n")
    

scaling_figure(deformations_tot, L_values, intercepts, slopes, names, colors)
    
    