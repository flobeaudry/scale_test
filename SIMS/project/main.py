from experiments.define_experiments import get_experiment
from processing.deformation_to_velocity import compute_velocity_fields
from analysis.scaling_analysis import scale_and_coarse, scaling_parameters, scaling_figure

# Define experiments
experiment_names = [
    "Divergence0",
    #"Divergence1",
    #"Shear0",
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
    u, v = compute_velocity_fields(F, exp_type, name)

    # Step 3: Perform scaling analysis
    deformations_L = scale_and_coarse(u, v, L_values, dx=dx, dy=dy)
    deformations_tot.append(deformations_L)

    # Step 4: Generate scaling figure
    intercept, slope = scaling_parameters(deformations_L, L_values)
    intercepts.append(intercept)
    slopes.append(slope)
    names.append(experiment["name"])
    colors.append(experiment["color"])
    
    print(f"Finished processing: {name}\n")
    

scaling_figure(deformations_tot, L_values, intercepts, slopes, names, colors)
    
    