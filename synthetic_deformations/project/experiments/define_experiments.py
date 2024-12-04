import numpy as np

def get_experiment(name):
    N = 80  # Grid size
    dx, dy = 1, 1 # Grid resolution
    mean, std = 0, 0.1
    
    if name == "Shear0":
        F_shear_u = np.zeros((N, N))
        F_shear_u[:, 14:16] = -2
        F_shear_u[:, :1] = 1
        F_shear_u[:, -1:] = 1
        F_shear_v = np.zeros((N, N))
        F = np.vstack([F_shear_v, F_shear_u])
        return {"F": F, "exp_type": "shear", "name": "Shear0", "color": "tab:orange"}
    
    if name == "Divergence0":
        F_div_u = np.zeros((N, N))
        F_div_u[:, 14:16] = -2
        F_div_u[:, :1] = 1
        F_div_u[:, -1:] = 1
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Divergence0", "color": "tab:blue"}

    if name == "DivShear0":
        F_shear_u = np.zeros((N, N))
        F_shear_u[:, 14:16] = -2
        F_shear_u[:, :1] = 1
        F_shear_u[:, -1:] = 1
        F_shear_v = np.zeros((N, N))
        F_shear = np.vstack([F_shear_v, F_shear_u])
        
        F_div_u = np.zeros((N, N))
        F_div_u[:, 14:16] = -2
        F_div_u[:, :1] = 1
        F_div_u[:, -1:] = 1
        F_div_v = np.zeros((N, N))
        F_div = np.vstack([F_div_u, F_div_v])
        
        F = F_shear + F_div
        return {"F": F, "exp_type": "both", "name": "DivShear0", "color": "tab:green"}
    
    # Add more experiments as needed
    raise ValueError(f"Experiment '{name}' not found.")