import numpy as np
from scipy.ndimage import rotate

def get_experiment(name):
    #N = 1024 # Grid size
    N = 30
    dx, dy = 1, 1 # Grid resolution
    mean, std = 0, 0.1
    
    if name == "Divergence_control":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, int(2 * N / 8):int(3 * N / 8)] = -1
        #F_div_u[:, int(9 * N / 8):int(10 * N / 8)] = -1
        
        F_div_u[:, ::8] = -1 
        
        #F_div_u[:, int(5 * N / 16):int(6 * N / 16)] = 1
        #F_div_u[:, int(12 * N / 16):int(13 * N / 16)] = 1
        #F_div_u[:, 2:3] = -1
        #F_div_u[:, -2:-1] = -1
        #F_div_u[:, 5:6] = 1
        #F_div_u[:, 12:13] = 1
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div control", "color": "tab:blue"}
    
    if name == "Divergence_alternate":
        F_div_u = np.zeros((N, N))
        
        indices = np.arange(0, F_div_u.shape[1], 8)

        # Use a condition to alternate between -1 and +1
        F_div_u[:, indices[::2]] = 1      # Set +1 on even indices
        F_div_u[:, indices[1::2]] = -1     # Set -1 on odd indices
        
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div alternate", "color": "tab:red"}
    
    
    if name == "Divergence_random":
        F_div_u = np.zeros((N, N))
        
        num_changes = F_div_u.shape[1] // 8

        # Create the pattern (+1 and -1, alternating)
        pattern = np.ones(num_changes)  # Start with +1
        pattern[1::2] = -1  # Assign -1 to every second position

        # Shuffle the pattern to randomly distribute +1 and -1
        np.random.shuffle(pattern)

        # Randomly select the indices in the second dimension
        indices = np.random.choice(F_div_u.shape[1], size=num_changes, replace=False)

        # Apply the pattern to the selected indices
        F_div_u[:, indices] = pattern
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div random", "color": "tab:purple"}
    
    
    if name == "Divergence_control_noise":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::8] = -1 
        
        noise_level = 0.001
        noise = np.random.normal(loc=0, scale=noise_level, size=(N,N))
        F_div_u = F_div_u + noise

        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div noise", "color": "tab:purple"}
    
    if name == "Divergence_control_noise_plus":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::8] = -1 
        
        noise_level = 0.01
        noise = np.random.normal(loc=0, scale=noise_level, size=(N,N))
        F_div_u = F_div_u + noise

        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div noise plus", "color": "tab:brown"}
    
    if name == "Divergence_uneven":
        F_div_u = np.zeros((N, N))
        
        F_div_u[:, ::10] = -1 
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div uneven", "color": "tab:red"}
    
    if name == "Divergence_uneven_noise":
        F_div_u = np.zeros((N, N))
        
        F_div_u[:, ::10] = -1 
        noise_level = 0.001
        noise = np.random.normal(loc=0, scale=noise_level, size=(N,N))
        F_div_u = F_div_u + noise

        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div uneven noise", "color": "tab:gray"}
    
    
    if name == "Divergence_SNR_10":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::8] = -1 
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div SNR 10", "color": "tomato"}
    
    if name == "Divergence_SNR_100":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::8] = -1 
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 100  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div SNR 100", "color": "mediumorchid"}
    
    if name == "Divergence_SNR_1":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::8] = -1 
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 1  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div SNR 1", "color": "blue"}
    
    if name == "Divergence_smallangle":
        F_div_u = np.zeros((N, N))
                
        spacing = int(10*np.sqrt(2))

        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                F_div_u[i, j] = -np.sqrt(1/2)
                #if j + 1 < N:  # Ensure we do not go out of bounds
                #    F_div_u[i, j + 1] = -np.sqrt(1/2)
                #if j + 2 < N:  # Ensure we do not go out of bounds
                #    F_div_u[i, j + 2] = -np.sqrt(1/2)

        F_div_v = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                F_div_v[i, j] = -np.sqrt(1/2)
                #if j + 1 < N:  # Ensure we do not go out of bounds
                #    F_div_v[i, j + 1] = -np.sqrt(1/2)
                #if j + 2 < N:  # Ensure we do not go out of bounds
                #    F_div_v[i, j + 2] = -np.sqrt(1/2)
                
        F_div_u[:, -1] += -np.sqrt(1/2)
        #F_div_u[-1, :]  = -np.sqrt(1/2)
        #F_div_v[:, -1] = -np.sqrt(1/2)
        F_div_v[-1, :]  += -np.sqrt(1/2) 
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div angle", "color": "tab:green"}
    
    
    if name == "divergence_with_angle_no_weighted":
        
        angle = 90
        boundary_value=-1
        spacing = 8

        F_base_u = np.zeros((N, N))
        F_base_u[:, ::spacing] = -1  # Vertical lines with spacing
    
        # Rotate the matrix to the specified angle
        F_rotated_u = rotate(F_base_u, angle, reshape=False, order=1, mode='nearest')
    
        # Fill gaps due to rotation by thresholding
        F_rotated_u = np.where(F_rotated_u < -0.5, -1, 0)  # Threshold to ensure discrete values
        
        F_base_v = np.zeros((N, N))
        F_base_v[:, 1::spacing] = -1  # Vertical lines with spacing
    
        # Rotate the matrix to the specified angle
        F_rotated_v = rotate(F_base_v, angle, reshape=False, order=1, mode='nearest')
    
        # Fill gaps due to rotation by thresholding
        F_rotated_v = np.where(F_rotated_v < -0.5, -1, 0)  # Threshold to ensure discrete values
    
        # Create separate divergence fields for u and v
        F_div_u = F_rotated_u.copy()
        F_div_v = F_rotated_v.copy()
        
        F_div_u = np.zeros((N, N))
        print(F_div_u)
        print(F_div_v)
    
        # Set the boundary values
        #F_div_u[:, -1] = boundary_value
        #F_div_u[-1, :] = boundary_value
        #F_div_v[:, -1] = boundary_value
        #F_div_v[-1, :] = boundary_value
    
        # Adjust the total sum of F_div_u and F_div_v to -1
        total_sum = np.sum(F_div_u) + np.sum(F_div_v)
        adjustment_factor = -1 / total_sum
        #F_div_u *= adjustment_factor
        #F_div_v *= adjustment_factor
    
        # Combine the fields
        F = np.vstack([F_div_u, F_div_v])
    
        return {"F": F, "exp_type": "div", "name": f"Div angle {angle}°", "color": "tab:green"}
   
   
    if name == "divergence_with_angle":
        
        angle = 45
        boundary_value=-1
        # Create a base vertical-line matrix
        spacing = 8  # Adjustable spacing for the pattern
        F_base_u = np.zeros((N, N))
        F_base_u[:, ::spacing] = -1  # Vertical lines with spacing
    
        # Rotate the matrix to the specified angle
        F_rotated_u = rotate(F_base_u, angle, reshape=False, order=1, mode='nearest')
    
        # Fill gaps due to rotation by thresholding
        F_rotated_u = np.where(F_rotated_u < -0.5, -1, 0)  # Threshold to ensure discrete values
    
        F_base_v = np.zeros((N, N))
        F_base_v[:, 1::spacing] = -1  # Vertical lines with spacing
    
        # Rotate the matrix to the specified angle
        F_rotated_v = rotate(F_base_v, angle, reshape=False, order=1, mode='nearest')
    
        # Fill gaps due to rotation by thresholding
        F_rotated_v = np.where(F_rotated_v < -0.5, -1, 0)  # Threshold to ensure discrete values

        # Calculate weights based on angle
        angle_rad = np.radians(angle)
        weight_u = np.cos(angle_rad)
        weight_v = np.sin(angle_rad)
    
        # Apply weights to create F_div_u and F_div_v
        F_div_u = weight_u * F_rotated_u
        F_div_v = weight_v * F_rotated_v
    
        # Set the boundary values
        F_div_u[:, -1] = boundary_value*weight_u
        #F_div_u[0, :] = boundary_value*weight_u
        #F_div_v[:, -1] = boundary_value*weight_v
        F_div_v[-1, :] = boundary_value*weight_v
    
        # Adjust the total sum of F_div_u and F_div_v to -1
        #total_sum = np.sum(F_div_u) + np.sum(F_div_v)
        #adjustment_factor = -1 / total_sum
        #F_div_u *= adjustment_factor
        #F_div_v *= adjustment_factor
    
        # Combine the fields
        F = np.vstack([F_div_u, F_div_v])
    
        return {"F": F, "exp_type": "div", "name": f"Div angle {angle}°", "color": "tab:green"} 
    
    if name == "Divergence_reversed":
        F_div_u = np.zeros((N, N))
        F_div_v = np.zeros((N, N))
        F_div_v[::8, :] = -1 
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div reversed", "color": "tab:orange"}
    
    if name == "Divergence_intensity":
        F_div_u = np.zeros((N, N))
        
        F_div_u[:, ::8] = -2
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div intensity", "color": "tab:cyan"}
    
    if name == "Divergence_width":
        F_div_u = np.zeros((N, N))
        
        F_div_u[:, ::8] = -1
        F_div_u[:, 1::8] = -1 
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div width", "color": "tab:pink"}
    
    if name == "Divergence_divergence":
        F_div_u = np.zeros((N, N))
        
        F_div_u[:, ::8] = 1
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div div", "color": "tab:olive"}
    
    if name == "Divergence_angle":
        # Initialize arrays
        F_div_v = np.zeros((N, N))
        #np.fill_diagonal(F_div_v[::-1], 1)
        F_div_u = np.zeros((N, N))
        np.fill_diagonal(F_div_u[0:], -1) 
        #np.fill_diagonal(F_div_u[3:], 1) 
        #np.fill_diagonal(F_div_v[1:], -1) 
        np.fill_diagonal(F_div_v[4:], 1) 
        #F_div_u[:,-1:] = 0
        F_div_v[:,-1:] = 0
        #F_div_u[-1:,:] = 0
        #F_div_v[-1:,:] = 0
        #F_div_u[:1,:] = 0
        F_div_v[:1,:] = 0
        #F_div_u[:,:1] = 0
        F_div_v[:,:1] = 0
        # Combine into F
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div angle", "color": "tab:brown"}
    
    if name == "Divergence_density":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, int(2 * N / 16):int(3 * N / 16)] = -1
        #F_div_u[:, int(4 * N / 16):int(5 * N / 16)] = 1
        #F_div_u[:, int(-2 * N / 16):int(-1 * N / 16)] = -1
        #F_div_u[:, int(7 * N / 16):int(8 * N / 16)] = 1
        #F_div_u[:, int(10 * N / 16):int(11 * N / 16)] = 1
        #F_div_u[:, int(13 * N / 16):int(14 * N / 16)] = 1
        
        F_div_u[:, int(2 * N / 16):int(3 * N / 16)] = -1
        F_div_u[:, int(6 * N / 16):int(7 * N / 16)] = -1
        F_div_u[:, int(9 * N / 16):int(10 * N / 16)] = -1
        F_div_u[:, int(13 * N / 16):int(14 * N / 16)] = -1
        F_div_v = np.zeros((N, N))
    
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div density", "color": "tab:red"}
    
    if name == "Divergence_intensity":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, int(2 * N / 16):int(3 * N / 16)] = -3
        #F_div_u[:, int(-2 * N / 16):int(-1 * N / 16)] = -5
        #F_div_u[:, int(5 * N / 16):int(6 * N / 16)] = 8
        #F_div_u[:, int(12 * N / 16):int(13 * N / 16)] = 4
        F_div_u[:, int(2 * N / 16):int(3 * N / 16)] = -10
        F_div_u[:, int(9 * N / 16):int(10 * N / 16)] = -5

        F_div_v = np.zeros((N, N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div intensity", "color": "tab:green"}
    
    if name == "Divergence1":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, 2:3] = -1
        F_div_v = np.zeros((N, N))
        F_div_v[6:7, :] = -1
        F_div_v[12:13, :] = -1
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Divergence1", "color": "tab:brown"}
    
    if name == "Divergence1_1":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, 2:3] = -1
        F_div_v = np.zeros((N, N))
        F_div_v[6:7, :] = -1
        F_div_v[11:13, :] = 1
        F_div_u[:, 6:7] = -1
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Divergence1_1", "color": "tab:brown"}

    
    if name == "Shear0":
        F_shear_u = np.zeros((N, N))
        #F_shear_u[:, 6:8] = -1
        #F_shear_u[:, :1] = 1
        #F_shear_u[:, -1:] = 1
        F_shear_v = np.zeros((N, N))
        #F_shear_v[:, 0:1] = 1
        F_shear_v[3:5, :] = -1
        #F_shear_u[:, 3:5] = -1
        #F_shear_v[-1:, :] = 8
        F = np.vstack([F_shear_v, F_shear_u])
        return {"F": F, "exp_type": "shear", "name": "Shear0", "color": "tab:orange"}
    
    if name == "Shear1":
        F_shear_u = np.zeros((N, N))
        #F_shear_u[:, :1] = 1
        #F_shear_u[:, -1:] = 1
        F_shear_v = np.zeros((N, N))
        F_shear_v[6:7, :] = -1
        F = np.vstack([F_shear_v, F_shear_u])
        return {"F": F, "exp_type": "shear", "name": "Shear1", "color": "tab:orange"}
    
    if name == "Shear1_2":
        F_shear_u = np.zeros((N, N))
        F_shear_u[:, 6:7] = -1
        F_shear_u[:, 10:11] = 1
        #F_shear_u[:, :1] = 1
        #F_shear_u[:, -1:] = 1
        F_shear_v = np.zeros((N, N))
        #F_shear_v[6:7, :] = -1
        F = np.vstack([F_shear_v, F_shear_u])
        return {"F": F, "exp_type": "shear", "name": "Shear1_2", "color": "tab:orange"}
    
    if name == "Shear1_1":
        F_shear_v = np.zeros((N, N))
        np.fill_diagonal(F_shear_v[::-1], 1)
        F_shear_u = np.zeros((N, N))
        np.fill_diagonal(F_shear_u[1:], -1) 
        F = np.vstack([F_shear_v, F_shear_u])
        return {"F": F, "exp_type": "shear", "name": "Shear1_1", "color": "tab:orange"}
    
    if name == "DivShear0":
        Div_u = np.zeros((N,N))
        Div_v = np.zeros((N,N))
        Shear_u = np.zeros((N,N))
        Shear_v = np.zeros((N,N))

        #Div_u[:, 6:7] = -1
        Div_v[6:7, :] = -1
        Shear_v[:, 6:7] = -1
        #Shear_u[5:6, :] = -1

        F = np.vstack([Div_u, Shear_u, Div_v, Shear_v])
        #F = np.sqrt(F_shear**2 + F_div**2)
        return {"F": F, "exp_type": "both", "name": "DivShear0", "color": "tab:green"}
    
    # Add more experiments as needed
    raise ValueError(f"Experiment '{name}' not found.")