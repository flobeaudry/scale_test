import numpy as np
from scipy.ndimage import rotate

def get_experiment(name):
    #N = 1024 # Grid size
    N = int(1024)
    dx, dy = 1, 1 # Grid resolution
    mean, std = 0, 0.1
    
    #spacing_control = 16
    #spacing_small = 4
    
    spacing_control = 4
    spacing_small = 2
    
    mean_intensity = 0.1
    
    mean = 0
    std = 1.0
    gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
    gaussian_values_div = (np.round(gaussian_values)) #make the values round values (either 0 or 1)
    
    
    
    if name == "control":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "control", "color": "tab:blue"}
    
    if name == "45 angle":
        spacing = int(spacing_control*np.sqrt(2))
        
        F_div_u = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    F_div_u[i, j] = 1*mean_intensity

        F_div_v = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    F_div_v[i, j] = 1*mean_intensity
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "45 angle", "color": "tab:cyan"}
    
    if name == "irregular spacing":
        mean = 1.0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(np.round(gaussian_values))*mean_intensity #make the values round values (either 0 or 1)

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular spacing", "color": "tab:green"}
   
    if name == "narrow spacing":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_small] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "narrow spacing", "color": "tab:olive"}
    
    if name == "irregular intensity":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(gaussian_values)*mean_intensity # Only positive values

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular intensity", "color": "tab:pink"}

    if name == "irregular domain":
        N = N-24
        
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular domain", "color": "tab:purple"}
    
    if name == "errors weighted":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        noise = np.random.randn(N, N)*mean_intensity/10
        F_div_u = (F_div_u + noise)/10
        
        noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)/10
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors weighted", "color": "firebrick"}
    
    
    

    
    
    
    
    
    
    
    
    
    if name == "control +-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        F_div_u[:, spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_control * 3] = -1*mean_intensity  
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "control +-", "color": "tab:blue"}
    
    
    if name == "irregular spacing +-":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)

        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular spacing +-", "color": "tab:green"}
    
    
    if name == "narrow spacing +-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_small::spacing_small*2] = 1 
        #F_div_u[:, ::spacing_small*2] = -1 
        F_div_u[:, spacing_small::spacing_small * 3] = 1*mean_intensity  
        F_div_u[:, 2 * spacing_small::spacing_small * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_small * 3] = -1*mean_intensity  
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "narrow spacing +-", "color": "tab:olive"}
    
    
    if name == "irregular intensity +-":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = ((gaussian_values)*mean_intensity) # Only positive values
        
        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular intensity +-", "color": "tab:pink"}
    

    if name == "irregular domain +-":
        N = N-24
        
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        F_div_u[:, spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_control * 3] = -1*mean_intensity  
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular domain +-", "color": "tab:purple"}
    
    if name == "errors +-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        F_div_u[:, spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_control * 3] = -1*mean_intensity  

        noise = np.random.randn(N, N)*mean_intensity/10
        F_div_u = F_div_u + noise
        
        noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + noise2
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors +-", "color": "tab:red"}
    
    
    
    
    
    
    
    
    if name == "DIV+_oneline":
        F_div_u = np.zeros((N, N))
        spacing_control = 255
        #F_div_u[:, int(N/2)] = 10
        #F_div_u[:, int(N/2-1)] = 10
        F_div_u[:, ::spacing_control] = 10
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+_oneline", "color": "blue"}
    
    
    if name == "DIV+constant":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1 
        F_div_u = np.ones((N,N))
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+constant", "color": "red"}
    if name == "DIV+increase":

        F_div_u = np.array([[1 + 2 * i for i in range(N)] for _ in range(N)]) / N
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+increase", "color": "blue"}
    
    
    
    if name == "DIV+":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+", "color": "tab:blue"}
    
    
    if name == "DIV+45":
        spacing = int(spacing_control*np.sqrt(2))
        
        F_div_u = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    F_div_u[i, j] = 1

        F_div_v = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    F_div_v[i, j] = 1
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+45", "color": "tab:cyan"}
    
    
    if name == "DIV+density":
        mean = 1.0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(np.round(gaussian_values)) #make the values round values (either 0 or 1)

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+density", "color": "tab:green"}
    
    
    if name == "DIV+frequency":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_small] = 1 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+frequency", "color": "tab:olive"}
    
    if name == "DIV+intensity":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(gaussian_values) # Only positive values

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+intensity", "color": "tab:pink"}
    

    if name == "DIV+domain":
        N = N-24
        
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+domain", "color": "tab:purple"}
    
    if name == "DIV+errors":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1 
        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+errors", "color": "tab:red"}
    
    if name == "errors_randn":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1 
        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        #noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        
        noise = np.random.randn(N, N)
        
        F_div_u = np.zeros((N, N))
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors_randn", "color": "tab:pink"}
    
    if name == "errors_c1":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1 
        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        #noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        
        noise = np.random.randn(N, N)
        noise2 = np.random.randn(N, N)
        noise = np.random.rand(N, N)
        noise2 = np.random.rand(N, N)
        
        F_div_u = np.zeros((N, N))
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + noise2
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors_c1", "color": "firebrick"}
        #return {"F": F, "exp_type": "div", "name": "errors", "color": "orangered"}
        
    if name == "errors2":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1 
        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        #noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        
        noise = np.random.randn(N, N)
        noise2 = np.random.randn(N, N)
        noise = np.random.rand(N, N)
        noise2 = np.random.rand(N, N)
        
        F_div_u = np.zeros((N, N))
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + noise2
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors", "color": "orangered"}
    
    if name == "errors_speckle":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1 
        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        #noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        noise = np.random.randn(N, N)
        
        L=1
        field = np.abs(np.random.randn(N, N))  # Ensure positive values
        # Generate speckle noise (Gamma-distributed)
        speckle_noise = np.random.gamma(shape=L, scale=1.0/L, size=(N, N))
        # Apply speckle noise
        noise = field * speckle_noise

        F_div_u = np.zeros((N, N))
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors_speckle", "color": "tab:purple"}
    
    
    
    
    if name == "DIV+-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        F_div_u[:, spacing_control::spacing_control * 3] = 1  
        F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1  
        F_div_u[:, ::spacing_control * 3] = -1  
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-", "color": "blue"}
    
    if name == "DIV+-45":
        spacing = int(spacing_control*np.sqrt(2))
        print(spacing)
        
        F_div_u = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N-(2*spacing), spacing*2):
                if i + j >= N and i + spacing < N and j + spacing < N:
                    F_div_u[i, j] = 1
                    F_div_u[i+spacing, j+spacing] = -1


        F_div_v = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N-(2*spacing), spacing*2):
                if i + j >= N and i + spacing < N and j + spacing < N:
                    F_div_v[i, j] = 1
                    F_div_v[i+spacing, j+spacing] = -1

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-45", "color": "tab:cyan"}
    
    if name == "DIV+-density":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)

        #F_div_u = np.zeros((N, N))
        F_div_u = np.ones((N, N))*0.1
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-density", "color": "tab:green"}
    if name == "DIV+-density_c1":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)

        #F_div_u = np.zeros((N, N))
        F_div_u = np.ones((N, N))*0.1
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-density_c1", "color": "lime"}
    
    if name == "DIV+-frequency":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_small::spacing_small*2] = 1 
        #F_div_u[:, ::spacing_small*2] = -1 
        F_div_u[:, spacing_small::spacing_small * 3] = 1  
        F_div_u[:, 2 * spacing_small::spacing_small * 3] = 1  
        F_div_u[:, ::spacing_small * 3] = -1  
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-frequency", "color": "tab:olive"}
    
    if name == "DIV+-frequency_c1":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_small::spacing_small*2] = 1 
        #F_div_u[:, ::spacing_small*2] = -1 
        F_div_u[:, spacing_small::spacing_small * 3] = 1  
        F_div_u[:, 2 * spacing_small::spacing_small * 3] = 1  
        F_div_u[:, ::spacing_small * 3] = -1  
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-frequency_c1", "color": "yellow"}
    
    if name == "DIV+-intensity":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (gaussian_values) # Only positive values

        #F_div_u = np.zeros((N, N))
        F_div_u = np.ones((N, N))*0.1
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-intensity", "color": "tab:pink"}
    
    if name == "DIV+-intensity_c1":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (gaussian_values) # Only positive values

        #F_div_u = np.zeros((N, N))
        F_div_u = np.ones((N, N))*0.1
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-intensity_c1", "color": "fuchsia"}
    

    if name == "DIV+-domain":
        N = N-24
        
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        F_div_u[:, spacing_control::spacing_control * 3] = 1  
        F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1  
        F_div_u[:, ::spacing_control * 3] = -1  
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-domain", "color": "tab:purple"}
    
    if name == "DIV+-errors":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        F_div_u[:, spacing_control::spacing_control * 3] = 1  
        F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1  
        F_div_u[:, ::spacing_control * 3] = -1  
        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-errors", "color": "tab:red"}
    
    
    
    
    
    
    
    
    
    if name == "DIVs":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)
        gaussian_values = gaussian_values_div

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIVs", "color": "tab:green"}
    
    if name == "DIVs45":
        mean = 0
        std = 1.0
        spacing = int(spacing_control*np.sqrt(2))
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N)
        gaussian_values = (np.round(gaussian_values))
        
        F_div_u = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    F_div_u[i, j] = gaussian_values[j]

        F_div_v = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    F_div_v[i, j] = gaussian_values[j]
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIVs45", "color": "tab:cyan"}
    
    if name == "DIVsfrequency":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_small)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_small)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIVsfrequency", "color": "tab:olive"}
    
    if name == "DIVsintensity":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (gaussian_values) # Only positive values

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIVsintensity", "color": "tab:pink"}
    

    if name == "DIVsdomain":
        N = N-24
        
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)
        gaussian_values = gaussian_values_div

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIVsdomain", "color": "tab:purple"}
    
    if name == "DIVserrors":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        signal_power = np.mean(F_div_u**2)
        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIVserrors", "color": "tab:red"}
    
    
    
    
    
    
    if name == "DIV+RAMPlin":
        F_div_u = np.zeros((N, N))
    
        # Ramp line intensity from 0 to 2 linearly across the grid
        num_lines = N // spacing_control  # Number of ramped lines
        intensities = np.linspace(0, 2, num_lines)  # Linearly ramp intensities from 0 to 2

        for idx, intensity in enumerate(intensities):
            line_position = idx * spacing_control
            if line_position < N:
                F_div_u[:, line_position] = intensity

        F_div_v = np.zeros((N, N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+RAMPlin", "color": "magenta"}
    
    if name == "DIV+RAMPsin":
        F_div_u = np.zeros((N, N))
    
        n_cycles = 16*4
        num_lines = N // spacing_control
        intensities = abs(np.sin(2 * np.pi * np.arange(num_lines) * n_cycles / num_lines) ) # Repeat n_cycles times

        for idx, intensity in enumerate(intensities):
            line_position = idx * spacing_control
            if line_position < N:
                F_div_u[:, line_position] = intensity
        
        F_div_v = np.zeros((N, N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+RAMPsin", "color": "tomato"}
    
    if name == "DIV+-RAMPsin":
        F_div_u = np.zeros((N, N))
    
        n_cycles = 16*4
        num_lines = N // spacing_control
        intensities = np.sin(2 * np.pi * np.arange(num_lines) * n_cycles / num_lines)  # Repeat n_cycles times
        intensities = 1.5 * intensities + 1 

        for idx, intensity in enumerate(intensities):
            line_position = idx * spacing_control
            if line_position < N:
                F_div_u[:, line_position] = intensity
        
        F_div_v = np.zeros((N, N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-RAMPsin", "color": "indianred"}
    
    if name == "DIV+-RAMPsin_pino2":
        F_div_u = np.zeros((N, N))
    
        n_cycles =70
        num_lines = N // spacing_control
        intensities = np.sin(2 * np.pi * np.arange(num_lines) * n_cycles / num_lines)  # Repeat n_cycles times
        intensities = 1.5 * intensities + 1 

        for idx, intensity in enumerate(intensities):
            line_position = idx * spacing_control
            if line_position < N:
                F_div_u[:, line_position] = intensity
        
        F_div_v = np.zeros((N, N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-RAMPsin_pino2", "color": "teal"}
    
    if name == "DIV+-RAMPlin":
        F_div_u = np.zeros((N, N))
    
        # Ramp line intensity from 0 to 2 linearly across the grid
        num_lines = N // spacing_control  # Number of ramped lines
        intensities = np.linspace(-2, 2, num_lines)  # Linearly ramp intensities from 0 to 2

        for idx, intensity in enumerate(intensities):
            line_position = idx * spacing_control
            if line_position < N:
                F_div_u[:, line_position] = intensity

        F_div_v = np.zeros((N, N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "DIV+-RAMPlin", "color": "mediumvioletred"}
    
    
    
    
    
    
    if name == "Divergence_control":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, int(2 * N / 8):int(3 * N / 8)] = -1
        #F_div_u[:, int(9 * N / 8):int(10 * N / 8)] = -1
        
        F_div_u[:, ::spacing_control] = -1 
        #F_div_u[:, :int(N/2)] = 0
        
        #F_div_u[:, int(5 * N / 16):int(6 * N / 16)] = 1
        #F_div_u[:, int(12 * N / 16):int(13 * N / 16)] = 1
        #F_div_u[:, 2:3] = -1
        #F_div_u[:, -2:-1] = -1
        #F_div_u[:, 5:6] = 1
        #F_div_u[:, 12:13] = 1
        
        F_div_v = np.zeros((N, N))
        F_div_u[:,0] = -1
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Conv control", "color": "tab:blue"}
    
    if name == "Divergence_control_div":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, int(2 * N / 8):int(3 * N / 8)] = -1
        #F_div_u[:, int(9 * N / 8):int(10 * N / 8)] = -1
        
        F_div_u[:, ::spacing_control] = 1 
        F_div_u[:, :int(N/2)] = 0
        
        #F_div_u[:, int(5 * N / 16):int(6 * N / 16)] = 1
        #F_div_u[:, int(12 * N / 16):int(13 * N / 16)] = 1
        #F_div_u[:, 2:3] = -1
        #F_div_u[:, -2:-1] = -1
        #F_div_u[:, 5:6] = 1
        #F_div_u[:, 12:13] = 1
        
        F_div_v = np.zeros((N, N))
        F_div_u[:,0] = 1
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div control", "color": "tab:olive"}
    
    
    if name == "Divergence_conv_control":
        F_div_u = np.zeros((N, N))
        
        F_div_u[:, spacing_control::spacing_control*2] = -1 
        F_div_u[:, ::spacing_control*2] = 1 
        #F_div_u[:, :int(N/2)] = 0
        
        
        F_div_v = np.zeros((N, N))
        #F_div_u[:,0] = -1
        F = np.vstack([F_div_u, F_div_v])
        
        print(np.mean(F_div_u))
        print(np.sum(F_div_u))
        return {"F": F, "exp_type": "div", "name": "Div-Conv control", "color": "tab:red"}
    
    if name == "Divergence_conv_control_4":
        F_div_u = np.zeros((N, N))
        
        F_div_u[:, spacing_small::spacing_small*2] = -1 
        F_div_u[:, ::spacing_small*2] = 1 
        F_div_u[:, :int(N/2)] = 0
        
        
        F_div_v = np.zeros((N, N))
        F_div_u[:,0] = -1
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div-Conv control-4", "color": "tab:cyan"}
        
    if name == "Divergence_control_half":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, int(2 * N / 8):int(3 * N / 8)] = -1
        #F_div_u[:, int(9 * N / 8):int(10 * N / 8)] = -1
        
        F_div_u[:, ::spacing_control] = -1 
        F_div_u[:, :int(N/2)] = 0
        
        F_div_v = np.zeros((N, N))
        #F_div_u[:,0] = -1
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div control half", "color": "tab:pink"}
    
    if name == "Divergence_spectrum":
        mean = 0
        std = 1.0

        #x_values = np.linspace(-5, 5, N // spacing_control)
        #print()
        #gaussian_values = np.exp(-0.5 * x_values**2)  # Standard Gaussian distribution
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = -abs(gaussian_values)

        # Initialize the divergence field
        F_div_u = np.zeros((N, N))

        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        #F_div_u[:, :int(N/2)] = 0
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Conv spectrum", "color": "tab:orange"}
    
    if name == "Divergence_spectrum_int":
        mean = 0
        std = 1.0

        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = -abs(np.round(gaussian_values))

        # Initialize the divergence field
        F_div_u = np.zeros((N, N))

        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Conv spec int", "color": "tab:cyan"}
    
    
    if name == "Divergence_spectrum_full":
        mean = 0
        std = 1.0
        
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)

        # Initialize the divergence field
        F_div_u = np.zeros((N, N))

        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div-conv spectrum", "color": "tab:pink"}
    
    if name == "Divergence_spectrum_full_4":
        mean = 0
        std = 1.0
        
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_small)

        # Initialize the divergence field
        F_div_u = np.zeros((N, N))

        for idx, j in enumerate(range(0, N, spacing_small)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "Div-conv spectrum-4", "color": "tab:purple"}
        return {"F": F, "exp_type": "div", "name": "Div-conv spectrum-4", "color": "cadetblue"}
    
    if name == "Divergence_spectrum_full_int":
        mean = 0
        std = 1.0
        
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = np.round(gaussian_values)
        # Initialize the divergence field
        F_div_u = np.zeros((N, N))

        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Div-conv s_vary", "color": "tab:brown"}
    
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
        F_div_u[:, ::spacing_control] = -1 
    
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        #noise = -abs(noise)
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "Div SNR 10", "color": "tomato"}
        return {"F": F, "exp_type": "div", "name": "CC SNR 10", "color": "blueviolet"}
    
    if name == "Divergence_SNR_100":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = -1 
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 100  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        #noise = -abs(noise)
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "Div SNR 100", "color": "mediumorchid"}
        return {"F": F, "exp_type": "div", "name": "CC SNR 100", "color": "tomato"}
    
    if name == "Divergence_SNR_1":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = -1 
        
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 1  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        #noise = -abs(noise)
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "Div SNR 1", "color": "blue"}
        return {"F": F, "exp_type": "div", "name": "CC SNR 1", "color": "darkred"}
    
    if name == "Divergence_DSC_10":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = -1 
        
        
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_small)
        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_small)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 10  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        #noise = -abs(noise)
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "Div SNR 10", "color": "tomato"}
        return {"F": F, "exp_type": "div", "name": "DCS SNR 10", "color": "mediumorchid"}
    
    if name == "Divergence_DSC_100":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = -1 
        
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_small)
        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_small)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 100  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        #noise = -abs(noise)
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "Div SNR 100", "color": "mediumorchid"}
        return {"F": F, "exp_type": "div", "name": "DCS SNR 100", "color": "salmon"}
    
    if name == "Divergence_DSC_1":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = -1 
        
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_small)
        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_small)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column
        
        signal_power = np.mean(F_div_u**2)

        noise_power = signal_power / 1  # SNR of 10 means noise power is 1/10th of signal power
        noise_std = np.sqrt(noise_power)
        
        # gaussian noise
        #noise = np.random.normal(loc=0, scale=noise_std, size=(N, N))
        # uniform noise between -sqrt(3) sigma, +sqrt(3) sigma
        noise = np.random.uniform(low=-noise_std*np.sqrt(3), high=noise_std*np.sqrt(3), size=(N, N))
        #noise = -abs(noise)
        
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "Div SNR 1", "color": "blue"}
        return {"F": F, "exp_type": "div", "name": "DCS SNR 1", "color": "saddlebrown"}
    
    if name == "Divergence_smallangle":
        F_div_u = np.zeros((N, N))
                
        spacing = int(spacing_control*np.sqrt(2))

        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    #F_div_u[i, j] = -np.sqrt(1/2)
                    F_div_u[i, j] = -1
                #F_div_u[i:i+3, j:j+3] = -np.sqrt(1/2)
                #if j + 1 < N:  # Ensure we do not go out of bounds
                #    F_div_u[i, j + 1] = -np.sqrt(1/2)
                #if j + 2 < N:  # Ensure we do not go out of bounds
                #    F_div_u[i, j + 2] = -np.sqrt(1/2)

        F_div_v = np.zeros((N, N))
        for i in range(N):
            for j in range((N - 1 - i) % spacing, N, spacing):
                if i + j >= N:
                    #F_div_v[i, j] = -np.sqrt(1/2)
                    F_div_v[i, j] = -1
                #F_div_v[i:i+3, j:j+3] = -np.sqrt(1/2)
                #if j + 1 < N:  # Ensure we do not go out of bounds
                #    F_div_v[i, j + 1] = -np.sqrt(1/2)
                #if j + 2 < N:  # Ensure we do not go out of bounds
                #    F_div_v[i, j + 2] = -np.sqrt(1/2)
                
        #F_div_u[:, 0] += -np.sqrt(1/2)
        #F_div_u[0, :]  += -np.sqrt(1/2)
        #F_div_v[:, 0] += -np.sqrt(1/2)
        #F_div_v[0, :]  += -np.sqrt(1/2) 
        
        #F_div_v[0, 0]  = 0
        #F_div_u[0, 0] = 0
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "Conv angle", "color": "tab:green"}
    
    
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