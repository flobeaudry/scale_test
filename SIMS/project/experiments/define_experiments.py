import numpy as np
import pickle
import math
from matplotlib.colors import to_rgb, to_hex
from scipy.ndimage import rotate
from experiments.define_experiments_helpers import draw_line, draw_simple_tree, generate_simple_tree, draw_radial_tree, generate_radial_tree_without_ratios, generate_radial_tree_too_full, generate_radial_tree, generate_full_radial_fracture_field, draw_fracture, generate_fracture_field_without_ratios, generate_fracture_field, draw_branch, generate_fractal_tree


def get_experiment(name):
    #N = 1024 # Grid size
    N = int(1024/2)
    #N = int(6)
    dx, dy = 1, 1 # Grid resolution
    mean, std = 0, 0.1
    
    #spacing_control = 16
    #spacing_small = 4
    
    #spacing_control = 4
    spacing_control = 8
    spacing_small = 3
    
    mean_intensity = 0.1
    
    mean = 0
    std = 1.0
    gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
    gaussian_values_div = (np.round(gaussian_values)) #make the values round values (either 0 or 1)
    
    if name == "sin+":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        min_val, max_val = 0, 1  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "sin+", "color": "darkkhaki"}
    
    if name == "sin01":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        min_val, max_val = 0, 1  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=N/3

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "k=N/3: sin 0 to 1", "color": "tab:orange"}
        return {"F": F, "exp_type": "div", "name": "continuous off-grid spacing", "color": "tab:cyan", "marker":"s"}
    
    if name == "sin-0.51":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        min_val, max_val = -0.5, 0.51  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k=N/3

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        #F_div_u = field
        
        
        x = np.arange(N)
        period = 6
        amplitude = 1
        mean = 0.1
        wave_1d = amplitude * np.sin(2*np.pi*x/period) + mean
        wave_2d = np.tile(wave_1d, (N,1))
        F_div_u = wave_2d
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        
        return {"F": F, "exp_type": "div", "name": "k=N/3: sin -0.25 to 0.75", "color": "tab:orange"}
    
    if name == "sin-11":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        min_val, max_val = -0.5, 0.51  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k = N/3
        #k = (1024/4)/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])

        #return {"F": F, "exp_type": "div", "name": "k=N/3: sin -0.5 to 0.5", "color": darker}
        return {"F": F, "exp_type": "div", "name": "λ=3Δx", "color": "tab:cyan", "marker":"s"}
    
    if name == "ksin01":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        min_val, max_val = 0, 1  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=N/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "k=N/4: sin 0 to 1", "color": "tab:green"}
    
    if name == "ksin-0.51":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        min_val, max_val = -0.25, 0.75  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k=N/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])

        return {"F": F, "exp_type": "div", "name": "k=N/4: sin -0.25 to 0.75", "color": "tab:green"}
    
    if name == "ksin-11":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        min_val, max_val = -0.5, 0.51  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k = N/4
        #k = (1024/4)/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        
        #return {"F": F, "exp_type": "div", "name": "k=N/4: sin -0.5 to 0.5", "color": darker}
        return {"F": F, "exp_type": "div", "name": "control (λ=4Δx)", "color": "tab:blue", "marker":"s"}
    
    
    
    
    
    if name == "sin01 err":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        #min_val, max_val = 0, 1  # Set min and max values
        min_val, max_val = 0, 0.1  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=N/3

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        #noise = abs(np.random.randn(N, N)*mean_intensity/10)
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = abs(F_div_u + noise)
        
        #noise2 = abs(np.random.randn(N, N)*mean_intensity/10)
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = abs(F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "k=N/3: sin 0 to 1 err", "color": "tab:orange"}
    
    if name == "sin-0.51 err":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        #min_val, max_val = -0.25, 0.75  # Set min and max values
        min_val, max_val = -0.025, 0.075  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k=N/3

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = (F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        
        return {"F": F, "exp_type": "div", "name": "k=N/3: sin -0.25 to 0.75 err", "color": "tab:orange"}
    
    if name == "sin-11 err":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        #min_val, max_val = -0.5, 0.51  # Set min and max values
        min_val, max_val = -0.05, -0.051  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k = N/3
        #k = (1024/4)/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = (F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        
        return {"F": F, "exp_type": "div", "name": "k=N/3: sin -0.5 to 0.5 err", "color": "tab:orange"}
    
    if name == "ksin01 err":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        #min_val, max_val = 0, 1  # Set min and max values
        min_val, max_val = 0, 0.1  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=N/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = abs(F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = abs(F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "k=N/4: sin 0 to 1 err", "color": "tab:green"}
    
    if name == "ksin-0.51 err":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        #min_val, max_val = -0.25, 0.75  # Set min and max values
        min_val, max_val = -0.025, 0.075  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k=N/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = (F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        
        return {"F": F, "exp_type": "div", "name": "k=N/4: sin -0.25 to 0.75 err", "color": "tab:green"}
    
    if name == "ksin-11 err":
        x = np.linspace(0, 2 * np.pi, N)  # x-domain
        y = np.linspace(0, 2 * np.pi, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define sinusoidal function parameters
        #min_val, max_val = -0.5, 0.51  # Set min and max values
        min_val, max_val = -0.05, 0.051  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude
        #k = 30  # Number of oscillations in the domain
        k=10
        k=10
        k = N/4
        #k = (1024/4)/4

        # Generate the (N, N) field
        field = A + B * np.sin(k * X)
        F_div_u = field
        
        print("TOT sum", np.sum(F_div_u))
        print("TOT mean", np.mean(F_div_u))
        
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = (F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        
        return {"F": F, "exp_type": "div", "name": "k=N/4: sin -0.5 to 0.5 err", "color": "tab:green"}
    
    
    
    
    
    
    
    
    if name == "exp":
        x = np.linspace(0, 1, N)  # x-domain centered at 0
        y = np.linspace(-1, 1, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define quadratic function parameters
        min_val, max_val = 0, 1  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude

        # Generate the (N, N) field with a quadratic function in x
        #field = A + B * (X ** 4)
        field = X**2
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "exponential 0 to 1", "color": "tab:green"}
    
    if name == "exp +-":
        x = np.linspace(0, 0.1, N)  # x-domain centered at 0
        y = np.linspace(0, 0.1, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define quadratic function parameters
        min_val, max_val = -0.5, 0.5  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude

        # Generate the (N, N) field with a quadratic function in x
        #field = A + B * (X ** 4)
        field = X**2
        F_div_u = field
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "exponential -0.5 to 0.5", "color": "tab:green"}
    
    if name == "lin":
        x = np.linspace(-0.5, 0.5, N)  # x-domain centered at 0
        y = np.linspace(0, 0, N)  # y-domain
        X, Y = np.meshgrid(x, y)

        # Define quadratic function parameters
        min_val, max_val = -1, 1  # Set min and max values
        A = (max_val + min_val) / 2  # Mean value
        B = (max_val - min_val) / 2  # Amplitude

        # Generate the (N, N) field with a quadratic function in x
        #field = A + B * (X ** 4)
        field = X**32
        F_div_u = field
        F_div_u = X
        
        #F_div_u = np.ones((N,N))*0.1
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "exp +-", "color": "red"}
    
    if name == "control":
        F_div_u = np.zeros((N, N))
        #spacing_control=3
        offset = 4
        F_div_u[:, offset::spacing_control] = 1*mean_intensity 
        #F_div_u[:, offset+1::spacing_control] = 1*mean_intensity  # second line !
        
        # vertical lines !
        #F_div_u[:, ::spacing_control] = 1*mean_intensity 
        #F_div_u[:, 1::spacing_control] = 1*mean_intensity  # second line !
        
        # horizontal lines !
        F_div_u[::spacing_control, :] = 1*mean_intensity
        #F_div_u[1::spacing_control, :] = 1*mean_intensity  
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        #F = np.vstack([F_div_v, F_div_u])
        return {"F": F, "exp_type": "div", "name": "control (s=4Δx)        ", "color": "tab:blue", "marker":"o"}
        #return {"F": F, "exp_type": "shear", "name": "control (s=4Δx)        ", "color": "tab:blue", "marker":"o"}
    
    
    if name == "control_diamonds_angled":
        angle_deg = 30
        spacing = spacing_control
        theta = np.deg2rad(angle_deg)

        x, y = np.meshgrid(np.arange(N), np.arange(N))

        # Family 1: rotated +angle
        coord1 = x * np.cos(theta) + y * np.sin(theta)
        lines1 = ((coord1 % spacing) < 1).astype(float)

        # Family 2: rotated -angle
        coord2 = x * np.cos(-theta) + y * np.sin(-theta)
        lines2 = ((coord2 % spacing) < 1).astype(float)

        # Combine both
        F_div_u = lines1
        F_div_u += lines2
        
        F_div_v = np.zeros((N,N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "control {angle_deg}deg diamonds", "color": "tab:purple", "marker": "o"}
    
    if name == "control_diamonds":
        angle_deg = 45
        spacing = spacing_control
        theta = np.deg2rad(angle_deg)

        # Coordinate grid
        x, y = np.meshgrid(np.arange(N), np.arange(N))

        # Rotate coordinates
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)

        # Create divergence patterns based on spacing
        F_div_u = ((x_rot % spacing) < 1).astype(float)
        F_div_u += ((y_rot % spacing) < 1).astype(float)
        F_div_v = np.zeros((N,N))

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": f"control_{angle_deg}deg", "color": "tab:olive", "marker":"o"}
    
    
    
    if name == "control err":
        F_div_u = np.zeros((N, N))
        #spacing_control=3
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors                   ", "color": "tab:blue", "marker":"s"}
    
    if name == "control err weighted":
        F_div_u = np.zeros((N, N))
        #spacing_control=3
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "$<\dot{\epsilon}{_{tot}} >{_{w}}$               err weighted", "color": "xkcd:royal blue", "marker":"s"}
    
    if name == "control err weighted2":
        F_div_u = np.zeros((N, N))
        #spacing_control=3
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "$< \partial u_i / \partial x_j >{_{w}}$                 R err weighted2", "color": "xkcd:bluish grey", "marker":"s"}
    
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
        return {"F": F, "exp_type": "div", "name": "45 angle", "color": "tab:cyan", "marker":"o"}
    
    if name == "irregular spacing":
        mean = 1.0
        std = 1.0
        spacing_control = 1
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(np.round(gaussian_values))*mean_intensity #make the values round values (either 0 or 1)

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N-1, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "irregular spacing", "color": "tab:purple", "marker":"o"}
        return {"F": F, "exp_type": "div", "name": "s≠constant", "color": "tab:purple", "marker":"o"}
   
    if name == "narrow spacing":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_small] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "off-grid spacing", "color": "tab:cyan", "marker":"o"}
        return {"F": F, "exp_type": "div", "name": "s=3Δx", "color": "tab:cyan", "marker":"o"}
    
    if name == "irregular intensity":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(gaussian_values)*mean_intensity # Only positive values

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N-1, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "irregular intensity", "color": "tab:pink", "marker":"o"}
        return {"F": F, "exp_type": "div", "name": "$\\mathbf{\\dot{\\epsilon}_{I}}$≠constant", "color": "tab:pink", "marker":"o"}


    if name == "irregular domain":
        N = N-1
        
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular domain", "color": "tab:purple", "marker":"o"}
    
    
    
    if name == "err":
        F_div_u = np.zeros((N, N))
        
        #noise = abs(np.random.randn(N, N)*mean_intensity/10)
        noise = np.random.uniform(0, mean_intensity/10, (N, N))
        F_div_u = (F_div_u + noise)
        
        #noise2 = abs(np.random.randn(N, N)*mean_intensity/10)
        noise2 = np.random.uniform(0, mean_intensity/10, (N, N))
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "err", "color": "tab:red"}
    
    if name == "err +-":
        F_div_u = np.zeros((N, N))
        
        #noise = (np.random.randn(N, N)*mean_intensity/10)
        noise = np.random.uniform(-mean_intensity/10, mean_intensity/10, (N, N))
        F_div_u = (F_div_u + noise)
        
        #noise2 = (np.random.randn(N, N)*mean_intensity/10)
        noise2 = np.random.uniform(-mean_intensity/10, mean_intensity/10, (N, N))
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "err +-", "color": "tab:red"}
    
    
    
    if name == "control err":
        F_div_u = np.zeros((N, N))
        #spacing_control=3
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
    
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = abs(F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = abs(F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "control err", "color": "tab:blue"}
    
    if name == "irregular spacing err":
        mean = 1.0
        std = 1.0
        spacing_control = 1
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(np.round(gaussian_values))*mean_intensity #make the values round values (either 0 or 1)

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = abs(F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = abs(F_div_v + noise2)
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular spacing err", "color": "tab:purple"}
   
    if name == "narrow spacing err":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_small] = 1*mean_intensity 
    
        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = abs(F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = abs(F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "narrow spacing (n=3) err", "color": "tab:cyan"}
    
    if name == "irregular intensity err":
        mean = 0
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = abs(gaussian_values)*mean_intensity # Only positive values

        F_div_u = np.zeros((N, N))
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        noise = (np.random.randn(N, N)*mean_intensity/10)
        F_div_u = abs(F_div_u + noise)
        
        noise2 = (np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = abs(F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular intensity err", "color": "tab:pink"}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    
    if name == "errors":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1*mean_intensity 
        noise = abs(np.random.randn(N, N)*mean_intensity/10)
        F_div_u = (F_div_u + noise)
        
        noise2 = abs(np.random.randn(N, N)*mean_intensity/10)
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors", "color": "tab:red"}
    
    if name == "onlyerrors":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, ::spacing_control] = 1*mean_intensity 
        noise = np.random.normal(0, 0.1, size = (N,N)) #white noise
        F_div_u = (F_div_u + noise)
        
        noise2 = np.random.normal(0, 0.1, size = (N,N))
        F_div_v = np.zeros((N, N))
        F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors", "color": "tab:red"}
    
    if name == "onlyerrors_speckle":
        F_div_u = np.zeros((N, N))
        F_div_u[:, ::spacing_control] = 1*mean_intensity
    
        # Generate speckle noise (Gamma-distributed)
        L = 1  # Number of looks (higher L -> less noise)
        mean_intensity = 0.1/10  # Average intensity
        speckle_noise_u = np.random.gamma(L, mean_intensity / L, size=(N, N))
    
        F_div_u = F_div_u + speckle_noise_u  # Multiplicative noise

        speckle_noise_v = np.random.gamma(L, mean_intensity / L, size=(N, N))
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + speckle_noise_v  # Multiplicative noise

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors", "color": "tab:brown"}
    
    if name == "errors_speckle":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, ::spacing_control] = 1*mean_intensity
    
        # Generate speckle noise (Gamma-distributed)
        L = 1  # Number of looks (higher L -> less noise)
        mean_intensity = 0.1/10  # Average intensity
        speckle_noise_u = np.random.gamma(L, mean_intensity / L, size=(N, N))
    
        F_div_u = F_div_u + speckle_noise_u  # Multiplicative noise

        speckle_noise_v = np.random.gamma(L, mean_intensity / L, size=(N, N))
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + speckle_noise_v  # Multiplicative noise

        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "onlyerrors", "color": "red"}
    
    if name == "koch":
        F_div_u = np.zeros((N, N))
        
        def koch_curve(p1, p2, depth, grid):
            if depth == 0:
                return
    
            # Calculate the points of division
            p3 = (2*p1 + p2) / 3
            p5 = (p1 + 2*p2) / 3
    
            # Calculate the peak point (rotation by 60°)
            angle = np.pi / 3
            vec = p5 - p3
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).dot(vec)
            p4 = p3 + rot
    
            # Draw the segments on the grid
            draw_line(p1, p3, grid)
            draw_line(p3, p4, grid)
            draw_line(p4, p5, grid)
            draw_line(p5, p2, grid)
    
            # Recursively subdivide
            koch_curve(p1, p3, depth-1, grid)
            koch_curve(p3, p4, depth-1, grid)
            koch_curve(p4, p5, depth-1, grid)
            koch_curve(p5, p2, depth-1, grid)

        # Function to draw a line in the grid with 1s
        def draw_line(p1, p2, grid):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            length = int(np.hypot(x2-x1, y2-y1))
            for i in range(length+1):
                t = i / length
                x = int((1-t) * x1 + t * x2)
                y = int((1-t) * y1 + t * y2)
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = 1

        # Main function to generate the fractal
        def generate_snowflake(N, depth):
            grid = np.zeros((N, N))
            size = N // 2
            center = N // 2
    
            # Initial equilateral triangle
            p1 = np.array([center, center - size//2])
            p2 = np.array([center - size//2, center + size//2])
            p3 = np.array([center + size//2, center + size//2])
    
            draw_line(p1, p2, grid)
            draw_line(p2, p3, grid)
            draw_line(p3, p1, grid)
    
            # Apply Koch curve recursively to each segment
            koch_curve(p1, p2, depth, grid)
            koch_curve(p2, p3, depth, grid)
            koch_curve(p3, p1, depth, grid)
    
            return grid

        # Parameters
        #N_fractal = int(N/2)
        N_fractal = int(N)
        depth = 4  # Recursion depth

        fractal = generate_snowflake(N_fractal, depth)
        
        #F_div_u = np.zeros((N,N))
        #F_div_u[N_fractal:, N_fractal:] = fractal
        F_div_u = fractal
    
        #F_div_v = np.zeros((N, N))
        #F_div_v[N_fractal:, N_fractal:] = fractal
        F_div_v = fractal
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "koch", "color": "orchid", 'marker':'P' }
    
    if name == "sierpinski":
        F_div_u = np.zeros((N, N))
        #sierpinski = np.loadtxt('SIMS/project/experiments/Sierpinski.txt', dtype=int)
        with open('SIMS/project/experiments/Sierpinski.txt') as f:
            sierpinski = np.array([[int(char) for char in line.strip()] for line in f])

        print("SHAPE", sierpinski.shape)
        F_div_u = sierpinski
        F_div_v = sierpinski
        
        #F_div_u = np.zeros((N,N))
        #F_div_u[N_fractal:, N_fractal:] = fractal
    
        #F_div_v = np.zeros((N, N))
        #F_div_v[N_fractal:, N_fractal:] = fractal
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "sierpinski", "color": "coral", 'marker':'P' }
    
    
    if name == "weierstrass":
        F_div_u = np.zeros((N, N))
        a = 0.5
        b = 5
        n_terms = 20
        
        x = np.linspace(-N, N ,N)
        W = np.zeros_like(x)
        for n in range (n_terms):
            W += a**n * np.cos(b**n * np.pi * x / N)
            
        #W = (W / np.max(np.abs(W)))
        W_pos = W
        F_div_u = np.tile(W_pos, (N, 1))  # vertical lines
        #F_div_u = pad_to_size(F, N)
        
        F_div_v = np.zeros((N,N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "weierstrass", "color": "coral", 'marker':'P' }
    
    if name == "fractal_tree":
        N_fractal = int(N/2*4)
        N_fractal = int(N/2)
        
        #N_fractal = int(N)
        
        # If we want a fixed ratio of 0 and 1's
        with open('SIMS/project/utils/rgps_div_ratios.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        print(loaded_data['thresholds'])
        print(loaded_data['ratios'])
        ratio = loaded_data["thresholds"][2]
        
        print(N_fractal)
        #tree_array = generate_fractal_tree(N=N_fractal, depth=5)
        #tree_array = generate_simple_tree(N=N_fractal, depth=11)
        
        #tree_array = generate_radial_tree(N=N_fractal, depth=10, branches_per_level=2)
        
        # with ratios !!
        #tree_array = generate_radial_tree(N=N_fractal, fill_fraction=ratio, max_trees=1000, depth=10, branches_per_level=3)
        #tree_array = generate_fracture_field(N=N_fractal, num_fractures = 15, depth = 10)
        tree_array = generate_fracture_field(N=N_fractal,depth = 6, fill_fraction = ratio, max_fractures = 1000)
        
        # re-calculate the ratio
        ones = len(np.where(tree_array != 0)[0])
        zeros = len(np.where(tree_array == 0)[0])
        ratio_recomp = ones/zeros
        print('input ratio: ', ratio, ' re-calc ratio: ', ratio_recomp)
        
        #print('NOZERO', np.where(tree_array != 0))
        F_div_u = np.zeros((N, N)) 
        #size_cut = int(N_fractal/4)
        size_cut = int(N_fractal/4)
        print(np.shape(tree_array))
        
        print(int(len(tree_array)/3))
        
        to_repeat = tree_array[:, int(len(tree_array)/4):-int(len(tree_array)/4)]
        new_tree_array = np.concatenate((to_repeat, to_repeat), axis=1)
        
        zoom_tree_array = tree_array[:-2*int(len(tree_array)/4), int(len(tree_array)/4):-int(len(tree_array)/4)]
        # need N_fractal = N
        
        
        #new_tree_array = np.concatenate((new_tree_array1, to_repeat), axis=0)
        #F_div_u[N_fractal//2:, N_fractal//2:] = tree_array[size_cut+20:-size_cut+20, size_cut:-size_cut]
        ########F_div_u[N_fractal//4:, N_fractal//4:] = tree_array[size_cut:-2*size_cut, size_cut:-2*size_cut]
        #F_div_u = tree_array                 
        
        #print(np.shape(F_div_u[N_fractal//2:, N_fractal//2:]))
        
        
        F_div_u[N//2:, N//2:] = tree_array
        #F_div_u[N//2:, N//2:] = new_tree_array
        #F_div_u[N//2:, N//2:] = zoom_tree_array
         
        F_div_v = np.zeros((N, N))                                 
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "fractal fracture", "color": "xkcd:pinky red", 'marker':'P' }
    
    
    if name == "radial_tree":
        N_fractal = int(N/2)
        
        # If we want a fixed ratio of 0 and 1's
        with open('SIMS/project/utils/rgps_div_ratios.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        ratio = loaded_data["thresholds"][1]
        
        # with ratios !!
        #tree_array = generate_radial_tree(N=N_fractal, fill_fraction=ratio, max_trees=1000, depth=4, branches_per_level=2)
        #tree_array = generate_radial_tree(N=N_fractal, fill_fraction=ratio, depth=4, branches_per_level=3)
        tree_array = generate_full_radial_fracture_field(N=N_fractal, fill_fraction=ratio, num_main_branches=50, depth=4, branch_angle_deg=30)
        # re-calculate the ratio
        ones = len(np.where(tree_array != 0)[0])
        zeros = len(np.where(tree_array == 0)[0])
        ratio_recomp = ones/zeros
        print('input ratio: ', ratio, ' re-calc ratio: ', ratio_recomp)
        
        F_div_u = np.zeros((N, N)) 
        F_div_u[N//2:, N//2:] = tree_array

        F_div_v = np.zeros((N, N))                                 
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "radial_tree", "color": "xkcd:marine blue", 'marker':'P' }
    
    if name == 'control_decay':
        
        def decaying_lines_in_segment(start, initial_spacing, decay_rate, segment_length, min_spacing=0.5):

            positions = []
            pos = 0
            spacing = initial_spacing
            while pos < segment_length and spacing >= min_spacing:
                positions.append(int(round(start + pos)))
                pos += spacing
                spacing *= decay_rate
            return positions

        # Parameters
        segment_length = int(N/8)
        initial_spacing = 4
        decay_rate = 0.9
        min_spacing = 0.5

        # Initialize field
        F_div_u = np.zeros((N, N), dtype=int)

        # Apply pattern to each segment
        for segment_start in range(0, N, segment_length):
            line_positions = decaying_lines_in_segment(
                start=segment_start,
                initial_spacing=initial_spacing,
                decay_rate=decay_rate,
                segment_length=segment_length,
                min_spacing=min_spacing
            )
            for x in line_positions:
                if x < N:
                    F_div_u[:, x] = 1  # vertical lines

        
        F_div_v = np.zeros((N, N)) 
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "control decay", "color": "xkcd:teal blue", 'marker':'P' }
    
    if name == 'cantor':
        """
        def generalized_cantor_line(base=4, remove_segments=[1, 2], depth=5):
            length = base ** depth
            arr = np.ones(length, dtype=int)

            def recurse(start, end, level):
                if level == 0 or end - start < base:
                    return
                seg_len = (end - start) // base
                for r in remove_segments:
                    arr[start + r * seg_len : start + (r + 1) * seg_len] = 0
                for b in range(base):
                    if b not in remove_segments:
                        recurse(start + b * seg_len, start + (b + 1) * seg_len, level - 1)

            recurse(0, length, depth)
            return arr

        def pad_to_size(array, target_size):

            m, n = array.shape
            pad_vert = (target_size - m) // 2
            pad_horz = (target_size - n) // 2

            padded = np.zeros((target_size, target_size), dtype=array.dtype)
            padded[pad_vert:pad_vert + m, pad_horz:pad_horz + n] = array
            return padded
        
        # Parameters
        base = 5
        remove = [1, 2]  # remove middle two quarters
        depth = 4
        size = base ** depth

        # Generate 2D pattern
        cantor_gen = generalized_cantor_line(base, remove, depth)
        print(np.shape(cantor_gen))
        F = np.tile(cantor_gen, (N, 1))  # vertical lines
        
        
        #n = 5  # depth of recursion (3^n total columns)
        #M = 3**n

        # Create 2D array with vertical lines at Cantor positions
        #cantor_1d = cantor_line_pattern(n)
        #F= np.tile(cantor_1d, (N, 1))
        
        F_div_u = pad_to_size(F, N)
        """
        
        def generate_cantor(n_iterations):
            arr = np.array([1], dtype=int)
            for _ in range(n_iterations):
                arr = np.concatenate([arr, np.zeros_like(arr), arr])
            return arr
        
        n=5
        height = N
        
        cantor_1d = generate_cantor(n)
        print(np.shape(cantor_1d))
        cantor_1d_fill = np.zeros(N)
        cantor_1d_fill[N-len(cantor_1d)-10:-10] = cantor_1d
       
        #cantor_1d = np.pad(cantor_1d, ((N-len(cantor_1d))), 'constant', constant_values=0)
        print(np.shape(cantor_1d))
        cantor_2d = np.tile(cantor_1d_fill ,(height, 1))
        
        F_div_u = cantor_2d
        
        F_div_v = np.zeros((N, N)) 
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "control cantor", "color": "xkcd:periwinkle", 'marker':'P' }
        
    
    if name == "fractal_shuffle":
        F_div_u = np.zeros((N, N))
        
        def koch_curve(p1, p2, depth, grid):
            if depth == 0:
                return
    
            # Calculate the points of division
            p3 = (2*p1 + p2) / 3
            p5 = (p1 + 2*p2) / 3
    
            # Calculate the peak point (rotation by 60°)
            angle = np.pi / 3
            vec = p5 - p3
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).dot(vec)
            p4 = p3 + rot
    
            # Draw the segments on the grid
            draw_line(p1, p3, grid)
            draw_line(p3, p4, grid)
            draw_line(p4, p5, grid)
            draw_line(p5, p2, grid)
    
            # Recursively subdivide
            koch_curve(p1, p3, depth-1, grid)
            koch_curve(p3, p4, depth-1, grid)
            koch_curve(p4, p5, depth-1, grid)
            koch_curve(p5, p2, depth-1, grid)

        # Function to draw a line in the grid with 1s
        def draw_line(p1, p2, grid):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            length = int(np.hypot(x2-x1, y2-y1))
            for i in range(length+1):
                t = i / length
                x = int((1-t) * x1 + t * x2)
                y = int((1-t) * y1 + t * y2)
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = 1

        # Main function to generate the fractal
        def generate_snowflake(N, depth):
            grid = np.zeros((N, N))
            size = N // 2
            center = N // 2
    
            # Initial equilateral triangle
            p1 = np.array([center, center - size//2])
            p2 = np.array([center - size//2, center + size//2])
            p3 = np.array([center + size//2, center + size//2])
    
            draw_line(p1, p2, grid)
            draw_line(p2, p3, grid)
            draw_line(p3, p1, grid)
    
            # Apply Koch curve recursively to each segment
            koch_curve(p1, p2, depth, grid)
            koch_curve(p2, p3, depth, grid)
            koch_curve(p3, p1, depth, grid)
    
            return grid

        # Parameters
        N_fractal = int(N/2)
        depth = 4  # Recursion depth

        fractal = generate_snowflake(N_fractal, depth)
        print("SUM BEFORE SHUFFLE", np.sum(fractal))
        #np.random.shuffle(fractal)
        print("SUM AFTER SHUFFLE", np.sum(fractal))
        F_div_u = np.zeros((N,N))
        F_div_u[N_fractal:, N_fractal:] = fractal
    
        F_div_v = np.zeros((N, N))
        F_div_v[N_fractal:, N_fractal:] = fractal
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "fractal_shuffle", "color": "red"}
    
    if name == "fractal+error":
        F_div_u = np.zeros((N, N))
        
        def koch_curve(p1, p2, depth, grid):
            if depth == 0:
                return
    
            # Calculate the points of division
            p3 = (2*p1 + p2) / 3
            p5 = (p1 + 2*p2) / 3
    
            # Calculate the peak point (rotation by 60°)
            angle = np.pi / 3
            vec = p5 - p3
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).dot(vec)
            p4 = p3 + rot
    
            # Draw the segments on the grid
            draw_line(p1, p3, grid)
            draw_line(p3, p4, grid)
            draw_line(p4, p5, grid)
            draw_line(p5, p2, grid)
    
            # Recursively subdivide
            koch_curve(p1, p3, depth-1, grid)
            koch_curve(p3, p4, depth-1, grid)
            koch_curve(p4, p5, depth-1, grid)
            koch_curve(p5, p2, depth-1, grid)

        # Function to draw a line in the grid with 1s
        def draw_line(p1, p2, grid):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            length = int(np.hypot(x2-x1, y2-y1))
            for i in range(length+1):
                t = i / length
                x = int((1-t) * x1 + t * x2)
                y = int((1-t) * y1 + t * y2)
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = 1

        # Main function to generate the fractal
        def generate_snowflake(N, depth):
            grid = np.zeros((N, N))
            size = N // 2
            center = N // 2
    
            # Initial equilateral triangle
            p1 = np.array([center, center - size//2])
            p2 = np.array([center - size//2, center + size//2])
            p3 = np.array([center + size//2, center + size//2])
    
            draw_line(p1, p2, grid)
            draw_line(p2, p3, grid)
            draw_line(p3, p1, grid)
    
            # Apply Koch curve recursively to each segment
            koch_curve(p1, p2, depth, grid)
            koch_curve(p2, p3, depth, grid)
            koch_curve(p3, p1, depth, grid)
    
            return grid

        # Parameters
        N_fractal = int(N/2)
        depth = 4  # Recursion depth

        fractal = generate_snowflake(N_fractal, depth)*mean_intensity/5
        
        F_div_u = np.zeros((N,N))
        F_div_u_n = np.zeros((N,N))
        F_div_u[N_fractal:, N_fractal:] = fractal
    
        F_div_v = np.zeros((N, N))
        F_div_v[N_fractal:, N_fractal:] = fractal
        
        #noise = np.random.randn(N, N)*mean_intensity/10
        spacing_control = 4
        F_div_u_n[:, ::spacing_control] = mean_intensity
        
        F_div_u = (F_div_u + F_div_u_n)
        #F_div_u = (F_div_u + noise)
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10
        #F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "fractal+error", "color": "mediumorchid"}
    
    
    if name == "fractalline01":
        F_div_u = np.zeros((N, N))
        
        def koch_curve(p1, p2, depth, grid):
            if depth == 0:
                return
    
            # Calculate the points of division
            p3 = (2*p1 + p2) / 3
            p5 = (p1 + 2*p2) / 3
    
            # Calculate the peak point (rotation by 60°)
            angle = np.pi / 3
            vec = p5 - p3
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).dot(vec)
            p4 = p3 + rot
    
            # Draw the segments on the grid
            draw_line(p1, p3, grid)
            draw_line(p3, p4, grid)
            draw_line(p4, p5, grid)
            draw_line(p5, p2, grid)
    
            # Recursively subdivide
            koch_curve(p1, p3, depth-1, grid)
            koch_curve(p3, p4, depth-1, grid)
            koch_curve(p4, p5, depth-1, grid)
            koch_curve(p5, p2, depth-1, grid)

        # Function to draw a line in the grid with 1s
        def draw_line(p1, p2, grid):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            length = int(np.hypot(x2-x1, y2-y1))
            for i in range(length+1):
                t = i / length
                x = int((1-t) * x1 + t * x2)
                y = int((1-t) * y1 + t * y2)
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = 1

        # Main function to generate the fractal
        def generate_snowflake(N, depth):
            grid = np.zeros((N, N))
            size = N // 2
            center = N // 2
    
            # Initial equilateral triangle
            p1 = np.array([center, center - size//2])
            p2 = np.array([center - size//2, center + size//2])
            p3 = np.array([center + size//2, center + size//2])
    
            draw_line(p1, p2, grid)
            draw_line(p2, p3, grid)
            draw_line(p3, p1, grid)
    
            # Apply Koch curve recursively to each segment
            koch_curve(p1, p2, depth, grid)
            koch_curve(p2, p3, depth, grid)
            koch_curve(p3, p1, depth, grid)
    
            return grid

        # Parameters
        N_fractal = int(N/2)
        depth = 4  # Recursion depth

        fractal = generate_snowflake(N_fractal, depth)*mean_intensity/10
        
        F_div_u = np.zeros((N,N))
        F_div_u_n = np.zeros((N,N))
        F_div_u[N_fractal:, N_fractal:] = fractal
    
        F_div_v = np.zeros((N, N))
        F_div_v[N_fractal:, N_fractal:] = fractal
        
        #noise = np.random.randn(N, N)*mean_intensity/10
        spacing_control = 4
        F_div_u_n[:, ::spacing_control] = mean_intensity
        
        F_div_u = (F_div_u + F_div_u_n)
        #F_div_u = (F_div_u + noise)
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10
        #F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "fractal+line01", "color": "goldenrod"}
    
    if name == "fractalline11":
        F_div_u = np.zeros((N, N))
        
        def koch_curve(p1, p2, depth, grid):
            if depth == 0:
                return
    
            # Calculate the points of division
            p3 = (2*p1 + p2) / 3
            p5 = (p1 + 2*p2) / 3
    
            # Calculate the peak point (rotation by 60°)
            angle = np.pi / 3
            vec = p5 - p3
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).dot(vec)
            p4 = p3 + rot
    
            # Draw the segments on the grid
            draw_line(p1, p3, grid)
            draw_line(p3, p4, grid)
            draw_line(p4, p5, grid)
            draw_line(p5, p2, grid)
    
            # Recursively subdivide
            koch_curve(p1, p3, depth-1, grid)
            koch_curve(p3, p4, depth-1, grid)
            koch_curve(p4, p5, depth-1, grid)
            koch_curve(p5, p2, depth-1, grid)

        # Function to draw a line in the grid with 1s
        def draw_line(p1, p2, grid):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            length = int(np.hypot(x2-x1, y2-y1))
            for i in range(length+1):
                t = i / length
                x = int((1-t) * x1 + t * x2)
                y = int((1-t) * y1 + t * y2)
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = 1

        # Main function to generate the fractal
        def generate_snowflake(N, depth):
            grid = np.zeros((N, N))
            size = N // 2
            center = N // 2
    
            # Initial equilateral triangle
            p1 = np.array([center, center - size//2])
            p2 = np.array([center - size//2, center + size//2])
            p3 = np.array([center + size//2, center + size//2])
    
            draw_line(p1, p2, grid)
            draw_line(p2, p3, grid)
            draw_line(p3, p1, grid)
    
            # Apply Koch curve recursively to each segment
            koch_curve(p1, p2, depth, grid)
            koch_curve(p2, p3, depth, grid)
            koch_curve(p3, p1, depth, grid)
    
            return grid

        # Parameters
        N_fractal = int(N/2)
        depth = 4  # Recursion depth

        fractal = generate_snowflake(N_fractal, depth)*mean_intensity
        
        F_div_u = np.zeros((N,N))
        F_div_u_n = np.zeros((N,N))
        F_div_u[N_fractal:, N_fractal:] = fractal
    
        F_div_v = np.zeros((N, N))
        F_div_v[N_fractal:, N_fractal:] = fractal
        
        #noise = np.random.randn(N, N)*mean_intensity/10
        spacing_control = 4
        F_div_u_n[:, ::spacing_control] = mean_intensity
        
        F_div_u = (F_div_u + F_div_u_n)
        #F_div_u = (F_div_u + noise)
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10
        #F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "fractal+line11", "color": "gold"}
    
    if name == "fractalline10":
        F_div_u = np.zeros((N, N))
        
        def koch_curve(p1, p2, depth, grid):
            if depth == 0:
                return
    
            # Calculate the points of division
            p3 = (2*p1 + p2) / 3
            p5 = (p1 + 2*p2) / 3
    
            # Calculate the peak point (rotation by 60°)
            angle = np.pi / 3
            vec = p5 - p3
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).dot(vec)
            p4 = p3 + rot
    
            # Draw the segments on the grid
            draw_line(p1, p3, grid)
            draw_line(p3, p4, grid)
            draw_line(p4, p5, grid)
            draw_line(p5, p2, grid)
    
            # Recursively subdivide
            koch_curve(p1, p3, depth-1, grid)
            koch_curve(p3, p4, depth-1, grid)
            koch_curve(p4, p5, depth-1, grid)
            koch_curve(p5, p2, depth-1, grid)

        # Function to draw a line in the grid with 1s
        def draw_line(p1, p2, grid):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            length = int(np.hypot(x2-x1, y2-y1))
            for i in range(length+1):
                t = i / length
                x = int((1-t) * x1 + t * x2)
                y = int((1-t) * y1 + t * y2)
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y] = 1

        # Main function to generate the fractal
        def generate_snowflake(N, depth):
            grid = np.zeros((N, N))
            size = N // 2
            center = N // 2
    
            # Initial equilateral triangle
            p1 = np.array([center, center - size//2])
            p2 = np.array([center - size//2, center + size//2])
            p3 = np.array([center + size//2, center + size//2])
    
            draw_line(p1, p2, grid)
            draw_line(p2, p3, grid)
            draw_line(p3, p1, grid)
    
            # Apply Koch curve recursively to each segment
            koch_curve(p1, p2, depth, grid)
            koch_curve(p2, p3, depth, grid)
            koch_curve(p3, p1, depth, grid)
    
            return grid

        # Parameters
        N_fractal = int(N/2)
        depth = 4  # Recursion depth

        fractal = generate_snowflake(N_fractal, depth)*mean_intensity
        
        F_div_u = np.zeros((N,N))
        F_div_u_n = np.zeros((N,N))
        F_div_u[N_fractal:, N_fractal:] = fractal
    
        F_div_v = np.zeros((N, N))
        F_div_v[N_fractal:, N_fractal:] = fractal
        
        #noise = np.random.randn(N, N)*mean_intensity/10
        spacing_control = 4
        F_div_u_n[:, ::spacing_control] = mean_intensity/10
        
        F_div_u = (F_div_u + F_div_u_n)
        #F_div_u = (F_div_u + noise)
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10
        #F_div_v = (F_div_v + noise2)
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "fractal+line10", "color": "orange"}
    
    
    
    
    
    
    
    
    if name == "control +- err":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        #spacing_control = 21
        
        # ORIGINAL
        #F_div_u[:, spacing_control::spacing_control * 3] = 1*mean_intensity  
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        #F_div_u[:, ::spacing_control * 3] = -1*mean_intensity  
        #spacing_control=3
        
        F_div_u[:, spacing_control::spacing_control * 2] = 1.1*mean_intensity
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_control * 2] = -1*mean_intensity
    
        noise = np.random.randn(N, N)*mean_intensity/10
        F_div_u = F_div_u + noise
        
        noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + noise2
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "control +- err", "color": "tab:blue"}
    
    
    if name == "irregular spacing +- err":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)
        
        #L=1
        #speckle_noise_u = np.random.gamma(L, 1 / L, size=N//spacing_control)
        #gaussian_values = np.round(speckle_noise_u)

        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        noise = np.random.randn(N, N)*mean_intensity/10
        F_div_u = F_div_u + noise
        
        noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + noise2
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular spacing +- err", "color": "tab:purple"}
    
    
    if name == "narrow spacing +- err":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_small::spacing_small*2] = 1 
        #F_div_u[:, ::spacing_small*2] = -1 
        spacing_small = 3
        #F_div_u[:, spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, 2 * spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, ::spacing_small * 3] = -1*mean_intensity  
        
        F_div_u[:, spacing_small::spacing_small * 2] = 1.1*mean_intensity
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_small * 2] = -1*mean_intensity
        
        noise = np.random.randn(N, N)*mean_intensity/10
        #F_div_u = F_div_u + noise
        
        noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = np.zeros((N, N))
        #F_div_v = F_div_v + noise2
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "narrow spacing +- err", "color": "tab:cyan", 'marker': "s"}
    
    if name == "narrow spacing +- err weighted":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_small::spacing_small*2] = 1 
        #F_div_u[:, ::spacing_small*2] = -1 
        spacing_small = 3
        #F_div_u[:, spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, 2 * spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, ::spacing_small * 3] = -1*mean_intensity  
        
        F_div_u[:, spacing_small::spacing_small * 2] = 1.1*mean_intensity
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_small * 2] = -1*mean_intensity
        
        #noise = np.random.randn(N, N)*mean_intensity/10
        #F_div_u = F_div_u + noise
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = np.zeros((N, N))
        #F_div_v = F_div_v + noise2
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "narrow spacing +- err weighted", "color": "dodgerblue", 'marker': "s"}
    
    
    if name == "irregular intensity +- err weighted":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = ((gaussian_values)*mean_intensity) # Only positive values
        
        #L = 1
        #speckle_noise_u = np.random.gamma(L, 1 / L, size=N//spacing_control)
        #gaussian_values = (speckle_noise_u)*mean_intensity
        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N-1, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        #noise = np.random.randn(N, N)*mean_intensity/10 #!!
        #F_div_u = F_div_u + noise #!!
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10 #!!
        F_div_v = np.zeros((N, N))
        #F_div_v = F_div_v + noise2 #!!
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors + weights: SNR           err weighted", "color": "deeppink", "marker": "s"}
    
    if name == "irregular intensity +- err weighted2":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = ((gaussian_values)*mean_intensity) # Only positive values
        
        #L = 1
        #speckle_noise_u = np.random.gamma(L, 1 / L, size=N//spacing_control)
        #gaussian_values = (speckle_noise_u)*mean_intensity
        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N-1, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        #noise = np.random.randn(N, N)*mean_intensity/10 #!!
        #F_div_u = F_div_u + noise #!!
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10 #!!
        F_div_v = np.zeros((N, N))
        #F_div_v = F_div_v + noise2 #!!
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors + weights: full SNR err weighted2", "color": "fuchsia", "marker": "s"}
    
    if name == "irregular intensity +- err":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = ((gaussian_values)*mean_intensity) # Only positive values
        
        #L = 1
        #speckle_noise_u = np.random.gamma(L, 1 / L, size=N//spacing_control)
        #gaussian_values = (speckle_noise_u)*mean_intensity
        
        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N-1, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        #noise = np.random.randn(N, N)*mean_intensity/10 #!!
        #F_div_u = F_div_u + noise #!!
        
        #noise2 = np.random.randn(N, N)*mean_intensity/10 #!!
        F_div_v = np.zeros((N, N))
        #F_div_v = F_div_v + noise2 #!!
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors             err", "color": "tab:pink", "marker": "s"}
    
    
    
    
    
    
    
    
    
    if name == "control +-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        spacing_control = 8

        
        # the ones to use (actual original !!)
        #F_div_u[:, spacing_control::spacing_control * 2] = 1.1*mean_intensity
        #F_div_u[:, ::spacing_control * 2] = -1*mean_intensity
        
        # for having an offset and two lines of thickness
        offset = 4
        F_div_u[:, offset+spacing_control::spacing_control * 2] = 1.1*mean_intensity
        F_div_u[:, offset+spacing_control+1::spacing_control * 2] = 1.1*mean_intensity
        F_div_u[:, offset::spacing_control * 2] = -1*mean_intensity
        F_div_u[:, offset+1::spacing_control * 2] = -1*mean_intensity
    
        F_div_v = np.zeros((N, N))
        
        #F_div_v[:, spacing_control::spacing_control * 2] = 1.1*mean_intensity
        #F_div_v[:, ::spacing_control * 2] = -1*mean_intensity
        
        #F = np.vstack([F_div_u, F_div_v])
        F = np.vstack([F_div_v, F_div_u])
        #return {"F": F, "exp_type": "div", "name": "control +-", "color": "tab:blue", "marker":"o"}
        return {"F": F, "exp_type": "shear", "name": "control +-", "color": "tab:blue", "marker":"o"}

    
    if name == "irregular spacing +-":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = (np.round(gaussian_values)) #make the values round values (either 0 or 1)
        
        #L=1
        #speckle_noise_u = np.random.gamma(L, 1 / L, size=N//spacing_control)
        #gaussian_values = np.round(speckle_noise_u)

        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N-1, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular spacing +-", "color": "tab:purple", "marker":"o"}
    
    
    if name == "narrow spacing +-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_small::spacing_small*2] = 1 
        #F_div_u[:, ::spacing_small*2] = -1 
        spacing_small = 3
        #F_div_u[:, spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, 2 * spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, ::spacing_small * 3] = -1*mean_intensity  
        
        F_div_u[:, spacing_small::spacing_small * 2] = 1.1*mean_intensity
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_small * 2] = -1*mean_intensity
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "narrow spacing +-", "color": "tab:cyan", "marker":"o"}
    
    if name == "narrow spacing ++-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_small::spacing_small*2] = 1 
        #F_div_u[:, ::spacing_small*2] = -1 
        spacing_small = 3
        #F_div_u[:, spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, 2 * spacing_small::spacing_small * 3] = 1*mean_intensity  
        #F_div_u[:, ::spacing_small * 3] = -1*mean_intensity  
        
        F_div_u[:, spacing_small::spacing_small * 2] = 2*mean_intensity
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_small * 2] = -1*mean_intensity
        
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "narrow spacing ++-", "color": "tab:olive", "marker":"o"}
    
    
    if name == "irregular intensity +-":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = ((gaussian_values)*mean_intensity) # Only positive values
        
        #L = 1
        #speckle_noise_u = np.random.gamma(L, 1 / L, size=N//spacing_control)
        #gaussian_values = (speckle_noise_u)*mean_intensity
        
        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N-1, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        #return {"F": F, "exp_type": "div", "name": "irregular intensity +-", "color": "tab:pink", "marker":"o"}
        return {"F": F, "exp_type": "div", "name": "$\\mathbf{\\dot{\\epsilon}_{I}}$≠constant", "color": "tab:pink", "marker":"o"}
    

    if name == "irregular domain +-":
        N = N-1
        
        F_div_u = np.zeros((N, N))
        F_div_u[:, spacing_control::spacing_control * 2] = 1.1*mean_intensity
        F_div_u[:, ::spacing_control * 2] = -1*mean_intensity
    
        F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irregular domain +-", "color": "tab:purple"}
    
    if name == "errors +-":
        F_div_u = np.zeros((N, N))
        #F_div_u[:, spacing_control::spacing_control*2] = 1 
        #F_div_u[:, ::spacing_control*2] = -1 
        #F_div_u[:, spacing_control::spacing_control * 3] = 1*mean_intensity  
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        #F_div_u[:, ::spacing_control * 3] = -1*mean_intensity  
        
        F_div_u[:, spacing_control::spacing_control * 2] = 1*mean_intensity+0.01 
        #F_div_u[:, 2 * spacing_control::spacing_control * 3] = 1*mean_intensity  
        F_div_u[:, ::spacing_control * 2] = -1*mean_intensity-0.01

        noise = np.random.randn(N, N)*mean_intensity/10
        F_div_u = F_div_u + noise
        
        noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = np.zeros((N, N))
        F_div_v = F_div_v + noise2
        
        #F_div_v = np.zeros((N, N))
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "errors +-", "color": "tab:red"}
    
    if name == "irregular intensity errors +-":
        mean = 0.5
        std = 1.0
        gaussian_values = np.random.normal(loc=mean, scale=std, size=N//spacing_control)
        gaussian_values = ((gaussian_values)*mean_intensity) # Only positive values
        
        F_div_u = np.zeros((N, N))
        #F_div_u = np.ones((N, N))*mean_intensity
        for idx, j in enumerate(range(0, N, spacing_control)):
            F_div_u[:, j] = gaussian_values[idx]  # Use a single value per column

        noise = np.random.randn(N, N)*mean_intensity/10
        F_div_u = F_div_u + noise
        
        F_div_v = np.zeros((N, N))
        noise2 = np.random.randn(N, N)*mean_intensity/10
        F_div_v = F_div_v + noise2
        
        F = np.vstack([F_div_u, F_div_v])
        return {"F": F, "exp_type": "div", "name": "irreg int errors +-", "color": "coral"}
    
    
    
    
    
    
    
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