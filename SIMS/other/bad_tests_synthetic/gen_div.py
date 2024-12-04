import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime, timedelta

"""
# Initialize the grid
nx, ny = 500, 500 # Number of grid points
    # In the "scale_test"/"SIM-plots" codes, the grids need to be ny, nx
    # Also the u field must have (ny, nx+1) shape; v field must have (ny+1, nx) 
dx, dy = 1.0, 1.0 # Spatial resolution of grid

# Initialize the parameters
base_divergence = -5e-6 # Base divergence to make the div evolve in time
constant_divergence = 1e-4 # Constant divergence in time
noise_amplitude = 0.05 # Amplitude of the random background noise

# Simulation parameters
# Directory to save the files
output_dir = "output12"
os.makedirs(output_dir, exist_ok=True)
# To match with the RDPS data we have
start_date = datetime(2002, 1, 1)
end_date = datetime(2002, 1, 31, 18)
time_delta = timedelta(hours=6)

current_time = start_date
u_fields = []
v_fields = []

def create_u_div_fields(t):
    divergence_rate = base_divergence * t
    #divergence_rate = constant_divergence
    
    # Create the u field
    # Diverge linearly from 0 to the max speed delta
    u = np.linspace(0, divergence_rate *nx, nx+1) # (nx+1,)
    # Reshape (makes column) and add the y component 
        # Basically copy the diverging line for all the y components ("clones")
    u = u.reshape(1,nx+1) * np.ones((ny, 1)) # (ny, nx+1)
    # Add the noise
    u += np.random.randn(ny, nx+1) * noise_amplitude
    
    # Create the v field (only noise)
    v = np.random.randn(ny+1, nx) * noise_amplitude # (ny+1, nx)
    
    return u, v

# Loop over the times needed for file generation
while current_time <= end_date:
    time_step = int((current_time - start_date).total_seconds() / 3600)  # Convert to hours
    u, v = create_u_div_fields(time_step)

    # Store fields for plotting
    u_fields.append(u)
    v_fields.append(v)

    # Formatting and saving the u, v files
    file_suffix = current_time.strftime("%Y_%m_%d_%H_%M") + ".12"
    u_filename = os.path.join(output_dir, f"u{file_suffix}")
    v_filename = os.path.join(output_dir, f"v{file_suffix}")
    np.savetxt(u_filename, u, fmt='%.6f')
    np.savetxt(v_filename, v, fmt='%.6f')

    current_time += time_delta

u_fields = np.array(u_fields)
v_fields = np.array(v_fields)



# Function to plot the fields at different time steps
def plot_fields(u_fields, v_fields, time_steps):
    fig, axes = plt.subplots(len(time_steps), 2, figsize=(12, len(time_steps) * 5))

    for i, t in enumerate(time_steps):
        u = u_fields[t]
        v = v_fields[t]

        # Plot u field
        ax_u = axes[i, 0]
        c_u = ax_u.imshow(u, cmap='seismic', origin='lower')
        ax_u.set_title(f'u Field at Time Step {t}')
        fig.colorbar(c_u, ax=ax_u)

        # Plot v field
        ax_v = axes[i, 1]
        c_v = ax_v.imshow(v, cmap='seismic', origin='lower')
        ax_v.set_title(f'v Field at Time Step {t}')
        fig.colorbar(c_v, ax=ax_v)

    plt.tight_layout()
    plt.show()

# Plot the fields at selected time steps
time_steps_to_plot = [0, len(u_fields)//4, len(u_fields)//2, len(u_fields)-1]  # Start, quarter, half, and end
#plot_fields(u_fields, v_fields, time_steps_to_plot)

print(np.shape(u_fields))

#fig, ax = plt.subplots()
#cax = ax.imshow(u_fields[0, :, :], cmap='turbo', origin='lower', vmin=0, vmax=1)
#fig.colorbar(cax, ax=ax, orientation='vertical', label='Velocity (u)')
def animate(t):
    ax.clear()
    cax = ax.imshow(u_fields[t,:,:], cmap='turbo', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'U field at Time Step {t}')

#ani = animation.FuncAnimation(fig, animate, frames=len(u_fields[:,0,0]), repeat=False)
#ani.save('linear_u_div.gif', writer='Pillow', fps=10)
#plt.show()

"""

# Initialize the grid
nx, ny = 500, 500  # Number of grid points
dx, dy = 1.0, 1.0    # Grid spatial resolution
convergence_rate = 0.5  # Convergence rate

start_date = datetime(2002, 1, 1)
#end_date = datetime(2002, 1, 10, 18) # Testing end date
end_date = datetime(2002, 1, 31, 18) # Full end date

time_delta = timedelta(hours=6)
time_steps = int((end_date - start_date).total_seconds() // 3600 // 6) + 1

# Where to put the files
output_dir = "/aos/home/fbeaudry/git/scale_test/output12"
os.makedirs(output_dir, exist_ok=True)

# Initialize u, v fields (over time)
u_fields = np.zeros((time_steps, ny, nx+1))
v_fields = np.zeros((time_steps, ny+1, nx))


# Center of the grid
center_x = nx // 2

# Function to create the velocity fields with convergence in the x-direction only
def create_fields(t):
    u = np.zeros((ny, nx+1))
    v = np.zeros((ny+1, nx))
    
    # Move all points towards the center as time progresses
    for i in range(nx):
        distance_from_center = i - center_x
        #u[:, i] = -convergence_rate * t * distance_from_center / nx
        u[:, i] = convergence_rate * t * distance_from_center / nx
            
        # Add random noise (small)
        u[:, i] += np.random.uniform(-0.01, 0.01)
        v[:, i] += np.random.uniform(-0.01, 0.01)
      
    """      
    # Move all points towards the center as time progresses
    for i in range(nx):
        for j in range(ny):
            distance_from_center = i - center_x
            #u[j, i] = -convergence_rate * t * distance_from_center / nx
            u[j, i] = convergence_rate * t * distance_from_center / nx
            
            # Add random noise (small)
            u[j, i] += np.random.uniform(-0.01, 0.01)
            v[j, i] += np.random.uniform(-0.01, 0.01)
    """
    
    return u, v

# Create and save the fields over time
current_time = start_date
for t in range(time_steps):
    u, v = create_fields(t)
    u_fields[t] = u
    v_fields[t] = v

    # Filenames gossage
    file_suffix = current_time.strftime("%Y_%m_%d_%H_%M") + ".12"
    u_filename = os.path.join(output_dir, f"u{file_suffix}")
    v_filename = os.path.join(output_dir, f"v{file_suffix}")

    # Save the u, v files
    np.savetxt(u_filename, u, fmt='%.6f')
    np.savetxt(v_filename, v, fmt='%.6f')

    current_time += time_delta





fig, ax = plt.subplots()
# Set up the initial plot with the color bar
cax = ax.imshow(u_fields[0, :, :], cmap='coolwarm', origin='lower', vmin=-0.5, vmax=0.5)
fig.colorbar(cax, ax=ax, orientation='vertical', label='Velocity (u)')

def animate(t):
    ax.clear()
    ax.imshow(u_fields[t, :, :], cmap='coolwarm', origin='lower', vmin=-0.5, vmax=0.5)
    #ax.quiver(np.arange(nx+1), np.arange(ny), u_fields[t, :, :], v_fields[t, :, :], color='black', scale=50)
    ax.set_title(f'dudx at Time Step {t}')

ani = animation.FuncAnimation(fig, animate, frames=time_steps, repeat=False)

# Save the animation as a GIF (or another format like MP4 if configured)
#ani.save('Animations/diverging_field_animation_slower_noquiver.gif', writer='Pillow', fps=10)

plt.show()