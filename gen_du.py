import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime, timedelta

# Initialize the grid
nx, ny = 500, 500  # Number of grid points
dx, dy = 1.0, 1.0    # Grid spatial resolution
divergence_rate = 5  # Divergence rate

num_plates = 20
plate_width = 20

# Initialize positions of plates on the left side
initial_x = np.linspace(0, plate_width, num_plates)
plate_y = ny // 2  # Middle of the grid for y position

# Define random dimensions for each plate
np.random.seed(0)  # For reproducibility
plate_widths = np.random.randint(3, 10, size=num_plates)
plate_heights = np.random.randint(3, 10, size=num_plates)

# Function to create the ice field and velocity field with divergence in the x-direction only
def create_fields(t):
    u = np.zeros((ny, nx+1))
    v = np.zeros((ny+1, nx))
    
    for i in range(num_plates):
        x_divergence = divergence_rate * t * i
        plate_width = plate_widths[i]
        plate_height = plate_heights[i]
        
        x_start = int(initial_x[i] + x_divergence)
        x_end = int(initial_x[i] + x_divergence + plate_width)
        y_start = plate_y - plate_height // 2
        y_end = plate_y + plate_height // 2

        # Ensure the indices are within bounds
        x_start = max(x_start, 0)
        x_end = min(x_end, nx)
        y_start = max(y_start, 0)
        y_end = min(y_end, ny)

        # Set initial velocity field based on divergence (u component only)
        u[y_start:y_end, x_start:x_end] = divergence_rate * t

    return u, v

# Time settings
start_date = datetime(2002, 1, 1)
end_date = datetime(2002, 1, 31, 18)  # January 31st, 18:00
time_delta = timedelta(hours=6)

# Directory to save the files
#output_dir = "output10"
output_dir = "output11"
os.makedirs(output_dir, exist_ok=True)

current_time = start_date
while current_time <= end_date:
    time_step = int((current_time - start_date).total_seconds() / 3600)  # Convert to hours
    u, v = create_fields(time_step)

    # Formatting filename
    file_suffix = current_time.strftime("%Y_%m_%d_%H_%M") + ".11"
    u_filename = os.path.join(output_dir, f"u{file_suffix}")
    v_filename = os.path.join(output_dir, f"v{file_suffix}")

    ## Save the fields as .npy files
    #np.save(u_filename, u)
    #np.save(v_filename, v)
    # Save the fields with .10 extension
    #u.astype(np.float32).tofile(u_filename)
    #v.astype(np.float32).tofile(v_filename)
    
    
    np.savetxt(u_filename, np.random.rand(ny, nx+1), fmt='%.6f')
    np.savetxt(v_filename, np.random.rand(ny+1, nx), fmt='%.6f')
    #np.savetxt(u_filename, u+np.random.rand(ny, nx+1), fmt='%.6f')
    #np.savetxt(v_filename, v+np.random.rand(ny+1, nx), fmt='%.6f')

    # Increment time by 6 hours
    current_time += time_delta

'''

# Initialize the grid
nx, ny = 100, 100 # Number of grid points
dx, dy = 1.0, 1.0 # Grid spatial resolution
time_steps = 30  # Number of days
dt = 1 # Increment (in days)
divergence_rate = 3

num_plates = 10
plate_width = 50


# Initialize the ice field and store through time
ice_field_over_time = np.zeros((ny, nx, time_steps))
u_over_time = np.zeros((ny, nx, time_steps))
v_over_time = np.zeros((ny, nx, time_steps))

# Define random dimensions for each plate
np.random.seed(0)  # For reproducibility
plate_widths = np.random.randint(3, 10, size=num_plates)
plate_heights = np.random.randint(3, 10, size=num_plates)

# Initialize positions of plates on the left side
initial_x = np.linspace(0, plate_widths[0], num_plates)
plate_y = ny // 2  # Middle of the grid for y position

# Function to create the ice field and velocity field with divergence in the x-direction only at time t
def update_fields(t):
    ice_field = np.zeros((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    
    # Apply divergence to each plate
    for i in range(num_plates):
        # Divergence in the x direction only
        x_divergence = divergence_rate * t * i
        plate_width = plate_widths[i]
        plate_height = plate_heights[i]
        
        x_start = int(initial_x[i] + x_divergence)
        x_end = int(initial_x[i] + x_divergence + plate_width)
        y_start = plate_y - plate_height // 2
        y_end = plate_y + plate_height // 2

        # Ensure the indices are within bounds
        x_start = max(x_start, 0)
        x_end = min(x_end, nx)
        y_start = max(y_start, 0)
        y_end = min(y_end, ny)

        ice_field[y_start:y_end, x_start:x_end] = 1
        
        # Set initial velocity field based on divergence (u component only for simplicity)
        u[y_start:y_end, x_start:x_end] = divergence_rate * t
        
    return ice_field, u, v

# Update the fields over time
for t in range(time_steps):
    ice_field, u, v = update_fields(t)
    ice_field_over_time[:, :, t] = ice_field
    u_over_time[:, :, t] = u
    v_over_time[:, :, t] = v
    
    
# Compute derivatives
def compute_derivatives(u, v):
    dudx = np.diff(u, axis=1) / dx
    dudy = np.diff(u, axis=0) / dy
    dvdx = np.diff(v, axis=1) / dx
    dvdy = np.diff(v, axis=0) / dy
    
    # Pad the derivatives to ensure they have consistent shapes
    dudx = np.pad(dudx, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    dudy = np.pad(dudy, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    dvdx = np.pad(dvdx, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    dvdy = np.pad(dvdy, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    
    return dudx, dudy, dvdx, dvdy

# Compute derivatives for each time step
dudx_over_time = np.zeros((ny, nx, time_steps))
dudy_over_time = np.zeros((ny, nx, time_steps))
dvdx_over_time = np.zeros((ny, nx, time_steps))
dvdy_over_time = np.zeros((ny, nx, time_steps))

for t in range(time_steps):
    u = u_over_time[:, :, t]
    v = v_over_time[:, :, t]
    dudx, dudy, dvdx, dvdy = compute_derivatives(u, v)
    dudx_over_time[:, :, t] = dudx
    dudy_over_time[:, :, t] = dudy
    dvdx_over_time[:, :, t] = dvdx
    dvdy_over_time[:, :, t] = dvdy

# Save the velocity derivatives
np.save("../artificial_fields/DUDX.npy", dudx_over_time)
np.save("../artificial_fields/DUDY.npy", dudy_over_time)
np.save("../artificial_fields/DVDX.npy", dvdx_over_time)
np.save("../artificial_fields/DVDY.npy", dvdy_over_time)
'''
'''

# Compute derivatives
def compute_derivatives(u, v):
    dudx = np.zeros((ny, nx-1))
    dudy = np.zeros((ny-1, nx))
    dvdx = np.zeros((ny, nx-1))
    dvdy = np.zeros((ny-1, nx))
    
    dudx[:, :] = np.diff(u, axis=1) / dx
    dudy[:, :] = np.diff(u, axis=0) / dy
    dvdx[:, :] = np.diff(v, axis=1) / dx
    dvdy[:, :] = np.diff(v, axis=0) / dy
    
    return dudx, dudy, dvdx, dvdy

# Compute derivatives for each time step
dudx_over_time = np.zeros((ny, nx-1, time_steps))
dudy_over_time = np.zeros((ny-1, nx, time_steps))
dvdx_over_time = np.zeros((ny, nx-1, time_steps))
dvdy_over_time = np.zeros((ny-1, nx, time_steps))

for t in range(time_steps):
    u = u_over_time[:, :, t]
    v = v_over_time[:, :, t]
    dudx, dudy, dvdx, dvdy = compute_derivatives(u, v)
    dudx_over_time[:, :, t] = dudx
    dudy_over_time[:, :, t] = dudy
    dvdx_over_time[:, :, t] = dvdx
    dvdy_over_time[:, :, t] = dvdy

# Visualization using animation for one of the derivatives (e.g., dudx)
fig, ax = plt.subplots()
def animate(t):
    ax.clear()
    ax.imshow(dudx_over_time[:, :, t], cmap='coolwarm', origin='lower', vmin=-0.1, vmax=0.1)
    ax.set_title(f'dudx at Time Step {t}')

ani = animation.FuncAnimation(fig, animate, frames=time_steps, repeat=False)
plt.show()

# save the velocity derivatives
np.save("artificial_fields/DUDX.npy", dudx_over_time)
np.save("artificial_fields/DUDY.npy", dudy_over_time)
np.save("artificial_fields/DVDX.npy", dvdx_over_time)
np.save("artificial_fields/DVDY.npy", dvdy_over_time)
'''



'''
# Initialize the ice field (Ice = 1; No ice = 0)
ice_field_over_time = np.zeros((ny,nx, time_steps))
#center_x, center_y = nx // 2, ny // 2

# Define random dimensions for each plate
np.random.seed(0)  # For reproducibility
plate_widths = np.random.randint(5, 20, size=num_plates)
plate_heights = np.random.randint(50, 1000, size=num_plates)

# Initialize positions of plates on the left side
initial_x = np.linspace(0, plate_widths[0], num_plates)
plate_y = ny // 2  # Middle of the grid for y position

# Function to create the ice field with divergence in the x-direction only at time t
def update_ice_field(t):
    ice_field = np.zeros((ny, nx))
    
    # Apply divergence to each plate
    for i in range(num_plates):
        # Divergence in the x direction only
        x_divergence = divergence_rate * t * i
        
        plate_width = plate_widths[i]
        plate_height = plate_heights[i]
        
        x_start = int(initial_x[i] + x_divergence)
        x_end = int(initial_x[i] + x_divergence + plate_width)
        y_start = plate_y - plate_height // 2
        y_end = plate_y + plate_height // 2

        # Ensure the indices are within bounds
        x_start = max(x_start, 0)
        x_end = min(x_end, nx)
        y_start = max(y_start, 0)
        y_end = min(y_end, ny)

        ice_field[y_start:y_end, x_start:x_end] = 1
    
    return ice_field

# Update the ice field over time
for t in range(time_steps):
    ice_field_over_time[:, :, t] = update_ice_field(t)

# Visualization using animation
fig, ax = plt.subplots()
def animate(t):
    ax.clear()
    ax.imshow(ice_field_over_time[:, :, t], cmap='Blues', origin='lower')
    ax.set_title(f'Ice Field at Time Step {t}')

ani = animation.FuncAnimation(fig, animate, frames=time_steps, repeat=False)
plt.show()
'''

'''
# Create the plates of ice at the center of the grid (can be moved)
ice_field[center_y - plate_width: center_y + plate_width, center_x - plate_width: center_x + plate_width] = 1


# Initialize the velocity field
u = np.zeros((ny, nx, time_steps))
v = np.zeros((ny, nx, time_steps))


# Apply divergence to the plates
for t in range(time_steps):
    divergence = np.linspace(-divergence_rate * t, divergence_rate * t, 2 * plate_width)
    
    u[center_y - plate_width:center_y + plate_width, center_x - plate_width:center_x + plate_width, t] = (np.outer(np.ones(2 * plate_width), divergence))
    
# Compute velocity derivatives
dudx = np.gradient(u, axis=1) / dx
dudy = np.gradient(u, axis=0) / dy
dvdx = np.gradient(v, axis=1) / dx
dvdy = np.gradient(v, axis=0) / dy


# Visualization (optional)
plt.imshow(ice_field, cmap='Blues', origin='lower')
plt.title('Initial Ice Field')
plt.colorbar(label='Ice presence (1: Ice, 0: No ice)')
plt.show()
'''