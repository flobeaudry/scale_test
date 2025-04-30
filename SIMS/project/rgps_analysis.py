import numpy as np
import os

file_path = "rgps_export.npy"

file_path = "RGPS_derivatives/DUDX.npy"

# Load the .npy file
data = np.load(file_path)

# Check the shape and type of the loaded data
print(data.shape)  # Example: (1000, 1000)
print(np.nansum(data))
