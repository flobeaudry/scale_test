import numpy as np
import os
from datetime import datetime, timedelta


def save_fields(u, v, out, start_date, end_date):
    """
        Function that saves u, v fields to a given output file (needs to be outputxx; xx between 0 and 99) for specified "fake times".
        For now, the u and v don't change in time, but need to adapt the function for that
        
        Args:
            u (np.ndarray): u velocity field (ny, nx+1)
            v (np.ndarray): v velocity field (ny+1, nx)
            out (int): two number int; the experiment number to be saved, being outputxx     
            start_date, end_date (datetime): start and end time of the form datetime(yyyy, mm, dd)
    """
    
    time_delta = timedelta(hours=6)
    time_steps = int((end_date - start_date).total_seconds() // 3600 // 6) + 1
    
    # Where to put the files
    output_dir = f"/aos/home/fbeaudry/git/scale_test/output{out}"
    os.makedirs(output_dir, exist_ok=True)

    u = np.pad(u, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=0)
    v = np.pad(v, pad_width=((0, 1), (0, 0)), mode='constant', constant_values=0)
    
    # Create and save the fields over time
    current_time = start_date
    for t in range(time_steps):
        # Add a small noize factor
        #u_n = u+0.05*np.random.rand(N,N+1)
        #v_n = v+0.05*np.random.rand(N+1,N)
        
        u_n = u
        v_n = v
    
        # Filenames gossage
        file_suffix = f"{current_time.strftime('%Y_%m_%d_%H_%M')}.{out}"
        u_filename = os.path.join(output_dir, f"u{file_suffix}")
        v_filename = os.path.join(output_dir, f"v{file_suffix}")

        # Save the u, v files
        np.savetxt(u_filename, u_n, fmt='%.6f')
        np.savetxt(v_filename, v_n, fmt='%.6f')

        # Update the time
        current_time += time_delta