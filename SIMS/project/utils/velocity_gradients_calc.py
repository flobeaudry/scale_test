import numpy as np

def calc_du_dx (u, dx):
    # Compute the velocity gradients based on a forward-difference scheme for du/dx and dv/dy
    du_dx = (u[:, 1:] - u[:, :-1]) / dx  # Gradient of u in x-direction
    return du_dx

def calc_dv_dy (v, dy):
    # Compute the velocity gradients based on a forward-difference scheme for du/dx and dv/dy
    dv_dy = (v[1:, :] - v[:-1, :])/ dy  # Gradient of v in y-direction
    return dv_dy

def calc_dv_dx (v, dx):
    # Compute the velocity gradients based on a central-difference scheme for dv/dx and dv/dx
    # dvdx with periodic in y (vertical), zero in x (horizontal); bottom bndy = Drichlet, and left and right are periodic !
    v_ip1jp1 = np.zeros_like(v)
    v_ip1    = np.zeros_like(v)
    v_im1jp1 = np.zeros_like(v)
    v_im1    = np.zeros_like(v)
    v_ip1jp1[1:-1, :] = np.roll(v, -1, axis=1)[2:, :]   # v[i+1, j+1]
    v_ip1[1:-1, :]    = np.roll(v, -1, axis=1)[1:-1, :] # v[i+1, j]
    v_im1jp1[1:-1, :] = np.roll(v, 1, axis=1)[2:, :]    # v[i-1, j+1]
    v_im1[1:-1, :]    = np.roll(v, 1, axis=1)[1:-1, :]  # v[i-1, j]
    dv_dx = (v_ip1jp1 + v_ip1 - v_im1jp1 - v_im1) / (2 * dx)
    return dv_dx

def calc_du_dy (u, dy):
    # !! to verif
    
    # Compute the velocity gradients based on a central-difference scheme for dv/dx and dv/dx
    # dudy with periodic in x (horizontal), zero in y (vertical); left bndy = Drichlet, and bottom and top are periodic !
    # to verif !!
    u_ip1jp1 = np.zeros_like(u)
    u_ijp1   = np.zeros_like(u)
    u_ip1jm1 = np.zeros_like(u)
    u_ijm1   = np.zeros_like(u)
    u_ip1jp1[:, 1:-1] = np.roll(u[:, :-2], -1, axis=0)   # u[i+1, j+1]
    u_ijp1[:, 1:-1]   = u[:, 2:]                         # u[i, j+1]
    u_ip1jm1[:, 1:-1] = np.roll(u[:, 2:], -1, axis=0)    # u[i+1, j-1]
    u_ijm1[:, 1:-1]   = u[:, :-2]                        # u[i, j-1]
    # Handle periodicity in x (axis=0)
    u_ip1jp1[-1, 1:-1] = u[0, :-2]
    u_ip1jm1[-1, 1:-1] = u[0, 2:]
    du_dy = (u_ip1jp1 + u_ijp1 - u_ip1jm1 - u_ijm1) / (2 * dy)
    return du_dy
