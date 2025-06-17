import numpy as np

def calc_du_dx (u, dx):
    # Compute the velocity gradients based on a forward-difference scheme for du/dx and dv/dy
    du_dx = (u[:, 1:] - u[:, :-1]) / dx  # Gradient of u in x-direction
    du_dx = np.pad(du_dx, ((0, 0), (1, 0)), mode='constant')  # 1 col on the left
    return du_dx

def calc_dv_dy (v, dy):
    # Compute the velocity gradients based on a forward-difference scheme for du/dx and dv/dy
    dv_dy = (v[1:, :] - v[:-1, :])/ dy  # Gradient of v in y-direction
    dv_dy = np.pad(dv_dy, ((0, 1), (0, 0)), mode='constant')  # 1 row at the bottom
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
    # Compute the velocity gradients based on a central-difference scheme for dv/dx and dv/dx
    # dudy with periodic in x (horizontal), zero in y (vertical); left bndy = Drichlet, and bottom and top are periodic !
    u_ip1jp1 = np.zeros_like(u)
    u_ijp1   = np.zeros_like(u)
    u_ip1jm1 = np.zeros_like(u)
    u_ijm1   = np.zeros_like(u)
    u_ip1jp1[:, 1:-1] = np.roll(u, -1, axis=0)[:, 2:]   # u[i+1, j+1]
    u_ijp1[:, 1:-1]    = np.roll(u, -1, axis=0)[:, 1:-1] # u[i+1, j]
    u_ip1jm1[:, 1:-1] = np.roll(u, 1, axis=0)[:, 2:]    # u[i-1, j+1]
    u_ijm1[:, 1:-1]    = np.roll(u, 1, axis=0)[:, 1:-1]  # u[i-1, j]
    du_dy = (u_ip1jp1 + u_ijp1 - u_ip1jm1 - u_ijm1) / (2 * dy)
    return du_dy

def calc_div(u, v, dx=1, dy=1):
    du_dx = calc_du_dx(u, dx)
    dv_dy = calc_dv_dy(v, dy)
    div = du_dx + dv_dy
    div = div[1:-1, 1:-1]
    return div

def calc_shear(u, v, dx=1, dy=1):
    du_dx = calc_du_dx(u, dx)
    dv_dy = calc_dv_dy(v, dy)
    dv_dx = calc_dv_dx(v, dx)
    du_dy = calc_du_dy(u, dy)
    shear = np.sqrt((du_dx - dv_dy)**2 + (du_dy + dv_dx)**2) 
    shear = shear[1:-1, 1:-1]
    return shear

def calc_shear_components(u, v, dx=1, dy=1):
    dv_dx = calc_dv_dx(v, dx)
    du_dy = calc_du_dy(u, dy)
    shear = (du_dy + dv_dx)
    shear = shear[1:-1, 1:-1]
    return shear

def calc_tot_defo(u, v, dx=1, dy=1):
    div = calc_div(u, v, dx, dy)
    shear = calc_shear(u, v, dx, dy)
    tot_defo = np.sqrt(div**2 + shear**2)
    tot_defo = tot_defo[1:-1, 1:-1]
    return tot_defo

def calc_tot_defo_components(u, v, dx=1, dy=1):
    div = calc_div(u, v, dx, dy)
    shear = calc_shear_components(u, v, dx, dy)
    tot_defo = np.sqrt(div**2 + shear**2)
    tot_defo = tot_defo[1:-1, 1:-1]
    return tot_defo