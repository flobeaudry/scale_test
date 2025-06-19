import numpy as np
import pickle
import math
from matplotlib.colors import to_rgb, to_hex
from scipy.ndimage import rotate


def draw_line(array, x0, y0, x1, y1, intensity = 0.1):
    """Draw a line in the array using Bresenham’s algorithm."""
    #x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= x0 < array.shape[0] and 0 <= y0 < array.shape[1]:
            array[x0, y0] = intensity
            print(intensity)
            #array[y0, x0] = 1*intensity
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
            
def draw_simple_tree(array, x, y, angle, length, depth):
    if depth == 0 or length < 1:
        return

    # Compute end point
    x_end = x + length * math.sin(angle)
    y_end = y + length * math.cos(angle)

    draw_line(array, x, y, x_end, y_end)

    # Recurse with half the length and ±15° (30° total between branches)
    #angle = 15 #half angle to get 30 degrees between baby branches
    #angle = 22.5
    #angle_babies = 22.5
    angle_babies = 15
    new_length = length*0.5
    draw_simple_tree(array, x_end, y_end, angle - math.radians(angle_babies), new_length, depth - 1)
    draw_simple_tree(array, x_end, y_end, angle + math.radians(angle_babies), new_length, depth - 1)

def generate_simple_tree(N=200, depth=5):
    array = np.zeros((N, N), dtype=int)
    start_x = N - 1
    start_y = N // 2
    initial_length = int(N / 2)
    draw_simple_tree(array, start_x, start_y, -math.pi / 2, initial_length, depth)
    return array





def draw_radial_tree(array, x, y, angle, length, depth, branches_per_level=4, angle_spread=math.radians(30), current_depth=0):
    if depth == 0 or length < 1:
        return
    
    #intensity = 0.5 ** (current_depth)  # Halve the intensity each level deeper
    intensity = 2 ** current_depth
    #print(current_depth)
    angles = np.linspace(-angle_spread / 2, angle_spread / 2, branches_per_level)

    for da in angles:
        new_angle = angle + da
        x_end = x + length * math.sin(new_angle)
        y_end = y + length * math.cos(new_angle)
        draw_line(array, x, y, x_end, y_end, intensity = intensity)
        draw_radial_tree(array, x_end, y_end, new_angle, length * 0.7, depth - 1, branches_per_level, angle_spread, current_depth+1)


    #angles = np.linspace(0, 2 * np.pi, branches_per_level, endpoint=False)
    #spread = math.radians(60)  # total angular spread in radians
    
    #branch_offsets = [-math.radians(15), math.radians(15)]
    #angles = [angle + offset for offset in branch_offsets]

    #angles = np.linspace(angle - spread / 2, angle + spread / 2, branches_per_level)
    
    #for a in angles:
    #    x_end = x + length * math.sin(a)
    #    y_end = y + length * math.cos(a)
    #    draw_line(array, x, y, x_end, y_end)
    #    #draw_radial_tree(array, x, y, a, length * 0.5, depth - 1, branches_per_level)
    #    draw_radial_tree(array, x_end, y_end, a, length * 0.7, depth - 1, branches_per_level)

def generate_radial_tree_without_ratios(N=200, depth=4, branches_per_level=5):
    array = np.zeros((N, N), dtype=int)
    #center_x = N // 2
    #center_y = N // 2
    start_x = N // 2             # vertical center
    start_y = 0                  # far left
    initial_angle = 0            # angle in radians: 0 = pointing right
    initial_length = N // 4
    draw_radial_tree(array, start_x, start_y, -np.pi//2, N // 6, depth, branches_per_level)
    #draw_radial_tree(array, start_x, start_y, initial_angle, initial_length, depth, branches_per_level)
    return array

def generate_radial_tree_too_full(N=256, fill_fraction=0.1, max_trees=1000, depth=5, branches_per_level=3):
    array = np.zeros((N, N), dtype=int)
    center_x, center_y = N // 2, N // 2
    radius = N // 3
    trees_drawn = 0
    target_pixels = int(N * N * fill_fraction)

    while np.count_nonzero(array) < target_pixels and trees_drawn < max_trees:
        angle = np.random.uniform(0, 2 * np.pi)
        x0 = int(center_x + radius * np.cos(angle))
        y0 = int(center_y + radius * np.sin(angle))
        draw_radial_tree(array, x0, y0, angle, N // 10, depth, branches_per_level)
        trees_drawn += 1

    return array
def generate_radial_tree(N=256, fill_fraction=0.02, depth=6, branches_per_level=2):
    array = np.zeros((N, N), dtype=float)
    total_pixels = N * N
    start_x, start_y = N // 2, N // 2
    angle = -np.pi / 2  # start upward

    # draw just one tree and stop if overfilled
    draw_radial_tree(array, start_x, start_y, angle, N // 7, depth, branches_per_level)
    filled_ratio = np.count_nonzero(array) / total_pixels

    if filled_ratio > fill_fraction:
        print(f"filled too much Target: {fill_fraction}, Actual: {filled_ratio:.3f}")

    return array

def generate_full_radial_fracture_field(N=256, fill_fraction=0.1, num_main_branches=50, depth=4, branch_angle_deg=30):
    array = np.zeros((N, N), dtype=int)
    center_x, center_y = N // 2, N // 2
    radius = N // 3
    total_pixels = N * N
    target_pixels = int(total_pixels * fill_fraction)
    drawn = 0
    branch_angle = math.radians(branch_angle_deg)

    i = 0
    while drawn < target_pixels and i < num_main_branches:
        angle = 2 * np.pi * i / num_main_branches
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        draw_radial_tree(
            array, x, y, angle, length=N // 8,
            depth=depth, branches_per_level=3, angle_spread=branch_angle
        )
        drawn = np.count_nonzero(array)
        i += 1

    return array





def draw_fracture(array, x, y, angle, length, depth, angle_spread=15, branch_prob=0.3):
    if depth == 0 or length < 1:
        return
    angle_rad = angle + math.radians(np.random.uniform(-angle_spread, angle_spread))
    x_end = x + length * math.cos(angle_rad)
    y_end = y + length * math.sin(angle_rad)
    draw_line(array, x, y, x_end, y_end)
    # main crack continues
    draw_fracture(array, x_end, y_end, angle, length * 0.9, depth - 1, angle_spread, branch_prob)
    # maybe add a branch
    if np.random.rand() < branch_prob:
        branch_angle = angle + np.random.uniform(-angle_spread, angle_spread)
        draw_fracture(array, x_end, y_end, branch_angle, length * 0.7, depth - 1, angle_spread, branch_prob)

def generate_fracture_field_without_ratios(N=256, num_fractures=10, depth=6):
    array = np.zeros((N, N), dtype=int)
    for _ in range(num_fractures):
        x = np.random.randint(0, N // 10)  # start on left side
        y = np.random.randint(0, N)
        angle = math.radians(0 + np.random.uniform(-10, 10))  # mostly rightward
        draw_fracture(array, x, y, angle, N // 5, depth)
    return array

def generate_fracture_field(N=256, depth=6, fill_fraction=0.1, max_fractures=1000):
    array = np.zeros((N, N), dtype=int)
    fracture_count = 0
    target_pixels = int(N * N * fill_fraction)

    while np.count_nonzero(array) < target_pixels and fracture_count < max_fractures:
        x = np.random.randint(0, N // 10)
        y = np.random.randint(0, N)
        angle = math.radians(np.random.uniform(-10, 10))
        draw_fracture(array, x, y, angle, N // 5, depth)
        fracture_count += 1

    return array


def draw_branch(array, x, y, angle, depth, max_depth, length):
    if depth > max_depth or length < 1:
        return

    # Compute end point of the branch
    x_end = x + length * math.sin(angle)
    y_end = y + length * math.cos(angle)

    draw_line(array, x, y, x_end, y_end)

    # Recursive branches
    draw_branch(array, x_end, y_end, angle - math.pi / 6, depth + 1, max_depth, length * 0.9)
    draw_branch(array, x_end, y_end, angle + math.pi / 6, depth + 1, max_depth, length * 0.9)

def generate_fractal_tree(N, depth=4):
    array = np.zeros((N, N), dtype=int)
    trunk_length = N / 6
    start_x = N - 1
    start_y = N // 2
    angle_offset = math.pi/16
    draw_branch(array, start_x, start_y, -math.pi / 2 - angle_offset, 0, depth, trunk_length)
    return array