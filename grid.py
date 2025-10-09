# file: grid.py

import numpy as np
import matplotlib.pyplot as plt
from bspline_module import generate_knots_and_colloc_pts

def create_channel_grid(settings):
    """
    Generates the 2D or 3D computational grid for the channel flow problem.

    This function sets up the physical domain and creates the collocation points
    for the hybrid discretization (B-spline in x,y; Fourier in z).

    Args:
        settings (dict): A dictionary containing all simulation parameters.

    Returns:
        dict: A dictionary containing all necessary grid information. For 3D,
              this includes 'z_colloc', 'Z', 'Nz', 'Lz'.
    """
    # Unpack common parameters
    Nx = settings['NX']
    Ny = settings['NY']
    Lx = settings['DOMAIN_LENGTH_X']
    H = settings['DOMAIN_HEIGHT_H']
    p_degree = settings['P_DEGREE'] # Note: Using same degree for x and y for simplicity
    q_degree = settings['Q_DEGREE']
    y_stretch = settings['Y_STRETCH_FACTOR']

    # Check if this is a 3D simulation
    is_3d = 'NZ' in settings and settings['NZ'] > 1

    print("--- Creating Computational Grid ---")
    if is_3d:
        print("Mode: 3D Grid Generation")
    else:
        print("Mode: 2D Grid Generation")

    # --- X-Direction (Wall-Bounded B-spline) ---
    # For a general channel, inlet/outlet are not periodic, so we use clamped B-splines.
    xmin, xmax = 0, Lx
    _x_knots, x_colloc = generate_knots_and_colloc_pts(
        p=p_degree,
        num_basis=Nx,
        xmin=xmin,
        xmax=xmax,
        stretch_factor=0.0  # Uniform grid in x
    )
    print(f"Generated {len(x_colloc)} B-spline collocation points in x-direction.")

    # --- Y-Direction (Wall-Bounded and Stretched B-spline) ---
    ymin, ymax = -H, H
    _y_knots, y_colloc = generate_knots_and_colloc_pts(
        p=q_degree,
        num_basis=Ny,
        xmin=ymin,
        xmax=ymax,
        stretch_factor=y_stretch
    )
    print(f"Generated {len(y_colloc)} stretched B-spline collocation points in y-direction.")

    # --- Z-Direction (Periodic Fourier) ---
    if is_3d:
        Nz = settings['NZ']
        Lz = settings['DOMAIN_LENGTH_Z']
        # For periodic Fourier methods, we need a uniform grid.
        # np.linspace with endpoint=False is perfect for this.
        z_colloc = np.linspace(0, Lz, Nz, endpoint=False)
        print(f"Generated {len(z_colloc)} uniform collocation points in z-direction (periodic).")
    else:
        # For a 2D case, create dummy z-arrays for consistency
        Nz = 1
        Lz = 0.0
        z_colloc = np.array([0.0])

    # --- Create Meshgrid ---
    if is_3d:
        X, Y, Z = np.meshgrid(x_colloc, y_colloc, z_colloc, indexing='ij')
    else:
        X, Y = np.meshgrid(x_colloc, y_colloc, indexing='ij')
        Z = None # No Z grid for 2D

    # --- Identify Boundary Indices ---
    wall_indices_y = (0, Ny - 1)  # bottom, top
    in_out_indices_x = (0, Nx - 1) # inlet, outlet

    print("Grid generation complete.")

    # --- Package all grid data into a dictionary ---
    grid_data = {
        'x_colloc': x_colloc,
        'y_colloc': y_colloc,
        'z_colloc': z_colloc,
        'X': X,
        'Y': Y,
        'Z': Z,
        'wall_indices_y': wall_indices_y,
        'in_out_indices_x': in_out_indices_x,
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'Lx': Lx, 'H': H, 'Lz': Lz,
        'is_3d': is_3d
    }

    return grid_data

# --- Main execution block for testing ---
if __name__ == '__main__':
    # --- Test Case 1: 2D Grid ---
    print("\n" + "="*50)
    print("Testing 2D grid generation...")
    print("="*50)
    settings_2d = {
        'NX': 300, 'NY': 100,
        'DOMAIN_LENGTH_X': 4.0, 'DOMAIN_HEIGHT_H': 1.0,
        'P_DEGREE': 4, 'Q_DEGREE': 4,
        'Y_STRETCH_FACTOR': 2.5
    }
    grid_2d = create_channel_grid(settings_2d)
    
    # Verification
    print("\n--- 2D Verification ---")
    print(f"Shape of X grid: {grid_2d['X'].shape} (Expected: ({settings_2d['NX']}, {settings_2d['NY']}))")
    assert grid_2d['is_3d'] is False

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.title("Generated 2D Channel Grid")
    plt.plot(grid_2d['X'], grid_2d['Y'], 'ko', markersize=2)
    plt.plot(grid_2d['X'], grid_2d['Y'], 'k-', linewidth=0.5)
    plt.plot(grid_2d['X'].T, grid_2d['Y'].T, 'k-', linewidth=0.5)
    plt.xlabel("x (B-spline Direction)")
    plt.ylabel("y (B-spline Direction)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # --- Test Case 2: 3D Grid ---
    print("\n" + "="*50)
    print("Testing 3D grid generation...")
    print("="*50)
    settings_3d = {
        'NX': 100, 'NY': 50, 'NZ': 16, # Smaller for faster testing
        'DOMAIN_LENGTH_X': 4.0, 'DOMAIN_HEIGHT_H': 1.0, 'DOMAIN_LENGTH_Z': 2.0,
        'P_DEGREE': 4, 'Q_DEGREE': 4,
        'Y_STRETCH_FACTOR': 2.0
    }
    grid_3d = create_channel_grid(settings_3d)

    # Verification
    print("\n--- 3D Verification ---")
    print(f"Shape of X grid: {grid_3d['X'].shape} (Expected: ({settings_3d['NX']}, {settings_3d['NY']}, {settings_3d['NZ']}))")
    assert grid_3d['is_3d'] is True
    
    # Visualization (3D scatter plot of points)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_3d['X'], grid_3d['Y'], grid_3d['Z'], c='k', marker='.')
    ax.set_title("Generated 3D Channel Grid Points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (Periodic Direction)")
    # Set aspect ratio to be equal
    ax.set_box_aspect([grid_3d['Lx'], 2*grid_3d['H'], grid_3d['Lz']])
    plt.show()
