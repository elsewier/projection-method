
import numpy as np
import matplotlib.pyplot as plt
from bspline_module import generate_knots_and_colloc_pts

def create_channel_grid(Nx, Ny, Nz, Lx, H, Lz, p, q, stretch_factor): # creates 2d by default

    is_3d = Nz > 1

    print("--- Creating Computational Grid ---")
    if is_3d:
        print(f"3D Grid Generation ({Nx}x{Ny}x{Nz})")
    else:
        print(f"2D Grid Generation ({Nx}x{Ny})")

    # x direction
    xmin, xmax = 0, Lx
    x_knots, x_colloc = generate_knots_and_colloc_pts(p, Nx, xmin, xmax, stretch_factor = 0.0)

    # y direction
    ymin, ymax = -H, H
    y_knots, y_colloc = generate_knots_and_colloc_pts(q, Ny, ymin, ymax, stretch_factor = 0.0)

    # z direction
    if is_3d:
        z_colloc = np.linspace(0, Lz, Nz, endpoint = False) # endpoint false excludes the last point
    else:
        Nz = 1
        Lz = 0.0
        z_colloc = np.array([0.0])

    # meshgrid
    if is_3d:
        X, Y, Z = np.meshgrid(x_colloc, y_colloc, z_colloc, indexing='ij') # indexing ij => (len(x), len(y)) if we use xy it will be (len(y), len(x))
    else:
        X, Y = np.meshgrid(x_colloc, y_colloc, indexing='ij')
        Z = None 

    # flagging
    wall_indices_y = (0, Ny - 1)  # bottom, top
    in_out_indices_x = (0, Nx - 1) # inlet, outlet

    print("Grid generation complete.")

    # --- Package all grid data into a dictionary ---
    grid_data = {
        'x_colloc': x_colloc,
        'y_colloc': y_colloc,
        'z_colloc': z_colloc,
        'x_knots': x_knots,
        'y_knots': y_knots,
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

# Test case
if __name__ == '__main__':
    # Test 1: 2D grid
    print("Testing 2D grid generation...")
    settings_2d = {
        'NX': 300, 'NY': 100,
        'DOMAIN_LENGTH_X': 4.0, 'DOMAIN_HEIGHT_H': 1.0,
        'P_DEGREE': 4, 'Q_DEGREE': 4,
        'Y_STRETCH_FACTOR': 2.5
    }
    grid_2d = create_channel_grid(Nx = 300, Ny = 100, Nz = 0, Lx = 8.0, H = 1.0, Lz = 0.0, p = 5, q = 5, stretch_factor = 2.0) 
    
    # Verification
    print("\n--- 2D Verification ---")
    assert grid_2d['is_3d'] is False

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(grid_2d['X'], grid_2d['Y'], 'ko', markersize=2)
    plt.plot(grid_2d['X'], grid_2d['Y'], 'k-', linewidth=0.5)
    plt.plot(grid_2d['X'].T, grid_2d['Y'].T, 'k-', linewidth=0.5)
    plt.xlabel("x (B-spline Direction)")
    plt.ylabel("y (B-spline Direction)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Test2: 3D
    print("Testing 3D grid generation...")
    grid_3d = create_channel_grid(Nx = 200, Ny = 50, Nz = 4, Lx = 8.0, H = 1.0, Lz = 2.0, p = 4, q = 4, stretch_factor = 2.0)

    # Verification
    print("\n--- 3D Verification ---")
    
    # Visualization (3D scatter plot of points)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_3d['X'], grid_3d['Z'], grid_3d['Y'], c='k', marker='.')
    ax.set_xlabel("x (Streamwise)")
    ax.set_ylabel("z (Wall-normal)")
    ax.set_zlabel("z (Spanwise)")
    ax.set_box_aspect([grid_3d['Lx'], grid_3d['Lz'], 2 * grid_3d['H']])
    plt.show()
