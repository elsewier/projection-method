
import numpy as np
import matplotlib.pyplot as plt
from bspline_module import generate_knots_and_colloc_pts, generate_periodic_knots_and_colloc_pts

def create_channel_grid(Nx, Ny, Nz, Lx, H, Lz, p, q, stretch_factor,periodic_x=False): # creates 2d by default


    print(f"3D Grid Generation ({Nx}x{Ny}x{Nz})")

    # x direction
    xmin, xmax = 0, Lx
    if periodic_x:
        x_knots, x_colloc = generate_periodic_knots_and_colloc_pts(p, Nx, xmin, xmax) 
    else:
        x_knots, x_colloc = generate_knots_and_colloc_pts(p, Nx, xmin, xmax) #

    # y direction
    ymin, ymax = -H, H
    y_knots, y_colloc = generate_knots_and_colloc_pts(q, Ny, ymin, ymax, stretch_factor = stretch_factor)

    dy_wall = np.abs(y_colloc[0] - y_colloc[1])
    print(f"dy_wall: {dy_wall:.5f}")

    # z direction
    z_colloc = np.linspace(0, Lz, Nz, endpoint = False) # endpoint false excludes the last point
    dz = Lz / Nz 
    kz = (2 * np.pi) * np.fft.rfftfreq(Nz, d = dz)
    Nz_fourier = len(kz) # number of fourier modes 
    # meshgrid
    X, Y, Z = np.meshgrid(x_colloc, y_colloc, z_colloc, indexing='ij') # indexing ij => (len(x), len(y)) if we use xy it will be (len(y), len(x))

    kz_3d = kz.reshape(1, 1, -1)



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
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'Nz_fourier': Nz_fourier,
        'Lx': Lx, 'H': H, 'Lz': Lz,
        'kz': kz, 'kz_3d': kz_3d 
    }

    return grid_data

