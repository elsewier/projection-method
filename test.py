import numpy as np
import matplotlib.pyplot as plt

from bspline_ops import BSplineOperator
from grid import create_channel_grid

# --- 1. Setup Grid and Operators ---
p = 7
q = 7
ngx = 64
ngy = 64

Lx = 4*np.pi
H = 1.0
Lz = 1.0 

grid = create_channel_grid(Nx=ngx, Ny=ngy, Nz=1, Lx=Lx, H=H, Lz=Lz, p=p, q=q, stretch_factor=0.99, periodic_x=True)
operators = BSplineOperator(grid, p=p, q=q, periodic_x=True)


# --- 2. Create the 2D Analytical Function and its Derivatives ---
X = grid['X']
Y = grid['Y']
y_coords = grid['y_colloc']
x_coords = grid['x_colloc']

kx = 4 * np.pi / Lx  # Ensures one full sine wave over the domain [0, Lx]
ky = np.pi / (2 * H)   # One half cosine wave in the y-domain [-H, H]

print(f"Using analytical function: sin({kx:.2f}*x) * cos({ky:.2f}*y)")

f_analytical   = np.sin(kx * X) * np.cos(ky * Y)
fx_analytical  =  kx * np.cos(kx * X) * np.cos(ky * Y)
fxx_analytical = -(kx**2) * np.sin(kx * X) * np.cos(ky * Y)
fy_analytical  = -ky * np.sin(kx * X) * np.sin(ky * Y)
fyy_analytical = -(ky**2) * np.sin(kx * X) * np.cos(ky * Y)

# Helper function for plotting
def plot_error(error_field, x_coords, y_coords, title, subtitle):
    plt.figure(figsize=(14, 5))
    
    # 2D Error Heatmap
    plt.subplot(1, 2, 1)
    # Corrected plotting to handle 2D arrays
    plt.pcolormesh(x_coords, y_coords, error_field[:,:,0].T, shading='gouraud', cmap='RdBu_r')
    plt.colorbar(label='Error')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title(f'{title}\n2D Error Distribution')

    # 1D Error Profile Slice
    plt.subplot(1, 2, 2)
    if 'Dy' in title:
        center_x_idx = error_field.shape[0] // 2
        plt.plot(error_field[center_x_idx, :], y_coords, '-o', markersize=4)
        plt.xlabel('Error (Numerical - Analytical)')
        plt.ylabel('y-coordinate')
        plt.title(f'{title}\nError Profile at x = {x_coords[center_x_idx]:.2f}')
    else: # Dx/Dxx
        center_y_idx = error_field.shape[1] // 2
        plt.plot(x_coords, error_field[:, center_y_idx], '-o', markersize=4)
        plt.xlabel('x-coordinate')
        plt.ylabel('Error (Numerical - Analytical)')
        plt.title(f'{title}\nError Profile at y = {y_coords[center_y_idx]:.2f}')

    plt.grid(True)
    plt.suptitle(subtitle, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Test for Dx ---
print("\n--- Testing Dx Operator ---")
print(np.shape(f_analytical))
# fx_numerical = operators.Dx @ f_analytical
fx_numerical = (operators.Dx @ f_analytical.reshape(ngx, -1)).reshape(ngx, ngy, 1)
error_dx = fx_numerical - fx_analytical
max_abs_error_dx = np.max(np.abs(error_dx))
print(f"Maximum absolute error in Dx operator: {max_abs_error_dx:.4e}")
if max_abs_error_dx > 1e-6:
    plot_error(error_dx, x_coords, y_coords, 'Dx Error', f'Max Error: {max_abs_error_dx:.4e}')

# --- Test for Dxx ---
print("\n--- Testing Dxx Operator ---")
fxx_numerical = (operators.Dxx @ f_analytical.reshape(ngx, -1)).reshape(ngx, ngy, 1)
error_dxx = fxx_numerical - fxx_analytical
max_abs_error_dxx = np.max(np.abs(error_dxx))
print(f"Maximum absolute error in Dxx operator: {max_abs_error_dxx:.4e}")
if max_abs_error_dxx > 1e-6:
    plot_error(error_dxx, x_coords, y_coords, 'Dxx Error', f'Max Error: {max_abs_error_dxx:.4e}')

# --- Test for Dy ---
print("\n--- Testing Dy Operator ---")
fy_numerical = (operators.Dy @ f_analytical.T).T
error_dy = fy_numerical - fy_analytical
max_abs_error_dy = np.max(np.abs(error_dy))
print(f"Maximum absolute error in Dy operator: {max_abs_error_dy:.4e}")
if max_abs_error_dy > 1e-6:
    plot_error(error_dy, x_coords, y_coords, 'Dy Error', f'Max Error: {max_abs_error_dy:.4e}')

# --- Test for Dyy ---
print("\n--- Testing Dyy Operator ---")
fyy_numerical = (operators.Dyy @ f_analytical.T).T
error_dyy = fyy_numerical - fyy_analytical

# Exclude boundaries for a clearer max error metric of the interior
max_abs_error_dyy_interior = np.max(np.abs(error_dyy))
print(f"Maximum absolute error in Dyy operator : {max_abs_error_dyy_interior:.4e}")
if max_abs_error_dyy_interior > 1e-6:
    plot_error(error_dyy, x_coords, y_coords, 'Dyy Error', f'Max Interior Error: {max_abs_error_dyy_interior:.4e}')

print("\n--- Test Complete ---")
