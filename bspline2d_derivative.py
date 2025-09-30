import numpy as np
import matplotlib.pyplot as plt
# Assuming your new class is saved in the same file
import bspline2d_module as bsp 

# --- Analytical Functions to Test Against ---

def analytical_func(x, y):
    """The analytical function f(x, y)."""
    return np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)

def analytical_df_dx(x, y):
    """The analytical partial derivative ∂f/∂x."""
    return 4 * np.pi * np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y)

def analytical_df_dy(x, y):
    """The analytical partial derivative ∂f/∂y."""
    return -4 * np.pi * np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)

def analytical_d2f_dx2(x, y):
    """The analytical second partial derivative ∂²f/∂x²."""
    return -(4 * np.pi)**2 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)

def analytical_d2f_dy2(x, y):
    """The analytical second partial derivative ∂²f/∂y²."""
    return -(4 * np.pi)**2 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)

def analytical_d2f_dxdy(x, y):
    """The analytical mixed partial derivative ∂²f/∂x∂y."""
    return - (4 * np.pi)**2 * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y)


# --- New Test Function Using the BSplineSurface Class ---

def run_test_with_class(p, q, num_basis_u, num_basis_v):
    """
    Runs a test case by fitting a BSplineSurface to an analytical function
    and checking the accuracy of its derivatives.
    """
    u_domain = (0.0, 1.0)
    v_domain = (0.0, 1.0)

    print(f"\n--- Running Test for num_basis = {num_basis_u}x{num_basis_v} ---")

    # === Step 1: Find the control points via collocation ===
    
    # Generate 1D knots and collocation points for both directions
    knots_u, colloc_pts_u = bsp.generate_knots_and_colloc_pts(p, num_basis_u, u_domain[0], u_domain[1])
    knots_v, colloc_pts_v = bsp.generate_knots_and_colloc_pts(q, num_basis_v, v_domain[0], v_domain[1])

    # Build 1D evaluation matrices (B[i, j] = B_j(x_i))
    Bu = np.zeros((num_basis_u, num_basis_u))
    for i in range(num_basis_u):
        for j in range(num_basis_u):
            Bu[i, j] = bsp.bspline_basis_physical(j, p, knots_u, colloc_pts_u[i], u_domain[0], u_domain[1])

    Bv = np.zeros((num_basis_v, num_basis_v))
    for i in range(num_basis_v):
        for j in range(num_basis_v):
            Bv[i, j] = bsp.bspline_basis_physical(j, q, knots_v, colloc_pts_v[i], v_domain[0], v_domain[1])

    # Evaluate the analytical function on the grid of collocation points
    colloc_grid_u, colloc_grid_v = np.meshgrid(colloc_pts_u, colloc_pts_v)
    F_analytical = analytical_func(colloc_grid_u, colloc_grid_v)

    # Solve the linear system Bv @ C @ Bu.T = F_analytical for the control points C
    # This finds the control points that make the surface pass through the analytical values
    Z = np.linalg.solve(Bv, F_analytical)
    coeffs_C = np.linalg.solve(Bu, Z.T).T # Transpose back to get correct shape
    
    # The surface is a height field, so control points are (x, y, z) where (x,y)
    # are from the collocation grid and z is the coefficient we just found.
    # Our BSplineSurface class handles this if we treat z as a 1D dimension.
    # Reshape C to be (num_u, num_v, 1) to represent a scalar value (height).
    control_points_3d = np.expand_dims(coeffs_C.T, axis=2)

    # === Step 2: Create the BSplineSurface object ===
    
    surface = bsp.BSplineSurface(control_points_3d, p, q, u_domain, v_domain)

    # === Step 3: Evaluate derivatives on a fine grid and find errors ===

    # Create a dense grid for evaluation (can be different from collocation grid)
    eval_u = np.linspace(u_domain[0], u_domain[1], 150)
    eval_v = np.linspace(v_domain[0], v_domain[1], 150)
    eval_grid_u, eval_grid_v = np.meshgrid(eval_u, eval_v)

    # Initialize arrays to store errors
    error_dfdu = np.zeros_like(eval_grid_u)
    error_dfdv = np.zeros_like(eval_grid_u)
    error_d2fdu2 = np.zeros_like(eval_grid_u)
    error_d2fdv2 = np.zeros_like(eval_grid_u)
    error_d2fdudv = np.zeros_like(eval_grid_u)

    # Loop over the evaluation grid
    for i in range(eval_grid_u.shape[0]):
        for j in range(eval_grid_u.shape[1]):
            u, v = eval_grid_u[i, j], eval_grid_v[i, j]
            
            # Get B-spline derivatives from the class method
            # [0] is used because the result is a 1D vector [z_deriv]
            approx_dfdu = surface.derivative(u, v, du=1, dv=0)[0]
            approx_dfdv = surface.derivative(u, v, du=0, dv=1)[0]
            approx_d2fdu2 = surface.derivative(u, v, du=2, dv=0)[0]
            approx_d2fdv2 = surface.derivative(u, v, du=0, dv=2)[0]
            approx_d2fdudv = surface.derivative(u, v, du=1, dv=1)[0]
            
            # Compare to analytical derivatives
            error_dfdu[i, j] = analytical_df_dx(u, v) - approx_dfdu
            error_dfdv[i, j] = analytical_df_dy(u, v) - approx_dfdv
            error_d2fdu2[i, j] = analytical_d2f_dx2(u, v) - approx_d2fdu2
            error_d2fdv2[i, j] = analytical_d2f_dy2(u, v) - approx_d2fdv2
            error_d2fdudv[i, j] = analytical_d2f_dxdy(u, v) - approx_d2fdudv

    # Report max errors
    max_err_dfdu = np.max(np.abs(error_dfdu))
    max_err_dfdv = np.max(np.abs(error_dfdv))
    max_err_d2fdu2 = np.max(np.abs(error_d2fdu2))
    max_err_d2fdv2 = np.max(np.abs(error_d2fdv2))
    max_err_d2fdudv = np.max(np.abs(error_d2fdudv))

    print(f"Max absolute error in df/du:      {max_err_dfdu:.2e}")
    print(f"Max absolute error in df/dv:      {max_err_dfdv:.2e}")
    print(f"Max absolute error in d2f/du2:    {max_err_d2fdu2:.2e}")
    print(f"Max absolute error in d2f/dv2:    {max_err_d2fdv2:.2e}")
    print(f"Max absolute error in d2f/dudv:   {max_err_d2fdudv:.2e}")
    
    # Data for plotting
    plot_data = (eval_grid_u, eval_grid_v, error_dfdu, error_dfdv, error_d2fdu2, error_d2fdv2, error_d2fdudv)
    return plot_data

# --- Main Execution ---

if __name__ == "__main__":
    p_degree = 5
    q_degree = 5
    num_basis = 100 # Use same number of basis functions in each direction

    print("================================================================")
    print("  B-Spline Surface Derivative Check (using BSplineSurface Class)")
    print(" f(u,v) = sin(4*pi*u) * cos(4*pi*v)")
    print("================================================================")
    
    plot_data = run_test_with_class(p_degree, q_degree, num_basis, num_basis)
    grid_u, grid_v, err_dfdu, err_dfdv, err_d2fdu2, err_d2fdv2, err_d2fdudv = plot_data

    # --- Plotting Results ---
    
    titles = [
        f"Error in df/du (N={num_basis})", f"Error in df/dv (N={num_basis})",
        f"Error in d2f/du2 (N={num_basis})", f"Error in d2f/dv2 (N={num_basis})",
        f"Error in d2f/dudv (N={num_basis})"
    ]
    error_fields = [err_dfdu, err_dfdv, err_d2fdu2, err_d2fdv2, err_d2fdudv]

    for title, errors in zip(titles, error_fields):
        plt.figure(figsize=(8, 7))
        plt.contourf(grid_u, grid_v, errors, levels=30, cmap='RdBu_r')
        plt.colorbar(label="Error Value")
        plt.title(title)
        plt.xlabel("u")
        plt.ylabel("v")
        plt.axis('square')
        plt.tight_layout()

    plt.show()