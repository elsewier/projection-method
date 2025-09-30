"""
2D B-spline operator module

This module uses the recursive Cox-de Boor algorithm for b-spline calculations
and extends the 1D functionality to support 2D B-spline surfaces through a
tensor product formulation.

============================
Ali Yesildag August 25/2025
============================
"""

import numpy as np

# --- 1D B-spline Helper Functions (Original Module) ---

def generate_knots_and_colloc_pts(p, num_basis, xmin, xmax, stretch_factor=0.0):
    """
    Generates a 1D clamped (non-periodic) knot vector and Greville collocation points.

    Parameters
    ----------
    p : int
        The degree of B-spline basis.
    num_basis : int
        The number of basis functions to generate.
    xmin : float
        The minimum value of the physical domain.
    xmax : float
        The maximum value of the physical domain.
    stretch_factor : float, optional
        Stretch factor to create a stretched mesh near the boundary. The default is 0.0.

    Returns
    -------
    knots_out : np.ndarray
        The knot vector of size (num_basis + p + 1).
    colloc_pts_out : np.ndarray
        The array of collocation points of size (num_basis).
    """
    if num_basis <= p:
        raise ValueError(f"Error: num_basis ({num_basis}) must be greater than p ({p}).")

    m = num_basis + p + 1
    knots_out = np.zeros(m)

    # Clamped end knots
    knots_out[:p+1] = 0.0
    knots_out[num_basis:] = 1.0

    # Interior knots
    for i in range(p + 1, num_basis):
        s = float(i - p) / float(num_basis - p)
        if abs(stretch_factor) < 1.0e-12:
            # Uniform spacing
            knots_out[i] = s
        else:
            # Tanh stretching
            knots_out[i] = 0.5 * (np.tanh(stretch_factor * (2.0 * s - 1.0)) + 1.0)

    # Greville collocation points
    colloc_norm = np.zeros(num_basis)
    for i in range(num_basis):
        colloc_norm[i] = np.sum(knots_out[i+1:i+p+1]) / float(p)

    # Map to physical domain
    colloc_pts_out = xmin + colloc_norm * (xmax - xmin)
    colloc_pts_out[0] = xmin
    colloc_pts_out[-1] = xmax
    
    return knots_out, colloc_pts_out

def bspline_basis_normalized(j, p, knots, xi, tol=1.e-12):
    """
    Computes a B-spline basis function value using the Cox-de Boor recursion.
    """
    if p == 0:
        # The last interval is closed to include xi=1.0
        if abs(knots[j+1] - 1.0) < tol:
             if xi >= knots[j] - tol and xi <= knots[j+1] + tol:
                 return 1.0
             else:
                 return 0.0
        # Half-open interval for all other cases
        else:
            if xi >= knots[j] - tol and xi < knots[j+1] - tol:
                return 1.0
            else:
                return 0.0
    else:
        term1 = 0.0
        d1 = knots[j+p] - knots[j]
        if abs(d1) > tol:
            term1 = (xi - knots[j]) / d1 * bspline_basis_normalized(j, p-1, knots, xi, tol)

        term2 = 0.0
        d2 = knots[j+p+1] - knots[j+1]
        if abs(d2) > tol:
            term2 = (knots[j+p+1] - xi) / d2 * bspline_basis_normalized(j+1, p-1, knots, xi, tol)
            
        return term1 + term2

def bspline_deriv1_normalized(j, p, knots, xi, tol=1.e-12):
    """
    Computes the first derivative of a B-spline basis function.
    """
    if p == 0:
        return 0.0

    term1 = 0.0
    d1 = knots[j+p] - knots[j]
    if abs(d1) > tol:
        term1 = bspline_basis_normalized(j, p-1, knots, xi, tol) / d1

    term2 = 0.0
    d2 = knots[j+p+1] - knots[j+1]
    if abs(d2) > tol:
        term2 = bspline_basis_normalized(j+1, p-1, knots, xi, tol) / d2

    return float(p) * (term1 - term2)

def bspline_deriv2_normalized(j, p, knots, xi, tol=1.e-12):
    """
    Computes the second derivative of a B-spline basis function.
    """
    if p <= 1:
        return 0.0
    
    term1 = 0.0
    d1 = knots[j+p] - knots[j]
    if abs(d1) > tol:
        term1 = bspline_deriv1_normalized(j, p-1, knots, xi, tol) / d1

    term2 = 0.0
    d2 = knots[j+p+1] - knots[j+1]
    if abs(d2) > tol:
        term2 = bspline_deriv1_normalized(j+1, p-1, knots, xi, tol) / d2
    
    return float(p) * (term1 - term2)

def bspline_basis_physical(j, p, knots, x, xmin, xmax):
    """
    Wrapper to evaluate basis function in the physical domain.
    """
    xi = (x - xmin) / (xmax - xmin)
    return bspline_basis_normalized(j, p, knots, xi)

def bspline_deriv1_physical(j, p, knots, x, xmin, xmax):
    """
    Wrapper to evaluate the first derivative in the physical domain.
    """
    L = xmax - xmin
    xi = (x - xmin) / L
    dval_dxi = bspline_deriv1_normalized(j, p, knots, xi)
    return dval_dxi * (1.0 / L)

def bspline_deriv2_physical(j, p, knots, x, xmin, xmax):
    """
    Wrapper to evaluate the second derivative in the physical domain.
    """
    L = xmax - xmin
    xi = (x - xmin) / L
    d2val_dxi2 = bspline_deriv2_normalized(j, p, knots, xi)
    return d2val_dxi2 * (1.0 / L**2)

def find_span(x, p, knots, xmin, xmax):
    """
    Finds the knot interval index for a given physical point x.
    """
    num_basis = len(knots) - p - 1
    xi = (x - xmin) / (xmax - xmin)
    
    # Handle the right endpoint explicitly
    if np.isclose(xi, 1.0):
        return num_basis - 1
    
    # Use binary search for efficiency
    low = p
    high = num_basis
    mid = (low + high) // 2
    
    while xi < knots[mid] or xi >= knots[mid + 1]:
        if xi < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
        
    return mid

# --- 2D B-spline Surface Class ---

class BSplineSurface:
    """
    A class to represent and evaluate a 2D B-spline surface.
    """
    def __init__(self, control_points, p, q, u_domain=(0., 1.), v_domain=(0., 1.)):
        """
        Initializes a B-spline surface.

        Args:
            control_points (np.ndarray): A NumPy array of control points with shape
                                         (num_basis_u, num_basis_v, dim).
            p (int): Degree in the u-direction.
            q (int): Degree in the v-direction.
            u_domain (tuple): Physical domain for u, e.g., (umin, umax).
            v_domain (tuple): Physical domain for v, e.g., (vmin, vmax).
        """
        self.control_points = np.array(control_points)
        if self.control_points.ndim != 3:
            raise ValueError("control_points must be a 3D array.")
        self.p = p
        self.q = q
        self.u_domain = u_domain
        self.v_domain = v_domain
        self.num_basis_u = self.control_points.shape[0]
        self.num_basis_v = self.control_points.shape[1]
        self.dim = self.control_points.shape[2]

        # Generate knot vectors
        self.U_knots, _ = generate_knots_and_colloc_pts(p, self.num_basis_u, 0., 1.)
        self.V_knots, _ = generate_knots_and_colloc_pts(q, self.num_basis_v, 0., 1.)

    def _evaluate_basis_functions(self, u, v, du, dv):
        """
        Helper to evaluate all relevant basis functions and their derivatives.
        """
        # Find spans
        u_span = find_span(u, self.p, self.U_knots, self.u_domain[0], self.u_domain[1])
        v_span = find_span(v, self.q, self.V_knots, self.v_domain[0], self.v_domain[1])

        # Select the relevant control points
        P_relevant = self.control_points[u_span - self.p : u_span + 1, 
                                          v_span - self.q : v_span + 1, :]

        # Determine which functions to call based on derivative order
        u_func = {
            0: bspline_basis_physical,
            1: bspline_deriv1_physical,
            2: bspline_deriv2_physical
        }.get(du)
        
        v_func = {
            0: bspline_basis_physical,
            1: bspline_deriv1_physical,
            2: bspline_deriv2_physical
        }.get(dv)
        
        if u_func is None or v_func is None:
            raise ValueError("Only 0th, 1st, and 2nd derivatives are supported.")

        # Evaluate basis functions in u
        N_values = np.zeros(self.p + 1)
        for i in range(self.p + 1):
            j = u_span - self.p + i
            N_values[i] = u_func(j, self.p, self.U_knots, u, self.u_domain[0], self.u_domain[1])

        # Evaluate basis functions in v
        M_values = np.zeros(self.q + 1)
        for i in range(self.q + 1):
            j = v_span - self.q + i
            M_values[i] = v_func(j, self.q, self.V_knots, v, self.v_domain[0], self.v_domain[1])
            
        return N_values, M_values, P_relevant

    def evaluate(self, u, v):
        """
        Evaluates the surface point S(u, v).

        Args:
            u (float): Parameter in the u-direction.
            v (float): Parameter in the v-direction.

        Returns:
            np.ndarray: The evaluated point on the surface.
        """
        N_values, M_values, P_relevant = self._evaluate_basis_functions(u, v, du=0, dv=0)
        
        # Perform the tensor product summation using np.einsum for clarity and efficiency
        surface_point = np.einsum('i,j,ijk->k', N_values, M_values, P_relevant)
        return surface_point

    def derivative(self, u, v, du=0, dv=0):
        """
        Evaluates the partial derivatives of the surface.

        Args:
            u (float): Parameter in the u-direction.
            v (float): Parameter in the v-direction.
            du (int): Order of the derivative with respect to u (0, 1, or 2).
            dv (int): Order of the derivative with respect to v (0, 1, or 2).

        Returns:
            np.ndarray: The evaluated partial derivative vector.
        """
        if du == 0 and dv == 0:
            return self.evaluate(u, v)

        N_values, M_values, P_relevant = self._evaluate_basis_functions(u, v, du, dv)

        # Perform the tensor product summation for the derivative
        derivative_vector = np.einsum('i,j,ijk->k', N_values, M_values, P_relevant)
        return derivative_vector