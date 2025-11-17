"""
2D B-spline operator module
"""

import numpy as np


def generate_knots_and_colloc_pts(p, num_basis, xmin, xmax, stretch_factor=0.0):

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

def generate_periodic_knots_and_colloc_pts(p, num_basis, xmin, xmax):
    m = num_basis + p + 1
    knots_out = np.zeros(m)

    uniform_knots = np.linspace(0, 1, num_basis + 1)
    knots_out = np.arange(-p, num_basis + 1) * (1.0 / num_basis)

    colloc_norm = np.zeros(num_basis)
    for i in range(num_basis):
        colloc_norm[i] = np.sum(knots_out[i+1:i+p+1]) / float(p)

    colloc_pts_out = xmin + colloc_norm * (xmax - xmin)

    return knots_out, colloc_pts_out



def bspline_basis_normalized(j, p, knots, xi, tol=1.e-12):
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
    xi = (x - xmin) / (xmax - xmin)
    return bspline_basis_normalized(j, p, knots, xi)

def bspline_deriv1_physical(j, p, knots, x, xmin, xmax):
    L = xmax - xmin
    xi = (x - xmin) / L
    dval_dxi = bspline_deriv1_normalized(j, p, knots, xi)
    return dval_dxi * (1.0 / L)

def bspline_deriv2_physical(j, p, knots, x, xmin, xmax):
    L = xmax - xmin
    xi = (x - xmin) / L
    d2val_dxi2 = bspline_deriv2_normalized(j, p, knots, xi)
    return d2val_dxi2 * (1.0 / L**2)

def find_span(x, p, knots, xmin, xmax):
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




    def __init__(self, points, p, q, x, y):
        self.points = np.array(points)
        self.p = p
        self.q = q
        self.x = x
        self.y = y
        self.num_basis_u = self.control_points.shape[0]
        self.num_basis_v = self.control_points.shape[1]
        self.dim = self.control_points.shape[2]

        # first generate knots and colloc points 
        self.x_knots, _ = generate_knots_and_colloc_pts(p, self.num_basis_u, 0., 1.)
        self.y_knots, _ = generate_knots_and_colloc_pts(q, self.num_basis_v, 0., 1.)

    def evaluate_basis_functions(self, x, y, dx, dy):

        # we need to find non-zero basis range 
        x_span = find_span(x, self.p, self.x_knots, self.x[0], self.x[1])
        y_span = find_span(y, self.q, self.y_knots, self.y[0], self.y[1])

        # this is the range that we interested (nonzero ones)
        points  = self.points[u_span - self.p : u_span + 1, v_span - self.q : v_span + 1, :]

        # get the function name based on input 
        x_func = {
            0: bspline_basis_physical,
            1: bspline_deriv1_physical,
            2: bspline_deriv2_physical
        }.get(du)
        
        y_func = {
            0: bspline_basis_physical,
            1: bspline_deriv1_physical,
            2: bspline_deriv2_physical
        }.get(dv)
        
        # now we need to evaluate them 
        N_val   = np.zeros(self.p + 1)
        M_val   = np.zeros(self.q + 1)

        for i in range(self.p + 1):
            j = u_span - self.p + i 
            N_val[i] = x_func(j, self.p, self.x_knots, x, self.x[0], self.x[1])

        for j in range(self.q + 1):
            j = y_span - self.q + i 
            M_val[i] = y_func(j, self.q, self.y_knots, y, self.y[0], self.x[1])

        return N_val, M_val, points 

    # now we can contruct our 2d bspline 
    # S(x,y) = sum_i sum_j N_i(x) * M_j(y) * points_ij
    # evaluate function calculates the value in single point
    def evaluate(self, x, y):
        N_val, M_val, points = self.evaluate_basis_functions(x, y, dx = 0, dy = 0)


        surface_point = np.zeros(points.shape[2]) # holds the dimension
        
        for j in range(M_val.shape[0]):
            for i in range(N_val.shape[0]):
                surface_point += N_val[i] * M_val[j] * points[i, j, :]
        return surface_point

    # evaluate derivative 
    def derivative(self, x, y, dx, dy):
        if dx == 0 and dy == 0:
            return self.evaluate(x, y)

        N_val, M_val, points = self.evaluate_basis_functions(x, y, dx = 0, dy = 0)

        derivative_vec = np.zeros(points.shape[2]) 
        
        for j in range(M_val.shape[0]):
            for i in range(N_val.shape[0]):
                derivative_vec += N_val[i] * M_val[j] * points[i, j, :]
 
        return derivative_vec

