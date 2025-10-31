import numpy as np
import scipy.sparse as sp 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres, cg
from pyamg.krylov import cg, gmres
import pyamg 
import time

from operators import build_A, apply_dirichlet, apply_neumann, boundary_flag
from bspline_ops import BSplineOperator 
from grid import create_channel_grid 

def compute_nonlinear_term(u, v, p, ops):
    # N(u, p) = -(u.grad)u - grad(p)

    # compute gradients of velocities 
    du_dx = ops.Dx @ u # apply along column Dx is (Nx, Nx), u is (Nx, Ny)
    du_dy = u @ ops.Dy.T # apply along rows Dy is (Ny, Ny) v is (Nx, Ny)
    dv_dx = ops.Dx @ v 
    dv_dy = v @ ops.Dy.T 

    # pressure gradient 
    dp_dx = ops.Dx @ p 
    dp_dy = p @ ops.Dy.T

    # build nonlinear term 
    N_u = -(u * du_dx + v * du_dy) - dp_dx 
    N_v = -(u * dv_dx + v * dv_dy) - dp_dy 

    return N_u, N_v 

def predictor_step(u_n, v_n, p_n, operators, nu, dt, alpha, beta, gamma, zeta):

    Nx = operators.Nx 
    Ny = operators.Ny 

    # LHS matrix
    A = build_A(operators, nu, dt, beta)

    # RHS vectors 
    # viscous term L(u_n)
    Laplacian_2D = operators.laplacian_2d()
    viscous_u = nu * (Laplacian_2D @ u_n.flatten())
    viscous_v = nu * (Laplacian_2D @ v_n.flatten())

    # nonlinear term 
    N_u, N_v = compute_nonlinear_term(u_n, v_n, p_n, operators)

    # combine them 
    rhs_u = u_n.flatten() + dt * (alpha * viscous_u + gamma * N_u.flatten())
    rhs_v = v_n.flatten() + dt * (alpha * viscous_v + gamma * N_v.flatten()) 

    # apply boundary conditions 
    left, right, bottom, top = boundary_flag(Nx, Ny)

    # inlet velocity profile (parabolic)
    # U(y) = U_max * (1 - (y / H)^2)
    H = operators.grid['H']
    U_max = 1.0 
    y_inlet = operators.grid['y_colloc']
    u_inlet = U_max * (1 - (y_inlet / H)**2)

    # no slip on the top and bottom walls 
    wall_idx = np.concatenate([bottom, top])
    A, rhs_u = apply_dirichlet(A, rhs_u, wall_idx, val = 0.0)
    A, rhs_v = apply_dirichlet(A, rhs_v, wall_idx, val = 0.0)

    # inlet velocity profile and v = 0 
    A, rhs_u = apply_dirichlet(A, rhs_u, left, val = u_inlet)
    A, rhs_v = apply_dirichlet(A, rhs_v, left, val = 0.0)

    # zero gradient for outlet 
    A, rhs_u = apply_neumann(A, rhs_u, operators)
    A, rhs_v = apply_neumann(A, rhs_v, operators)

    # pyamg + gmres 
    start_time = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(A)
    M = ml.aspreconditioner(cycle = 'V')
    u_tilde_vector, info = gmres(A, rhs_u, M = M, tol = 1e-06, restart = 40, maxiter = 80)
    v_tilde_vector, info = gmres(A, rhs_v, M = M, tol = 1e-06, restart = 40, maxiter = 80)
    end_time = time.perf_counter()
    print(f"Elapsed solver time:{end_time - start_time}")

    u_tilde = np.reshape(u_tilde_vector,(Nx, Ny))
    v_tilde = np.reshape(v_tilde_vector,(Nx, Ny))

    return u_tilde, v_tilde 



if __name__ == '__main__':
    p = 5 
    q = 5 
    Nx = 200
    Ny = 200
    grid = create_channel_grid(Nx = Nx, Ny = Ny, Nz = 1, Lx = 1.0,H = 0.5, Lz = 0.0, p = p, q = q, stretch_factor = 2.0)
    operators = BSplineOperator(grid, p = p, q = q)

    u_initial = np.zeros((Nx, Ny))
    v_initial = np.zeros((Nx, Ny))
    p_initial = np.zeros((Nx, Ny))

    nu = 1/180
    dt = 0.005 
    alpha1 = 29 / 60
    beta1 = 37 / 160
    gamma1 = 8 / 15
    zeta1 = 0.0

    # substep 1 
    # predict
    u_predicted, v_predicted = predictor_step(u_initial, v_initial, p_initial, operators, nu, dt, alpha1, beta1, gamma1, zeta1)
    # correct 
    
    # update 

    print(f"Max u-velocity after 1 predictor step: {np.max(u_predicted):.4f}")
    # plt.pcolormesh(grid['X'], grid['Y'], u_predicted, shading = 'gouraud', cmap = 'viridis')
    # plt.show()






