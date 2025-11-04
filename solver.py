import numpy as np
import scipy.sparse as sp 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres, cg
from pypardiso import spsolve
import time

from operators import build_A, build_P, apply_dirichlet, apply_neumann, boundary_flag, pin_pressure
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
    N_u = -(u * du_dx + v * du_dy)# - dp_dx 
    N_v = -(u * dv_dx + v * dv_dy)# - dp_dy 

    return N_u, N_v 
    
def compute_advection_term(u, v, ops):
    du_dx = ops.Dx @ u
    du_dy = u @ ops.Dy.T
    dv_dx = ops.Dx @ v
    dv_dy = v @ ops.Dy.T
    
    N_u = -(u * du_dx + v * du_dy)
    N_v = -(u * dv_dx + v * dv_dy)
    
    return N_u, N_v

def predictor_step(rhs_u, rhs_v, operators, nu, dt, beta):

    Nx = operators.Nx 
    Ny = operators.Ny 

    # LHS matrix
    A = build_A(operators, nu, dt, beta)


    # apply boundary conditions 
    left, right, bottom, top = boundary_flag(Nx, Ny)

    # no slip on the top and bottom walls 
    wall_idx = np.concatenate([bottom, top])
    A_u, rhs_u = apply_dirichlet(A.copy(), rhs_u, wall_idx, val = 0.0)
    A_v, rhs_v = apply_dirichlet(A.copy(), rhs_v, wall_idx, val = 0.0)

    start_time = time.perf_counter()

    u_tilde_vector = spsolve(A_u, rhs_u)
    v_tilde_vector = spsolve(A_v, rhs_v)
    end_time = time.perf_counter()
    # print(f"Elapsed solver time:{end_time - start_time}")

    u_tilde = np.reshape(u_tilde_vector,(Nx, Ny))
    v_tilde = np.reshape(v_tilde_vector,(Nx, Ny))

    return u_tilde, v_tilde


def correction_step(u_tilde, v_tilde, operators, dt):
    Nx = operators.Nx 
    Ny = operators.Ny 

    # we need to solve nabla^2 phi' = (1/delta t) nabla . u_tilde' 
    du_dx = operators.Dx @ u_tilde 
    dv_dy = v_tilde @ operators.Dy.T 

    div = du_dx + dv_dy 
    # print(np.abs(np.max(div)))

    # RHS of poisson equation 
    rhs_p = (1 / dt) * div.flatten()
    # poisson matrix 
    P = build_P(operators)

    # apply bcs 
    left, right, bottom, top = boundary_flag(Nx, Ny)
    
    wall_idx = np.concatenate([bottom, top])
    P, rhs_p = apply_neumann(P, rhs_p, wall_idx, operators)

    rhs_p -= np.mean(rhs_p)
    P, rhs_p = pin_pressure(P, rhs_p, 0) # pin a corner to 0 

    # solve for pressure correction 
    # print("Solving pressure Poisson for substep-1")
    start_time = time.time()
    phi_vec = spsolve(P, rhs_p)
    end_time = time.time()
    # print(f"Pressure Poisson took:{end_time - start_time:.4f} seconds")

    phi = np.reshape(phi_vec, (Nx, Ny))

    return phi


def update_step(u_tilde, v_tilde, phi, p_n, operators, dt):
    grad_phi_x = operators.Dx @ phi 
    grad_phi_y = phi @ operators.Dy.T 

    # u^n+1 = u_tilde - \Delta t * grad_phi 
    u_new = u_tilde - dt * grad_phi_x
    v_new = v_tilde - dt * grad_phi_y

    # p^n+1 = p^n + phi 
    p_new = p_n + phi 

    # # i may need to enforce boundary conditions again 
    # left, right, bottom, top = boundary_flag(Nx, Ny)
    # wall_idx = np.concatenate([bottom, top])
    # u_new.ravel()[wall_idx] = 0.0
    # v_new.ravel()[wall_idx] = 0.0

    return u_new, v_new, p_new



if __name__ == '__main__':
    p = 5 
    q = 5 
    Nx = 40
    Ny = 40
    grid = create_channel_grid(Nx = Nx, Ny = Ny, Nz = 1, Lx = 0.5,H = 0.5, Lz = 0.0, p = p, q = q, stretch_factor = 2.0)
    operators = BSplineOperator(grid, p = p, q = q, periodic_x = True)


    H = grid['H']
    Y = grid['Y']
    u_n = 1.0 * (1 - (Y / H)**2) # parabolic velocity profile 
    v_n = np.zeros((Nx, Ny))
    p_n = np.zeros((Nx, Ny))
    
    Re = 180
    num_steps = 1000

    nu = 1 / Re
    dt = 0.001 

    body_force = (2.0 * nu * 1.0) / (H**2) # U_center = 1.0
    alpha1, beta1, gamma1, zeta1 = 29/60, 37/160, 8/15, 0.0
    alpha2, beta2, gamma2, zeta2 = -3/40, 5/24, 5/12, -17/ 60
    alpha3, beta3, gamma3, zeta3 = 1/6, 1/6, 3/4, -5/12

    Laplacian_2D = operators.laplacian_2d()

    for n in range(num_steps):
        print(f"Time-step {n + 1} / {num_steps}")
    
        u_old, v_old, p_old = u_n.copy(), v_n.copy(), p_n.copy()

        # compute N(u^n, p^n)
        Nu_old, Nv_old = compute_nonlinear_term(u_old, v_old, p_old, operators)

        print("Substep 1")
        # assemble RHS 
        viscous_u_old = nu * (Laplacian_2D @ u_old.flatten())
        viscous_v_old = nu * (Laplacian_2D @ v_old.flatten())

        rhs_u1 = u_old.flatten() + dt * (alpha1 * viscous_u_old + gamma1 * Nu_old.flatten()) + dt * body_force
        rhs_v1 = v_old.flatten() + dt * (alpha1 * viscous_v_old + gamma1 * Nv_old.flatten())

        u_tilde1, v_tilde1 = predictor_step(rhs_u1, rhs_v1, operators, nu, dt, beta1)
        phi1 = correction_step(u_tilde1, v_tilde1, operators, dt)
        u1, v1, p1 = update_step(u_tilde1, v_tilde1, phi1, p_old, operators, dt)

        # compute N(u', p')
        Nu1, Nv1 = compute_nonlinear_term(u1, v1, p1, operators)

        # print(f" Max |u_tilde1|: {np.max(np.abs(u_tilde1)):.2e}")
        # print(f" Max |v_tilde1|: {np.max(np.abs(v_tilde1)):.2e}")
        # print(f" Max |phi1|: {np.max(np.abs(phi1)):.2e}")
        # print(f" Max |u1|: {np.max(np.abs(u1)):.2e}")

    
        print("Substep 2")
        # assemble RHS 
        viscous_u1  = nu * (Laplacian_2D @ u1.flatten())
        viscous_v1  = nu * (Laplacian_2D @ v1.flatten())

        rhs_u2 = u1.flatten() + dt * (alpha2 * viscous_u1 + gamma2 * Nu1.flatten() + zeta2 * Nu_old.flatten())
        rhs_v2 = v1.flatten() + dt * (alpha2 * viscous_v1 + gamma2 * Nv1.flatten() + zeta2 * Nv_old.flatten())

        u_tilde2, v_tilde2 = predictor_step(rhs_u2, rhs_v2, operators, nu, dt, beta2)
        phi2 = correction_step(u_tilde2, v_tilde2, operators, dt)
        u2, v2, p2 = update_step(u_tilde2, v_tilde2, phi2, p1, operators, dt)

        # compute N(u'', p'')
        Nu2, Nv2 = compute_nonlinear_term(u2, v2, p2, operators)
        # print(f" Max |u_tilde2|: {np.max(np.abs(u_tilde2)):.2e}")
        # print(f" Max |v_tilde2|: {np.max(np.abs(v_tilde2)):.2e}")
        # print(f" Max |phi2|: {np.max(np.abs(phi2)):.2e}")
        # print(f" Max |u2|: {np.max(np.abs(u2)):.2e}")

        print("Substep 3")
        # assemble RHS 
        viscous_u2  = nu * (Laplacian_2D @ u2.flatten())
        viscous_v2  = nu * (Laplacian_2D @ v2.flatten())

        rhs_u3 = u2.flatten() + dt * (alpha3 * viscous_u2 + gamma3 * Nu2.flatten() + zeta3 * Nu1.flatten())
        rhs_v3 = v2.flatten() + dt * (alpha3 * viscous_v2 + gamma3 * Nv2.flatten() + zeta3 * Nv1.flatten())

        u_tilde3, v_tilde3 = predictor_step(rhs_u3, rhs_v3, operators, nu, dt, beta3)
        phi3 = correction_step(u_tilde3, v_tilde3, operators, dt)
        u3, v3, p3 = update_step(u_tilde3, v_tilde3, phi3, p2, operators, dt)

        # print(f" Max |u_tilde3|: {np.max(np.abs(u_tilde3)):.2e}")
        # print(f" Max |v_tilde3|: {np.max(np.abs(v_tilde3)):.2e}")
        # print(f" Max |phi3|: {np.max(np.abs(phi3)):.2e}")
        # print(f" Max |u3|: {np.max(np.abs(u3)):.2e}")
        # final update 
        u_n, v_n, p_n = u3, v3, p3 
        # u_n, v_n, p_n = u1, v1, p1 


        print(f"Step {n+1}: Max u-velocity = {np.max(u_n):.4f}")


    plt.pcolormesh(grid['X'], grid['Y'], u_n, shading = 'gouraud', cmap = 'viridis')
    plt.colorbar()
    plt.show()






