import numpy as np
import scipy.sparse as sp 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres, cg
from pypardiso import spsolve, PyPardisoSolver
import time
import os
import plotly.graph_objects as go 

from operators import build_A, build_P, apply_dirichlet, apply_dirichlet_rhs, apply_neumann, boundary_flag, pin_pressure
from bspline_ops import BSplineOperator 
from grid import create_channel_grid 

def compute_nonlinear_term(u, v, w, ops):
    # N(u, p) = -(u.grad)u 
    # NOTE: the gradP term is included in the correction step!
    
    # i need to apply 3/2 dealiasing 
    ngx, ngy, ngz = u.shape # physical size 
    Nz = int(ngz * 3 / 2) # larger grid 

    Nz_fourier = ops.grid['Nz_fourier'] # number of fourier modes we have before 

    # we need to transform our velocities into fourier space 
    u_hat = np.fft.rfft(u, axis = 2)
    v_hat = np.fft.rfft(v, axis = 2)
    w_hat = np.fft.rfft(w, axis = 2)

    nzz = Nz // 2 + 1 # fourier modes for new large grid 
    u_hat1 = np.zeros((ngx, ngy, nzz), dtype = np.complex128)
    v_hat1 = np.zeros((ngx, ngy, nzz), dtype = np.complex128)
    w_hat1 = np.zeros((ngx, ngy, nzz), dtype = np.complex128)

    # copy original values into low frequency part 
    u_hat1[:, :, :Nz_fourier] = u_hat
    v_hat1[:, :, :Nz_fourier] = v_hat
    w_hat1[:, :, :Nz_fourier] = w_hat

    # now we can transfrom our physical grid 
    u1 = np.fft.irfft(u_hat1, n = Nz, axis = 2)
    v1 = np.fft.irfft(v_hat1, n = Nz, axis = 2)
    w1 = np.fft.irfft(w_hat1, n = Nz, axis = 2)

    # now we can compute our derivatives
    # compute gradients of velocities 
    # X-derivatives
    # NOTE: matmul operation only works for 2D arrays. for that we collapse last two dimensions into one.
    # NOTE: Dx applied into each row. then the result reshaped into original form
    du_dx = (ops.Dx @ u1.reshape(ngx, -1)).reshape(u1.shape)
    dv_dx = (ops.Dx @ v1.reshape(ngx, -1)).reshape(v1.shape)
    dw_dx = (ops.Dx @ w1.reshape(ngx, -1)).reshape(w1.shape)
    
    # Y-derivatives
    # u1 is in shape (ngx, ngy, nzz) --> (ngx, nnz, ngy) 
    # matrix multiply with Dy.T (ngy, ngy). --> shape of the result will be (ngx, nnz, ngy)
    # transpose back to original 
    # NOTE: Dy is applied to each column. 
    du_dy = (u1.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)
    dv_dy = (v1.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)
    dw_dy = (w1.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)

    # Z-derivatives
    kzz = (2 * np.pi) * np.fft.rfftfreq(Nz, d = ops.grid['Lz'] / Nz)
    kzz_3d = kzz.reshape(1, 1, -1) # make it 3d array for derivative calculation 

    du_dz = np.fft.irfft(1j * kzz_3d * u_hat1, n = Nz)
    dv_dz = np.fft.irfft(1j * kzz_3d * v_hat1, n = Nz)
    dw_dz = np.fft.irfft(1j * kzz_3d * w_hat1, n = Nz)

    # build nonlinear term on dealised physical grid 
    Nu = - (u1 * du_dx + v1 * du_dy + w1 * du_dz)
    Nv = - (u1 * dv_dx + v1 * dv_dy + w1 * dv_dz)
    Nw = - (u1 * dw_dx + v1 * dw_dy + w1 * dw_dz)

    # transfrom to fourier space 
    Nu_hat = np.fft.rfft(Nu, axis = 2)
    Nv_hat = np.fft.rfft(Nv, axis = 2)
    Nw_hat = np.fft.rfft(Nw, axis = 2)

    # take only the part that corresponds to our original grid size 
    Nu_hat1 = Nu_hat[:, :, :Nz_fourier]
    Nv_hat1 = Nv_hat[:, :, :Nz_fourier]
    Nw_hat1 = Nw_hat[:, :, :Nz_fourier]


    # drop the last mode (nyquist mode)
    if ngz > 1:
        Nu_hat1[:, :, -1] = 0.0
        Nv_hat1[:, :, -1] = 0.0
        Nw_hat1[:, :, -1] = 0.0

        

    return Nu_hat1, Nv_hat1, Nw_hat1
    

def predictor_step(rhs_u, rhs_v, rhs_w, A_matrices, solver, wall_idx):
    # rhs are in fourier space!!

    ngx, ngy, nz = rhs_u.shape

    u_tilde_hat = np.zeros((ngx, ngy, nz), dtype = np.complex128)
    v_tilde_hat = np.zeros((ngx, ngy, nz), dtype = np.complex128)
    w_tilde_hat = np.zeros((ngx, ngy, nz), dtype = np.complex128)

    for k in range(nz): # loop over each fourier mode 
        rhs_u_k = rhs_u[:, :, k].flatten()
        rhs_v_k = rhs_v[:, :, k].flatten()
        rhs_w_k = rhs_w[:, :, k].flatten()

        A = A_matrices[k] 
        pardiso_solver = solver[k]

        rhs_u_k = apply_dirichlet_rhs(rhs_u_k, wall_idx, val = 0.0)
        rhs_v_k = apply_dirichlet_rhs(rhs_v_k, wall_idx, val = 0.0)
        rhs_w_k = apply_dirichlet_rhs(rhs_w_k, wall_idx, val = 0.0)

        # solve the linear system for each velocity component 
        u_tilde_vec = pardiso_solver.solve(A, rhs_u_k.real) + 1j * pardiso_solver.solve(A, rhs_u_k.imag)
        v_tilde_vec = pardiso_solver.solve(A, rhs_v_k.real) + 1j * pardiso_solver.solve(A, rhs_v_k.imag)
        w_tilde_vec = pardiso_solver.solve(A, rhs_w_k.real) + 1j * pardiso_solver.solve(A, rhs_w_k.imag)

        u_tilde_hat[:, :, k] = u_tilde_vec.reshape(ngx, ngy)
        v_tilde_hat[:, :, k] = v_tilde_vec.reshape(ngx, ngy)
        w_tilde_hat[:, :, k] = w_tilde_vec.reshape(ngx, ngy)

    return u_tilde_hat, v_tilde_hat, w_tilde_hat



def correction_step(u_tilde_hat, v_tilde_hat, w_tilde_hat, ops, dt, Poisson_matrices, Poisson_solver, wall_idx):

    ngx, ngy, nz = u_tilde_hat.shape 
    
    phi_hat = np.zeros((ngx, ngy, nz), dtype = np.complex128)

    for k in range(nz):
        u_tilde_k = u_tilde_hat[:, :, k]
        v_tilde_k = v_tilde_hat[:, :, k]
        w_tilde_k = w_tilde_hat[:, :, k]

        du_dx = ops.Dx @ u_tilde_k
        dv_dy = v_tilde_k @ ops.Dy.T
        dw_dz = 1j * ops.kz[k] * w_tilde_k 

        div = du_dx + dv_dy + dw_dz 

        # RHS of the poisson equation 
        rhs_p_k = (1 / dt) * div.flatten()

        # kz_val = ops.kz[k]
        # P_k = build_P(ops, kz_val)
        P = Poisson_matrices[k]

        pardiso_solver = Poisson_solver[k]

        rhs_p_k[wall_idx] = 0.0 # Neumann on RHS

        if k ==0:
            rhs_p_k[wall_idx] = (1/dt) * v_tilde_k.flatten()[wall_idx]
            rhs_p_k -= np.mean(rhs_p_k)
            rhs_p_k[0] = 0.0 # pinning pressure on RHS
        # else:
        #     rhs_p_k = apply_dirichlet_rhs(rhs_p_k, wall_idx, val = 0.0)

        phi_k = pardiso_solver.solve(P, rhs_p_k.real) + 1j * pardiso_solver.solve(P, rhs_p_k.imag)

        phi_hat[:, :, k] = phi_k.reshape(ngx, ngy) 

    return phi_hat



def update_step(u_tilde_hat, v_tilde_hat, w_tilde_hat, p_hat, phi_hat, ops, dt):
    ngx, ngy, nz = u_tilde_hat.shape 
    
    grad_phi_x = np.zeros((ngx, ngy, nz), dtype = np.complex128)
    grad_phi_y = np.zeros((ngx, ngy, nz), dtype = np.complex128)

    for k in range(nz):
        phi_k = phi_hat[:, :, k]
        grad_phi_x[:, :, k] = ops.Dx @ phi_k 
        grad_phi_y[:, :, k] = phi_k @ ops.Dy.T

    grad_phi_z = 1j * ops.grid['kz_3d'] * phi_hat 

    # update velocities
    u_new_hat = u_tilde_hat - dt * grad_phi_x
    v_new_hat = v_tilde_hat - dt * grad_phi_y
    w_new_hat = w_tilde_hat - dt * grad_phi_z

    # update pressure 
    p_new_hat = p_hat + phi_hat 

    return u_new_hat, v_new_hat, w_new_hat, p_new_hat 

def massflux_correction(u_phys, y_coords, U_bulk_target, dt):
    ngx, ngy, nz = u_phys.shape 

    u_y = np.mean(u_phys, axis = (0, 2))
    U_bulk_curr = np.trapezoid(u_y, y_coords) / (y_coords[-1] - y_coords[0])

    dp_dx = (U_bulk_target - U_bulk_curr) / dt  

    print(f"U_current = {U_bulk_curr:.6f}, U_target = {U_bulk_target:.6f},Required dp/dx = {dp_dx:.6f}")

    return dp_dx


if __name__ == '__main__':
    p = 7 
    q = 7 
    ngx = 128
    ngy = 96
    ngz = 1


    Lx = 4.0 * np.pi 
    dx = Lx / ngx
    H = 1
    Lz = 2.0 * np.pi 

    grid = create_channel_grid(Nx = ngx, Ny = ngy, Nz = ngz, Lx = Lx,H = H, Lz = Lz, p = p, q = q, stretch_factor = 2.0, periodic_x=True)
    operators = BSplineOperator(grid, p = p, q = q, periodic_x = True)

    y_coords = grid['Y'][0, :, 0]
    print(y_coords[0:10])

    Nz_fourier = grid['Nz_fourier']
    # initial conditions 
    X, Y, Z = grid['X'], grid['Y'], grid['Z']
    # u_vel = 1.0 * (1 - (Y / H)**2)
    u_vel = 1.5 * 1.0 * (1 - (Y / H)**2) # 1.0 is U_bulk target and U_bulk = (2/3) * U_max

    u_n = u_vel#np.ones((ngx, ngy, ngz))#u_vel
    v_n = np.zeros((ngx, ngy, ngz))
    w_n = np.zeros((ngx, ngy, ngz))
    p_n = np.zeros((ngx, ngy, ngz))

    u_n += 0.01 * np.random.randn(ngx, ngy, ngz) * (1 - (Y / H)**2)
    # v_n += 0.01 * np.random.randn(ngx, ngy, ngz) * (1 - (Y / H)**2)
    # w_n += 0.01 * np.random.randn(ngx, ngy, ngz) * (1 - (Y / H)**2)
    
    # simulation parameters 
    Re = 180 
    num_steps = 1500
    nu = 1.0 / Re 
    dt = 0.001
    U_bulk_phys = 1.0
    U_bulk = 1.0  * ngz 


    alpha   = [29/60, -3/40, 1/6]
    beta    = [37/160, 5/24, 1/6]
    gamma   = [8/15, 5/12, 3/4]
    zeta    = [0.0, -17/60, -5/12]


    left, right, bottom, top = boundary_flag(ngx, ngy)
    wall_idx = np.concatenate([bottom, top])

    # these matrices are not changing 
    Laplacian = [] #[k], this is for viscous term calculation 
    A = [[], [], []] # [substep][k], bc applied for solving 
    Poisson = [] # bc applied for solving
    A_solvers = [[], [], []] # [substep][k]
    Poisson_solvers = [] #[k]

    print("Pre-calculating Laplacian and A matrices and factorizing...")
    for k in range(Nz_fourier):
        Laplacian_ = build_P(operators, operators.kz[k]) 

        Laplacian.append(Laplacian_.tocsr())
        

        Laplacian_bc, _ = apply_neumann(Laplacian_, np.zeros(ngx * ngy), wall_idx, operators)
        if k == 0:
            Laplacian_bc, _ = pin_pressure(Laplacian_bc, np.zeros(ngx * ngy), 0)

        Poisson.append(Laplacian_bc.tocsr())

        Laplacian_solver = PyPardisoSolver()
        Laplacian_solver.factorize(Laplacian_bc.tocsr())
        Poisson_solvers.append(Laplacian_solver)


    for i in range(3): # for beta
        for k in range(Nz_fourier):
            A_ = build_A(operators, nu, dt, beta[i], operators.kz[k])
            A_bc, _ = apply_dirichlet(A_, np.zeros(ngx * ngy), wall_idx, val = 0.0)

            A[i].append(A_bc.tocsr())
            A_solver = PyPardisoSolver()
            A_solver.factorize(A_bc.tocsr())

            A_solvers[i].append(A_solver) # factorized!



    # pre-allocation 
    viscous_u_old   = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_v_old   = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_w_old   = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)

    viscous_u1      = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_v1      = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_w1      = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)

    viscous_u2      = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_v2      = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_w2      = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)

    timesteps = []
    div_res   = []
    inf_err   = []

    output_dir = "figures"
    # time-stepping loop 
    for n in range(num_steps):

        start_time = time.perf_counter()
        print(f"Time-step {n + 1} / {num_steps}")

        u_old, v_old, w_old, p_old = u_n.copy(), v_n.copy(), w_n.copy(), p_n.copy()

        # print("Substep 1")
        # compute nonlinear term 
        Nu_old, Nv_old, Nw_old = compute_nonlinear_term(u_old, v_old, w_old, operators)

        # transfrom u^n to fourier space 
        u_old_hat = np.fft.rfft(u_old, axis = 2)
        v_old_hat = np.fft.rfft(v_old, axis = 2)
        w_old_hat = np.fft.rfft(w_old, axis = 2)
        p_old_hat = np.fft.rfft(p_old, axis = 2)

        # calculate mass flux and pressure gradient
        dp_dx = massflux_correction(u_old, y_coords, U_bulk_phys, dt) 

        # assemble RHS 

        for k in range(Nz_fourier):
            viscous_u_old[:, :, k] = (Laplacian[k] @ u_old_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_v_old[:, :, k] = (Laplacian[k] @ v_old_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_w_old[:, :, k] = (Laplacian[k] @ w_old_hat[:, :, k].flatten()).reshape(ngx, ngy)



        rhs_u1_hat = u_old_hat + dt * (alpha[0] * nu * viscous_u_old + gamma[0] * Nu_old)
        rhs_v1_hat = v_old_hat + dt * (alpha[0] * nu * viscous_v_old + gamma[0] * Nv_old)
        rhs_w1_hat = w_old_hat + dt * (alpha[0] * nu * viscous_w_old + gamma[0] * Nw_old)

        rhs_u1_hat[:, :, 0] += dt * gamma[0] * dp_dx 


        u_tilde1_hat, v_tilde1_hat, w_tilde1_hat = predictor_step(rhs_u1_hat, rhs_v1_hat, rhs_w1_hat, A[0], A_solvers[0], wall_idx)
        phi1_hat = correction_step(u_tilde1_hat, v_tilde1_hat, w_tilde1_hat, operators, dt, Poisson, Poisson_solvers, wall_idx)
        u1_hat, v1_hat, w1_hat, p1_hat = update_step(u_tilde1_hat, v_tilde1_hat, w_tilde1_hat, p_old_hat, phi1_hat, operators, dt)

        if ngz > 1:
            u1_hat[:, :, -1] = 0.0
            v1_hat[:, :, -1] = 0.0
            w1_hat[:, :, -1] = 0.0

        # print(f" Max |u_tilde1|: {np.max(np.abs(u_tilde1_hat)):.2e}")
        # print(f" Max |v_tilde1|: {np.max(np.abs(v_tilde1_hat)):.2e}")
        # print(f" Max |phi1|: {np.max(np.abs(phi1_hat)):.2e}")
        # print(f" Max |u1|: {np.max(np.abs(u1_hat)):.2e}")

        # print("Substep 2")
        u1_phys = np.fft.irfft(u1_hat, n = ngz)
        v1_phys = np.fft.irfft(v1_hat, n = ngz)
        w1_phys = np.fft.irfft(w1_hat, n = ngz)

        # calculate mass flux and pressure gradient
        dp_dx = massflux_correction(u1_phys, y_coords, U_bulk_phys, dt) # NOTE: do we need to correct mass flux for each substep?

        Nu1, Nv1, Nw1 = compute_nonlinear_term(u1_phys, v1_phys, w1_phys, operators)

        for k in range(Nz_fourier):
            viscous_u1[:, :, k] = (Laplacian[k] @ u1_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_v1[:, :, k] = (Laplacian[k] @ v1_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_w1[:, :, k] = (Laplacian[k] @ w1_hat[:, :, k].flatten()).reshape(ngx, ngy)

        rhs_u2_hat = u1_hat + dt * (alpha[1] * nu * viscous_u1  + gamma[1] * Nu1 + zeta[1] * Nu_old)
        rhs_v2_hat = v1_hat + dt * (alpha[1] * nu * viscous_v1  + gamma[1] * Nv1 + zeta[1] * Nv_old)
        rhs_w2_hat = w1_hat + dt * (alpha[1] * nu * viscous_w1  + gamma[1] * Nw1 + zeta[1] * Nw_old)

        rhs_u2_hat[:, :, 0] += dt * gamma[1] * dp_dx 

        u_tilde2_hat, v_tilde2_hat, w_tilde2_hat = predictor_step(rhs_u2_hat, rhs_v2_hat, rhs_w2_hat, A[1], A_solvers[1], wall_idx)
        phi2_hat = correction_step(u_tilde2_hat, v_tilde2_hat, w_tilde2_hat, operators, dt, Poisson, Poisson_solvers, wall_idx)
        u2_hat, v2_hat, w2_hat, p2_hat = update_step(u_tilde2_hat, v_tilde2_hat, w_tilde2_hat, p1_hat, phi2_hat, operators, dt)

        if ngz > 1:
            u2_hat[:, :, -1] = 0.0
            v2_hat[:, :, -1] = 0.0
            w2_hat[:, :, -1] = 0.0
        # print(f" Max |u_tilde2|: {np.max(np.abs(u_tilde2_hat)):.2e}")
        # print(f" Max |v_tilde2|: {np.max(np.abs(v_tilde2_hat)):.2e}")
        # print(f" Max |phi2|: {np.max(np.abs(phi2_hat)):.2e}")
        # print(f" Max |u2|: {np.max(np.abs(u2_hat)):.2e}")

        # print("Substep 3")
        u2_phys = np.fft.irfft(u2_hat, n = ngz)
        v2_phys = np.fft.irfft(v2_hat, n = ngz)
        w2_phys = np.fft.irfft(w2_hat, n = ngz)

        # calculate mass flux and pressure gradient
        dp_dx = massflux_correction(u2_phys, y_coords, U_bulk_phys, dt) 

        Nu2, Nv2, Nw2 = compute_nonlinear_term(u2_phys, v2_phys, w2_phys, operators)

        for k in range(Nz_fourier):
            viscous_u2[:, :, k] = (Laplacian[k] @ u2_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_v2[:, :, k] = (Laplacian[k] @ v2_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_w2[:, :, k] = (Laplacian[k] @ w2_hat[:, :, k].flatten()).reshape(ngx, ngy)

        rhs_u3_hat = u2_hat + dt * (alpha[2] * nu * viscous_u2  + gamma[2] * Nu2 + zeta[2] * Nu1)
        rhs_v3_hat = v2_hat + dt * (alpha[2] * nu * viscous_v2  + gamma[2] * Nv2 + zeta[2] * Nv1)
        rhs_w3_hat = w2_hat + dt * (alpha[2] * nu * viscous_w2  + gamma[2] * Nw2 + zeta[2] * Nw1)

        rhs_u3_hat[:, :, 0] += dt * gamma[2] * dp_dx 

        u_tilde3_hat, v_tilde3_hat, w_tilde3_hat = predictor_step(rhs_u3_hat, rhs_v3_hat, rhs_w3_hat, A[2], A_solvers[2], wall_idx)
        phi3_hat = correction_step(u_tilde3_hat, v_tilde3_hat, w_tilde3_hat,operators, dt, Poisson, Poisson_solvers, wall_idx)
        u3_hat, v3_hat, w3_hat, p3_hat = update_step(u_tilde3_hat, v_tilde3_hat, w_tilde3_hat, p2_hat, phi3_hat, operators, dt)

        # print(f" Max |u_tilde3|: {np.max(np.abs(u_tilde3_hat)):.2e}")
        # print(f" Max |v_tilde3|: {np.max(np.abs(v_tilde3_hat)):.2e}")
        # print(f" Max |phi3|: {np.max(np.abs(phi3_hat)):.2e}")
        # print(f" Max |u3|: {np.max(np.abs(u3_hat)):.2e}")

        if ngz > 1:
            u3_hat[:, :, -1] = 0.0
            v3_hat[:, :, -1] = 0.0
            w3_hat[:, :, -1] = 0.0

        # final update 
        u_n = np.fft.irfft(u3_hat, n = ngz)
        v_n = np.fft.irfft(v3_hat, n = ngz)
        w_n = np.fft.irfft(w3_hat, n = ngz)
        p_n = np.fft.irfft(p3_hat, n = ngz)

        # u_n[:, [0, -1], :] = 0.0
        # v_n[:, [0, -1], :] = 0.0
        # w_n[:, [0, -1], :] = 0.0

        max_u_wall = np.max(np.abs(u_n[:, [0, -1], :]))
        max_v_wall = np.max(np.abs(v_n[:, [0, -1], :]))
        max_w_wall = np.max(np.abs(w_n[:, [0, -1], :]))

        print(f"Max wall |u| : {max_u_wall:.2e}, |v| : {max_v_wall:.2e}, |w|: {max_w_wall:.2e}")

        div_hat = np.zeros_like(u3_hat)
        for k in range(Nz_fourier):
            u_k, v_k, w_k = u3_hat[:,:, k], v3_hat[:, :, k], w3_hat[:, :, k]
            div_k = (operators.Dx @ u_k) + (v_k @ operators.Dy.T) + (1j * operators.kz[k] * w_k)
            div_hat[:, :, k] = div_k 
        max_div = np.max(np.abs(np.fft.irfft(div_hat, n = ngz)))
        print(f"Max divergence: {max_div:.2e}")

        # u_vel is the parabolic velocity profile 
        u_vel = 1.5 * 1.0 * (1 - (Y / H)**2) # 1.0 is U_bulk target and U_bulk = (2/3) * U_max
        rel_err = np.linalg.norm(u_n - u_vel) / np.linalg.norm(u_vel)
        inf_norm = np.max(np.abs(u_n - u_vel)) / np.max(np.abs(u_vel))
        print(f"Relative Error :{rel_err}")
        print(f"Infinity Norm Error :{inf_norm}")

        timesteps.append(n + 1)
        div_res.append(max_div)
        inf_err.append(inf_norm)

        end_time = time.perf_counter()
        print(f"Timestep took :{end_time - start_time} seconds")
        if (n + 1) % 10 == 0:
            plt.figure(figsize = (10,5))
            z_slice = ngz // 2
            plt.pcolormesh(X[:,:,z_slice], Y[:,:,z_slice], u_n[:,:,z_slice], shading = 'gouraud', cmap = 'viridis')
            plt.colorbar(); plt.title(f"U Velocity at Time-step {n + 1}")
            filename = os.path.join(output_dir, f"frame_{n+1:04d}.png")
            plt.savefig(filename, dpi=150); plt.close()

        # fig = go.Figure()
        #
        # vmin = np.min(u_n)
        # vmax = np.max(u_n)
        #
        # for k in range(ngz):
        #     fig.add_trace(go.Surface(x = grid['X'][:,:,k], y= grid['Y'][:,:,k], z = grid['Z'][:,:,k], 
        #                              surfacecolor=u_n[:,:,k], colorscale='Viridis', cmin=vmin, cmax=vmax))
        # # fig.show()
        # fig.write_html('test.html')
        # stop

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(14,6))

    # ax1 = axes[0,0]
    xx = np.array(timesteps) * dt
    ax1 = axes[0]
    ax1.plot(xx, div_res, linewidth = 2)
    ax1.set_yscale('log')
    ax1.set_title('Max divergence history', fontsize = 14)
    ax1.set_xlabel('Time ($t$)', fontsize=14)
    ax1.grid(True,linewidth=0.5, alpha=0.5)

    # ax2 = axes[0,1]
    ax2 = axes[1]
    ax2.plot(xx,inf_err, linewidth = 2)
    ax2.set_title('Velocity Error (Infinity Norm)', fontsize=14)
    ax2.set_xlabel('Time ($t$)', fontsize=14)
    ax2.grid(True,linewidth=0.5, alpha=0.5)
    plt.tight_layout()



    # ax3 = axes[1,0]
    # ax3.plot(timesteps, max_w)
    # ax3.set_title(' Maximum |w| velocity')
    #
    # ax4 = axes[1,1]
    # ax4.plot(timesteps, max_p)
    # ax4.set_title(' Maximum |p|')
    #
    plt.savefig("maxvals.png", dpi = 300)

    plt.show()
    
    
    print("Plotting final velocity profile...")
    
    # Take a slice from the middle of the domain
    z_slice = ngz // 2
    x_slice = ngx // 2
    
    final_u_profile = u_n[x_slice, :, z_slice]
    exact_u_profile = u_vel[x_slice, :, z_slice]
    y_coords = grid['Y'][x_slice, :, z_slice]
    
    plt.figure(figsize=(8, 8))
    plt.plot(exact_u_profile, y_coords, 'r-', label='Parabolic Profile', linewidth=3)
    plt.plot(final_u_profile, y_coords, 'b--o', label='Numerical Profile', markersize=4)
    plt.xlabel('u-velocity', fontsize=14)
    plt.ylabel('y-coordinate', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig("profile_comparison.png", dpi=300)
    plt.show()
