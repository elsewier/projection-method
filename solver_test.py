# this will build matrices for per fourier mode
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres
from pypardiso import spsolve, PyPardisoSolver
import time
import os
import matplotlib.pyplot as plt

# It is assumed that bspline_ops and grid modules are in the same directory.
# You will need to have these files: bspline_ops.py, grid.py
from bspline_ops import BSplineOperator
from grid import create_channel_grid

def build_A(operators, nu, dt, beta, kz):
    # A0 = (I - dt * beta * nu * Laplacian_3D)
    Ix  = sp.eye(operators.Nx, format = 'csr')
    Iy  = sp.eye(operators.Ny, format = 'csr')
    Dxx = sp.csr_matrix(operators.Dxx)
    Dyy = sp.csr_matrix(operators.Dyy)
    Laplacian_2D = sp.kron(Dxx, Iy, format = 'csr') + sp.kron(Ix, Dyy, format = 'csr')

    I_2D = sp.eye(operators.Nx * operators.Ny, format = 'csr')
    A0  = I_2D - dt * beta * nu * (Laplacian_2D - (kz**2) * I_2D)

    A0.sum_duplicates()
    A0.eliminate_zeros()
    return A0

def build_P(operators, kz):
    # Poisson pressure correction : Laplacian_3D
    Ix  = sp.eye(operators.Nx, format = 'csr')
    Iy  = sp.eye(operators.Ny, format = 'csr')
    Dxx = sp.csr_matrix(operators.Dxx)
    Dyy = sp.csr_matrix(operators.Dyy)
    Laplacian_2D = sp.kron(Dxx, Iy, format = 'csr') + sp.kron(Ix, Dyy, format = 'csr')

    I_2D = sp.eye(operators.Nx * operators.Ny, format = 'csr')
    P0  = Laplacian_2D - (kz**2) * I_2D

    P0.sum_duplicates()
    P0.eliminate_zeros()
    return P0


def boundary_flag(Nx, Ny):
    left_idx, right_idx, bottom_idx, top_idx = [], [], [], []

    for i in range(Nx):
        bottom_idx.append(i * Ny)
        top_idx.append(i * Ny + (Ny - 1))
    # Periodic in x, so left/right are not typically used for wall BCs
    for j in range(Ny):
        left_idx.append(j)
        right_idx.append((Nx - 1) * Ny + j)

    return (np.array(left_idx, dtype=np.int32),
            np.array(right_idx, dtype=np.int32),
            np.array(bottom_idx, dtype=np.int32),
            np.array(top_idx, dtype=np.int32))

def apply_dirichlet_rhs(b, idx, val=0.0):
    """Only applies Dirichlet condition to the right-hand side vector."""
    is_scalar = np.isscalar(val)
    for i in range(idx.shape[0]):
        k = int(idx[i])
        b[k] = val[i] if not is_scalar else val
    return b

def apply_dirichlet(A, b, idx, val=0.0):
    """Modifies the matrix A and optionally the vector b for Dirichlet BCs."""
    A_lil = A.tolil()
    is_scalar = np.isscalar(val)
    for i in range(idx.shape[0]):
        k = int(idx[i])
        A_lil.rows[k] = [k]
        A_lil.data[k] = [1.0]
        if b is not None:
            b[k] = val[i] if not is_scalar else val
    A_csr = A_lil.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()
    return A_csr, b

def apply_neumann(A, b, boundary_indices, ops):
    """Modifies matrix A and optionally vector b for Neumann BCs (du/dn=0)."""
    Nx, Ny = ops.Nx, ops.Ny
    Dy_s = sp.csr_matrix(ops.Dy)
    A_lil = A.tolil()

    for idx in boundary_indices:
        i = idx // Ny
        j = idx % Ny
        if j == 0 or j == Ny - 1: # Bottom or top wall
            start, end = Dy_s.indptr[j], Dy_s.indptr[j + 1]
            cols_y, vals_y = Dy_s.indices[start:end], Dy_s.data[start:end]
            A_lil.rows[idx] = list(i * Ny + cols_y)
            A_lil.data[idx] = list(vals_y)
            if b is not None:
                b[idx] = 0.0
    A_csr = A_lil.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()
    return A_csr, b

def pin_pressure(A, b, idx):
    """Pins a single degree of freedom in the pressure matrix."""
    A_lil = A.tolil()
    A_lil.rows[idx] = [idx]
    A_lil.data[idx] = [1.0]
    if b is not None:
        b[idx] = 0.0
    A_csr = A_lil.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()
    return A_csr, b

def compute_nonlinear_term(u, v, w, ops):
    ngx, ngy, ngz = u.shape
    Nz_dealias = int(ngz * 3 / 2)
    Nz_fourier = ops.grid['Nz_fourier']
    nzz_dealias = Nz_dealias // 2 + 1

    u_hat, v_hat, w_hat = [np.fft.rfft(f, axis=2) for f in (u, v, w)]

    u_hat_d = np.zeros((ngx, ngy, nzz_dealias), dtype=np.complex128)
    v_hat_d = np.zeros((ngx, ngy, nzz_dealias), dtype=np.complex128)
    w_hat_d = np.zeros((ngx, ngy, nzz_dealias), dtype=np.complex128)

    u_hat_d[:, :, :Nz_fourier] = u_hat
    v_hat_d[:, :, :Nz_fourier] = v_hat
    w_hat_d[:, :, :Nz_fourier] = w_hat

    u_d, v_d, w_d = [np.fft.irfft(f, n=Nz_dealias, axis=2) for f in (u_hat_d, v_hat_d, w_hat_d)]

    du_dx = (ops.Dx @ u_d.reshape(ngx, -1)).reshape(u_d.shape)
    dv_dx = (ops.Dx @ v_d.reshape(ngx, -1)).reshape(v_d.shape)
    dw_dx = (ops.Dx @ w_d.reshape(ngx, -1)).reshape(w_d.shape)

    du_dy = (u_d.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)
    dv_dy = (v_d.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)
    dw_dy = (w_d.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)

    kzz_d = (2 * np.pi) * np.fft.rfftfreq(Nz_dealias, d=ops.grid['Lz'] / Nz_dealias)
    kzz_3d = kzz_d.reshape(1, 1, -1)

    du_dz = np.fft.irfft(1j * kzz_3d * u_hat_d, n=Nz_dealias)
    dv_dz = np.fft.irfft(1j * kzz_3d * v_hat_d, n=Nz_dealias)
    dw_dz = np.fft.irfft(1j * kzz_3d * w_hat_d, n=Nz_dealias)

    Nu = -(u_d * du_dx + v_d * du_dy + w_d * du_dz)
    Nv = -(u_d * dv_dx + v_d * dv_dy + w_d * dv_dz)
    Nw = -(u_d * dw_dx + v_d * dw_dy + w_d * dw_dz)

    Nu_hat, Nv_hat, Nw_hat = [np.fft.rfft(f, axis=2) for f in (Nu, Nv, Nw)]

    Nu_hat1 = Nu_hat[:, :, :Nz_fourier]
    Nv_hat1 = Nv_hat[:, :, :Nz_fourier]
    Nw_hat1 = Nw_hat[:, :, :Nz_fourier]

    if ngz > 1 and Nz_fourier > 1:
        Nu_hat1[:, :, -1] = 0.0
        Nv_hat1[:, :, -1] = 0.0
        Nw_hat1[:, :, -1] = 0.0

    return Nu_hat1, Nv_hat1, Nw_hat1

def predictor_step(rhs_u, rhs_v, rhs_w, A_matrices, solvers, wall_idx):
    ngx, ngy, nz = rhs_u.shape
    u_tilde_hat, v_tilde_hat, w_tilde_hat = [np.zeros_like(rhs_u) for _ in range(3)]

    for k in range(nz):
        rhs_u_k, rhs_v_k, rhs_w_k = [f[:, :, k].flatten() for f in (rhs_u, rhs_v, rhs_w)]
        A = A_matrices[k]
        pardiso_solver = solvers[k]
        
        rhs_u_k = apply_dirichlet_rhs(rhs_u_k, wall_idx, val=0.0)
        rhs_v_k = apply_dirichlet_rhs(rhs_v_k, wall_idx, val=0.0)
        rhs_w_k = apply_dirichlet_rhs(rhs_w_k, wall_idx, val=0.0)

        # CORRECTED: Pardiso solver call must be solve(A, b)
        u_tilde_vec = pardiso_solver.solve(A, rhs_u_k.real) + 1j * pardiso_solver.solve(A, rhs_u_k.imag)
        v_tilde_vec = pardiso_solver.solve(A, rhs_v_k.real) + 1j * pardiso_solver.solve(A, rhs_v_k.imag)
        w_tilde_vec = pardiso_solver.solve(A, rhs_w_k.real) + 1j * pardiso_solver.solve(A, rhs_w_k.imag)

        u_tilde_hat[:, :, k] = u_tilde_vec.reshape(ngx, ngy)
        v_tilde_hat[:, :, k] = v_tilde_vec.reshape(ngx, ngy)
        w_tilde_hat[:, :, k] = w_tilde_vec.reshape(ngx, ngy)

    return u_tilde_hat, v_tilde_hat, w_tilde_hat

def correction_step(u_tilde_hat, v_tilde_hat, w_tilde_hat, ops, dt, Poisson_matrices, Poisson_solvers, wall_idx):
    ngx, ngy, nz = u_tilde_hat.shape
    phi_hat = np.zeros_like(u_tilde_hat)

    for k in range(nz):
        u_tilde_k, v_tilde_k, w_tilde_k = [f[:, :, k] for f in (u_tilde_hat, v_tilde_hat, w_tilde_hat)]

        div = (ops.Dx @ u_tilde_k) + (v_tilde_k @ ops.Dy.T) + (1j * ops.kz[k] * w_tilde_k)
        rhs_p_k = (1 / dt) * div.flatten()
        P = Poisson_matrices[k]
        pardiso_solver = Poisson_solvers[k]

        if k == 0:
            rhs_p_k[wall_idx] = 0.0
            rhs_p_k -= np.mean(rhs_p_k)
            rhs_p_k[0] = 0.0
        else:
            rhs_p_k = apply_dirichlet_rhs(rhs_p_k, wall_idx, val=0.0)

        # CORRECTED: Pardiso solver call must be solve(A, b)
        phi_k = pardiso_solver.solve(P, rhs_p_k.real) + 1j * pardiso_solver.solve(P, rhs_p_k.imag)
        phi_hat[:, :, k] = phi_k.reshape(ngx, ngy)

    return phi_hat

def update_step(u_tilde_hat, v_tilde_hat, w_tilde_hat, p_hat, phi_hat, ops, dt):
    grad_phi_x = np.zeros_like(phi_hat)
    grad_phi_y = np.zeros_like(phi_hat)
    for k in range(phi_hat.shape[2]):
        grad_phi_x[:, :, k] = ops.Dx @ phi_hat[:, :, k]
        grad_phi_y[:, :, k] = phi_hat[:, :, k] @ ops.Dy.T

    grad_phi_z = 1j * ops.grid['kz_3d'] * phi_hat

    u_new_hat = u_tilde_hat - dt * grad_phi_x
    v_new_hat = v_tilde_hat - dt * grad_phi_y
    w_new_hat = w_tilde_hat - dt * grad_phi_z
    p_new_hat = p_hat + phi_hat

    return u_new_hat, v_new_hat, w_new_hat, p_new_hat

def massflux_correction(u_phys, y_coords, U_bulk_target, dt):
    u_y_avg = np.mean(u_phys, axis=(0, 2))
    U_bulk_curr = np.trapezoid(u_y_avg, y_coords) / (y_coords[-1] - y_coords[0])
    dp_dx = (U_bulk_target - U_bulk_curr) / dt
    print(f"U_current = {U_bulk_curr:.6f}, U_target = {U_bulk_target:.6f}, Required dp/dx = {dp_dx:.6f}")
    return dp_dx

if __name__ == '__main__':
    p, q = 7, 7
    ngx, ngy, ngz = 128, 128, 1
    Lx, H, Lz = 4.0 * np.pi, 1.0, 1.0 * np.pi

    grid = create_channel_grid(Nx=ngx, Ny=ngy, Nz=ngz, Lx=Lx, H=H, Lz=Lz, p=p, q=q, stretch_factor=1.5)
    operators = BSplineOperator(grid, p=p, q=q, periodic_x=True)
    y_coords = grid['Y'][0, :, 0]
    Nz_fourier = grid['Nz_fourier']
    X, Y, Z = grid['X'], grid['Y'], grid['Z']

    u_vel = 1.5 * (1 - (Y / H)**2)
    u_n, v_n, w_n, p_n = u_vel.copy(), np.zeros_like(u_vel), np.zeros_like(u_vel), np.zeros_like(u_vel)

    Re = 180
    num_steps = 300
    nu = 1.0 / Re
    dt = 0.005
    U_bulk_phys = 1.0

    alpha = [29/60, -3/40, 1/6]
    beta = [37/160, 5/24, 1/6]
    gamma = [8/15, 5/12, 3/4]
    zeta = [0.0, -17/60, -5/12]

    _, _, bottom, top = boundary_flag(ngx, ngy)
    wall_idx = np.concatenate([bottom, top])

    Laplacian = []
    Poisson = []
    A_bc = [[], [], []] # Store BC-applied matrices for predictor
    A_solvers = [[], [], []]
    Poisson_solvers = []

    print("Pre-calculating and factorizing matrices with boundary conditions...")
    for k in range(Nz_fourier):
        Laplacian_k = build_P(operators, operators.kz[k])
        Laplacian.append(Laplacian_k)
        
        P_k_bc = Laplacian_k.copy()
        if k == 0:
            P_k_bc, _ = apply_neumann(P_k_bc, None, wall_idx, operators)
            P_k_bc, _ = pin_pressure(P_k_bc, None, 0)
        else:
            P_k_bc, _ = apply_dirichlet(P_k_bc, None, wall_idx, val=0.0)
        Poisson.append(P_k_bc)
        
        poisson_solver = PyPardisoSolver()
        poisson_solver.factorize(P_k_bc)
        Poisson_solvers.append(poisson_solver)

        for i in range(3):
            A_k = build_A(operators, nu, dt, beta[i], operators.kz[k])
            A_k_bc, _ = apply_dirichlet(A_k, None, wall_idx, val=0.0)
            A_bc[i].append(A_k_bc)

            A_solver = PyPardisoSolver()
            A_solver.factorize(A_k_bc)
            A_solvers[i].append(A_solver)

    viscous_u_old, viscous_v_old, viscous_w_old = [np.zeros((ngx, ngy, Nz_fourier), dtype=np.complex128) for _ in range(3)]
    viscous_u1, viscous_v1, viscous_w1 = [np.zeros((ngx, ngy, Nz_fourier), dtype=np.complex128) for _ in range(3)]
    viscous_u2, viscous_v2, viscous_w2 = [np.zeros((ngx, ngy, Nz_fourier), dtype=np.complex128) for _ in range(3)]

    output_dir = "figures"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for n in range(num_steps):
        start_time = time.perf_counter()
        print(f"\nTime-step {n + 1} / {num_steps}")

        u_old, v_old, w_old, p_old = u_n.copy(), v_n.copy(), w_n.copy(), p_n.copy()

        # Substep 1
        Nu_old, Nv_old, Nw_old = compute_nonlinear_term(u_old, v_old, w_old, operators)
        u_old_hat, v_old_hat, w_old_hat, p_old_hat = [np.fft.rfft(f, axis=2) for f in (u_old, v_old, w_old, p_old)]
        dp_dx = massflux_correction(u_old, y_coords, U_bulk_phys, dt)
        for k in range(Nz_fourier):
            viscous_u_old[:, :, k] = (Laplacian[k] @ u_old_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_v_old[:, :, k] = (Laplacian[k] @ v_old_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_w_old[:, :, k] = (Laplacian[k] @ w_old_hat[:, :, k].flatten()).reshape(ngx, ngy)
        rhs_u1_hat = u_old_hat + dt * (alpha[0] * nu * viscous_u_old + gamma[0] * Nu_old)
        rhs_v1_hat = v_old_hat + dt * (alpha[0] * nu * viscous_v_old + gamma[0] * Nv_old)
        rhs_w1_hat = w_old_hat + dt * (alpha[0] * nu * viscous_w_old + gamma[0] * Nw_old)
        if Nz_fourier > 0: rhs_u1_hat[:, :, 0] += dt * (gamma[0] / sum(gamma)) * dp_dx
        u_tilde1_hat, v_tilde1_hat, w_tilde1_hat = predictor_step(rhs_u1_hat, rhs_v1_hat, rhs_w1_hat, A_bc[0], A_solvers[0], wall_idx)
        phi1_hat = correction_step(u_tilde1_hat, v_tilde1_hat, w_tilde1_hat, operators, dt, Poisson, Poisson_solvers, wall_idx)
        u1_hat, v1_hat, w1_hat, p1_hat = update_step(u_tilde1_hat, v_tilde1_hat, w_tilde1_hat, p_old_hat, phi1_hat, operators, dt)
        if ngz > 1 and Nz_fourier > 1: u1_hat[:, :, -1], v1_hat[:, :, -1], w1_hat[:, :, -1] = 0, 0, 0
        u1_phys, v1_phys, w1_phys = [np.fft.irfft(f, n=ngz) for f in (u1_hat, v1_hat, w1_hat)]

        # Substep 2
        dp_dx = massflux_correction(u1_phys, y_coords, U_bulk_phys, dt)
        Nu1, Nv1, Nw1 = compute_nonlinear_term(u1_phys, v1_phys, w1_phys, operators)
        for k in range(Nz_fourier):
            viscous_u1[:, :, k] = (Laplacian[k] @ u1_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_v1[:, :, k] = (Laplacian[k] @ v1_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_w1[:, :, k] = (Laplacian[k] @ w1_hat[:, :, k].flatten()).reshape(ngx, ngy)
        rhs_u2_hat = u1_hat + dt * (alpha[1] * nu * viscous_u1 + gamma[1] * Nu1 + zeta[1] * Nu_old)
        rhs_v2_hat = v1_hat + dt * (alpha[1] * nu * viscous_v1 + gamma[1] * Nv1 + zeta[1] * Nv_old)
        rhs_w2_hat = w1_hat + dt * (alpha[1] * nu * viscous_w1 + gamma[1] * Nw1 + zeta[1] * Nw_old)
        if Nz_fourier > 0: rhs_u2_hat[:, :, 0] += dt * (gamma[1] / sum(gamma)) * dp_dx
        u_tilde2_hat, v_tilde2_hat, w_tilde2_hat = predictor_step(rhs_u2_hat, rhs_v2_hat, rhs_w2_hat, A_bc[1], A_solvers[1], wall_idx)
        phi2_hat = correction_step(u_tilde2_hat, v_tilde2_hat, w_tilde2_hat, operators, dt, Poisson, Poisson_solvers, wall_idx)
        u2_hat, v2_hat, w2_hat, p2_hat = update_step(u_tilde2_hat, v_tilde2_hat, w_tilde2_hat, p1_hat, phi2_hat, operators, dt)
        if ngz > 1 and Nz_fourier > 1: u2_hat[:, :, -1], v2_hat[:, :, -1], w2_hat[:, :, -1] = 0, 0, 0
        u2_phys, v2_phys, w2_phys = [np.fft.irfft(f, n=ngz) for f in (u2_hat, v2_hat, w2_hat)]

        # Substep 3
        dp_dx = massflux_correction(u2_phys, y_coords, U_bulk_phys, dt)
        Nu2, Nv2, Nw2 = compute_nonlinear_term(u2_phys, v2_phys, w2_phys, operators)
        for k in range(Nz_fourier):
            viscous_u2[:, :, k] = (Laplacian[k] @ u2_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_v2[:, :, k] = (Laplacian[k] @ v2_hat[:, :, k].flatten()).reshape(ngx, ngy)
            viscous_w2[:, :, k] = (Laplacian[k] @ w2_hat[:, :, k].flatten()).reshape(ngx, ngy)
        rhs_u3_hat = u2_hat + dt * (alpha[2] * nu * viscous_u2 + gamma[2] * Nu2 + zeta[2] * Nu1)
        rhs_v3_hat = v2_hat + dt * (alpha[2] * nu * viscous_v2 + gamma[2] * Nv2 + zeta[2] * Nv1)
        rhs_w3_hat = w2_hat + dt * (alpha[2] * nu * viscous_w2 + gamma[2] * Nw2 + zeta[2] * Nw1)
        if Nz_fourier > 0: rhs_u3_hat[:, :, 0] += dt * (gamma[2] / sum(gamma)) * dp_dx
        u_tilde3_hat, v_tilde3_hat, w_tilde3_hat = predictor_step(rhs_u3_hat, rhs_v3_hat, rhs_w3_hat, A_bc[2], A_solvers[2], wall_idx)
        phi3_hat = correction_step(u_tilde3_hat, v_tilde3_hat, w_tilde3_hat, operators, dt, Poisson, Poisson_solvers, wall_idx)
        u_n_hat, v_n_hat, w_n_hat, p_n_hat = update_step(u_tilde3_hat, v_tilde3_hat, w_tilde3_hat, p2_hat, phi3_hat, operators, dt)
        if ngz > 1 and Nz_fourier > 1: u_n_hat[:, :, -1], v_n_hat[:, :, -1], w_n_hat[:, :, -1] = 0, 0, 0

        u_n, v_n, w_n, p_n = [np.fft.irfft(f, n=ngz) for f in (u_n_hat, v_n_hat, w_n_hat, p_n_hat)]

        max_u_wall = np.max(np.abs(u_n[:, [0, -1], :]))
        max_v_wall = np.max(np.abs(v_n[:, [0, -1], :]))
        max_w_wall = np.max(np.abs(w_n[:, [0, -1], :]))
        print(f"Max wall |u|: {max_u_wall:.2e}, |v|: {max_v_wall:.2e}, |w|: {max_w_wall:.2e}")
        
        div_u_hat = np.zeros_like(u_n_hat)
        for k in range(Nz_fourier):
            u_k, v_k, w_k = u_n_hat[:,:,k], v_n_hat[:,:,k], w_n_hat[:,:,k]
            div_k = (operators.Dx @ u_k) + (v_k @ operators.Dy.T) + (1j * operators.kz[k] * w_k)
            div_u_hat[:,:,k] = div_k
        max_div = np.max(np.abs(np.fft.irfft(div_u_hat, n=ngz)))
        print(f"Max divergence: {max_div:.2e}")

        end_time = time.perf_counter()
        print(f"Timestep took: {end_time - start_time:.4f} seconds")

        if (n + 1) % 10 == 0:
            plt.figure(figsize=(10, 5))
            z_slice = ngz // 2
            plt.pcolormesh(X[:, :, z_slice], Y[:, :, z_slice], u_n[:, :, z_slice], shading='gouraud', cmap='viridis')
            plt.colorbar(); plt.title(f"U Velocity at Time-step {n + 1}")
            filename = os.path.join(output_dir, f"frame_{n+1:04d}.png")
            plt.savefig(filename, dpi=150); plt.close()

    print("\nPlotting final velocity profile...")
    z_slice, x_slice = ngz // 2, ngx // 2
    final_u_profile = u_n[x_slice, :, z_slice]
    exact_u_profile = u_vel[x_slice, :, z_slice]
    y_coords_slice = grid['Y'][x_slice, :, z_slice]
    
    plt.figure(figsize=(8, 8))
    plt.plot(exact_u_profile, y_coords_slice, 'r-', label='Initial Parabolic Profile', linewidth=3)
    plt.plot(final_u_profile, y_coords_slice, 'b--o', label=f'Numerical Profile (t={num_steps*dt})', markersize=4)
    plt.xlabel('u-velocity'); plt.ylabel('y-coordinate')
    plt.title('Velocity Profile Comparison'); plt.legend(); plt.grid(True)
    plt.savefig("profile_comparison.png", dpi=300)
    plt.show()
