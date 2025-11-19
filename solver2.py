import numpy as np
import scipy.sparse as sp 
import matplotlib.pyplot as plt
from pypardiso import PyPardisoSolver
import time
import os

from operators import build_A, build_P, apply_dirichlet, apply_dirichlet_rhs, apply_neumann, boundary_flag, pin_pressure
from bspline_ops import BSplineOperator 
from grid import create_channel_grid 

# --- OPTIMIZATION 1: Pass Fourier data in, return Fourier data ---
def compute_nonlinear_term(u_hat, v_hat, w_hat, ops):
    # N(u, p) = -(u.grad)u 
    
    # 1. Dealiasing Setup (3/2 rule for Z-direction only)
    # Note: standard spectral elements don't usually dealias X/Y (B-spline directions)
    # unless using quadrature integration. We stick to Z-dealiasing here.
    
    ngx = ops.Nx
    ngy = ops.Ny
    Nz_fourier = ops.grid['Nz_fourier'] # Original modes
    Nz_phys = ops.Nz # Original physical size
    
    # Extended grid for dealiasing
    Nz_deal = int(Nz_phys * 3 / 2) 
    
    # 2. Pad Fourier modes to 3/2 size
    u_hat_pad = np.zeros((ngx, ngy, Nz_deal//2 + 1), dtype=np.complex128)
    v_hat_pad = np.zeros((ngx, ngy, Nz_deal//2 + 1), dtype=np.complex128)
    w_hat_pad = np.zeros((ngx, ngy, Nz_deal//2 + 1), dtype=np.complex128)
    
    u_hat_pad[:, :, :Nz_fourier] = u_hat
    v_hat_pad[:, :, :Nz_fourier] = v_hat
    w_hat_pad[:, :, :Nz_fourier] = w_hat

    # 3. Transform to Physical Space on Dealiased Grid
    u_phys = np.fft.irfft(u_hat_pad, n=Nz_deal, axis=2)
    v_phys = np.fft.irfft(v_hat_pad, n=Nz_deal, axis=2)
    w_phys = np.fft.irfft(w_hat_pad, n=Nz_deal, axis=2)

    # 4. Compute Derivatives (Broadcasting & Reshaping)
    
    # Dx: (Nx, Nx) @ (Nx, Y*Z)
    du_dx = (ops.Dx @ u_phys.reshape(ngx, -1)).reshape(u_phys.shape)
    dv_dx = (ops.Dx @ v_phys.reshape(ngx, -1)).reshape(v_phys.shape)
    dw_dx = (ops.Dx @ w_phys.reshape(ngx, -1)).reshape(w_phys.shape)
    
    # Dy: u @ Dy.T
    # Reshape to (Nx*Nz, Ny) so we can apply Dy.T on the right
    # Note: u_phys is (Nx, Ny, Nz). We need Ny last -> transpose(0, 2, 1) -> (Nx, Nz, Ny)
    u_perm = u_phys.transpose(0, 2, 1)
    v_perm = v_phys.transpose(0, 2, 1)
    w_perm = w_phys.transpose(0, 2, 1)
    
    # Apply Dy (Right multiply by Transpose of Derivative Matrix)
    du_dy = (u_perm @ ops.Dy.T).transpose(0, 2, 1)
    dv_dy = (v_perm @ ops.Dy.T).transpose(0, 2, 1)
    dw_dy = (w_perm @ ops.Dy.T).transpose(0, 2, 1)

    # Dz: Spectral derivative
    # Create kz for the dealiased grid
    kzz = (2 * np.pi) * np.fft.rfftfreq(Nz_deal, d=ops.grid['Lz'] / ops.grid['Nz'])
    kzz_3d = kzz.reshape(1, 1, -1)

    du_dz = np.fft.irfft(1j * kzz_3d * u_hat_pad, n=Nz_deal, axis=2)
    dv_dz = np.fft.irfft(1j * kzz_3d * v_hat_pad, n=Nz_deal, axis=2)
    dw_dz = np.fft.irfft(1j * kzz_3d * w_hat_pad, n=Nz_deal, axis=2)

    # 5. Compute Nonlinear Term
    Nu = - (u_phys * du_dx + v_phys * du_dy + w_phys * du_dz)
    Nv = - (u_phys * dv_dx + v_phys * dv_dy + w_phys * dv_dz)
    Nw = - (u_phys * dw_dx + v_phys * dw_dy + w_phys * dw_dz)

    # 6. Transform back and truncation
    Nu_hat_pad = np.fft.rfft(Nu, axis=2)
    Nv_hat_pad = np.fft.rfft(Nv, axis=2)
    Nw_hat_pad = np.fft.rfft(Nw, axis=2)

    # Extract original modes
    Nu_hat = Nu_hat_pad[:, :, :Nz_fourier]
    Nv_hat = Nv_hat_pad[:, :, :Nz_fourier]
    Nw_hat = Nw_hat_pad[:, :, :Nz_fourier]
    
    # Zero out Nyquist if needed (usually handled by slice, but safe to keep)
    if Nz_phys % 2 == 0:
        Nu_hat[:, :, -1] = 0.0
        Nv_hat[:, :, -1] = 0.0
        Nw_hat[:, :, -1] = 0.0

    return Nu_hat, Nv_hat, Nw_hat

def predictor_step(rhs_u, rhs_v, rhs_w, A_matrices, solver, wall_idx):
    ngx, ngy, nz = rhs_u.shape
    u_tilde_hat = np.zeros_like(rhs_u)
    v_tilde_hat = np.zeros_like(rhs_v)
    w_tilde_hat = np.zeros_like(rhs_w)

    for k in range(nz): 
        # Vectorizing this loop is hard with Pardiso, keeping loop is fine for 1D decomposition
        rhs_u_k = apply_dirichlet_rhs(rhs_u[:, :, k].flatten(), wall_idx, val = 0.0)
        rhs_v_k = apply_dirichlet_rhs(rhs_v[:, :, k].flatten(), wall_idx, val = 0.0)
        rhs_w_k = apply_dirichlet_rhs(rhs_w[:, :, k].flatten(), wall_idx, val = 0.0)

        A = A_matrices[k] 
        pardiso_solver = solver[k]

        # Solve (Real and Imaginary parts separately)
        u_vec = pardiso_solver.solve(A, rhs_u_k.real) + 1j * pardiso_solver.solve(A, rhs_u_k.imag)
        v_vec = pardiso_solver.solve(A, rhs_v_k.real) + 1j * pardiso_solver.solve(A, rhs_v_k.imag)
        w_vec = pardiso_solver.solve(A, rhs_w_k.real) + 1j * pardiso_solver.solve(A, rhs_w_k.imag)

        u_tilde_hat[:, :, k] = u_vec.reshape(ngx, ngy)
        v_tilde_hat[:, :, k] = v_vec.reshape(ngx, ngy)
        w_tilde_hat[:, :, k] = w_vec.reshape(ngx, ngy)

    return u_tilde_hat, v_tilde_hat, w_tilde_hat

def correction_step(u_tilde, v_tilde, w_tilde, ops, dt, Poisson_solvers, wall_idx):
    # Calculate Divergence of u_tilde
    # Div = Dx*u + v*Dy_T + dz*w
    
    ngx, ngy, nz = u_tilde.shape
    
    # Dx @ u (Nx, Nx) @ (Nx, Ny) -> (Nx, Ny)
    du_dx = (ops.Dx @ u_tilde.reshape(ngx, -1)).reshape(ngx, ngy, nz)
    
    # v @ Dy.T (Ny, Ny) 
    # We transpose to (Nx, Nz, Ny) to apply Dy to the last axis efficiently
    dv_dy = (v_tilde.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)
    
    dw_dz = 1j * ops.grid['kz_3d'] * w_tilde

    div = du_dx + dv_dy + dw_dz 
    
    phi_hat = np.zeros_like(u_tilde)

    for k in range(nz):
        rhs_p_k = (1.0 / dt) * div[:, :, k].flatten()
        
        # --- BOUNDARY CONDITION CORRECTION ---
        # Ideally, since v_tilde satisfies Dirichlet (0 at wall), div at wall is 0.
        # We apply Homogeneous Neumann for pressure correction.
        # The RHS of Poisson eq at the boundary node should be 0 if using FEM/FDM Neumann construction.
        rhs_p_k[wall_idx] = 0.0 
        
        # Pin Pressure at one point for k=0 mode (singular matrix otherwise)
        if k == 0:
            rhs_p_k -= np.mean(rhs_p_k)
            rhs_p_k[0] = 0.0 

        pardiso_solver = Poisson_solvers[k]
        phi_k = pardiso_solver.solve(None, rhs_p_k.real) + 1j * pardiso_solver.solve(None, rhs_p_k.imag)
        phi_hat[:, :, k] = phi_k.reshape(ngx, ngy) 

    return phi_hat

def update_step(u_tilde, v_tilde, w_tilde, p_old, phi, ops, dt):
    # Grad Phi
    grad_phi_x = (ops.Dx @ phi.reshape(ops.Nx, -1)).reshape(phi.shape)
    grad_phi_y = (phi.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)
    grad_phi_z = 1j * ops.grid['kz_3d'] * phi 

    u_new = u_tilde - dt * grad_phi_x
    v_new = v_tilde - dt * grad_phi_y
    w_new = w_tilde - dt * grad_phi_z
    p_new = p_old + phi 

    return u_new, v_new, w_new, p_new

def massflux_correction(u_phys, y_coords, U_bulk_target, dt):
    # Calculate current bulk velocity
    # Mean over X and Z, then integrate over Y
    u_profile = np.mean(u_phys, axis=(0, 2))
    # Use numpy 2.0 trapezoid or legacy trapz
    try:
        U_bulk_curr = np.trapezoid(u_profile, y_coords) / (y_coords[-1] - y_coords[0])
    except AttributeError:
        U_bulk_curr = np.trapz(u_profile, y_coords) / (y_coords[-1] - y_coords[0])

    # Determine forcing required to bridge the gap in one timestep
    dp_dx = (U_bulk_target - U_bulk_curr) / dt 
    return dp_dx

if __name__ == '__main__':
    # --- Grid Setup ---
    p, q = 7, 7
    ngx, ngy, ngz = 64, 64, 1
    Lx, H, Lz = 4.0 * np.pi, 1.0, 1.0 * np.pi / 4
    
    # Important: periodic_x=True ensures Dx matrix wraps around
    grid = create_channel_grid(Nx=ngx, Ny=ngy, Nz=ngz, Lx=Lx, H=H, Lz=Lz, p=p, q=q, stretch_factor=0.99, periodic_x=True)
    operators = BSplineOperator(grid, p=p, q=q, periodic_x=True)

    # --- Initial Conditions ---
    y_coords = grid['Y'][0, :, 0]
    Nz_fourier = grid['Nz_fourier']
    
    # Start with Laminar Parabolic Profile
    U_bulk_target = 1.0
    u_phys = 1.5 * U_bulk_target * (1 - (grid['Y'] / H)**2)
    v_phys = np.zeros_like(u_phys)
    w_phys = np.zeros_like(u_phys)
    p_phys = np.zeros_like(u_phys)

    # Add small noise to trigger turbulence (if Re is high enough)
    noise_level = 0.0 # Set to 0.01 for turbulence transition
    u_phys += noise_level * np.random.randn(*u_phys.shape) * (1 - (grid['Y']/H)**2)

    # Move to Fourier Space immediately
    u_hat = np.fft.rfft(u_phys, axis=2)
    v_hat = np.fft.rfft(v_phys, axis=2)
    w_hat = np.fft.rfft(w_phys, axis=2)
    p_hat = np.fft.rfft(p_phys, axis=2)

    # --- Simulation Parameters ---
    Re = 180 
    nu = 1.0 / Re 
    dt = 0.005
    num_steps = 50
    
    # RK3 Coefficients (Spalart)
    alpha   = [29/60, -5/12, 1/6]  # Note: I corrected alpha[1] based on standard Spalart/Moser
    beta    = [29/60, 5/12, 1/6]   # Often alpha+beta combined, check your specific source
    # User provided:
    alpha   = [29/60, -3/40, 1/6]
    beta    = [37/160, 5/24, 1/6] # These define the LHS implicit factor
    gamma   = [8/15, 5/12, 3/4]
    zeta    = [0.0, -17/60, -5/12]

    # --- Precompute Matrices ---
    left, right, bottom, top = boundary_flag(ngx, ngy)
    wall_idx = np.concatenate([bottom, top])

    Laplacian = [] 
    A = [[], [], []] 
    Poisson_solvers = []
    A_solvers = [[], [], []]

    print("Factorizing Matrices...")
    for k in range(Nz_fourier):
        # 1. Poisson Laplacian (Neumann)
        Lap_P = build_P(operators, operators.kz[k]) 
        Lap_P_bc, _ = apply_neumann(Lap_P, np.zeros(ngx*ngy), wall_idx, operators)
        if k == 0:
            Lap_P_bc, _ = pin_pressure(Lap_P_bc, np.zeros(ngx*ngy), 0)
        
        P_solver = PyPardisoSolver()
        P_solver.factorize(Lap_P_bc.tocsr())
        Poisson_solvers.append(P_solver)
        
        # Store raw Laplacian for Explicit Viscous term calculation
        Laplacian.append(Lap_P.tocsr()) 

    for i in range(3): # Substeps
        for k in range(Nz_fourier):
            # 2. Helmholtz Matrix (Dirichlet)
            # LHS = I - beta * dt * nu * L
            A_mat = build_A(operators, nu, dt, beta[i], operators.kz[k])
            A_bc, _ = apply_dirichlet(A_mat, np.zeros(ngx*ngy), wall_idx, val=0.0)
            
            A_sol = PyPardisoSolver()
            A_sol.factorize(A_bc.tocsr())
            A_solvers[i].append(A_sol)

    # --- Time Loop ---
    viscous_u = [np.zeros_like(u_hat), np.zeros_like(u_hat), np.zeros_like(u_hat)]
    viscous_v = [np.zeros_like(v_hat), np.zeros_like(v_hat), np.zeros_like(v_hat)]
    viscous_w = [np.zeros_like(w_hat), np.zeros_like(w_hat), np.zeros_like(w_hat)]
    
    nonlinear_u = [None, None, None]
    nonlinear_v = [None, None, None]
    nonlinear_w = [None, None, None]

    timesteps, div_history = [], []
    
    if not os.path.exists("figures"): os.makedirs("figures")

    for n in range(num_steps):
        t0 = time.perf_counter()
        
        # --- Substep 1 ---
        # 1. Nonlinear N(u^n)
        # Note: We pass u_hat directly now, saving FFTs
        Nu_old, Nv_old, Nw_old = compute_nonlinear_term(u_hat, v_hat, w_hat, operators)
        nonlinear_u[0], nonlinear_v[0], nonlinear_w[0] = Nu_old, Nv_old, Nw_old
        
        # 2. Explicit Viscous L(u^n)
        for k in range(Nz_fourier):
            viscous_u[0][:,:,k] = (Laplacian[k] @ u_hat[:,:,k].flatten()).reshape(ngx, ngy)
            viscous_v[0][:,:,k] = (Laplacian[k] @ v_hat[:,:,k].flatten()).reshape(ngx, ngy)
            viscous_w[0][:,:,k] = (Laplacian[k] @ w_hat[:,:,k].flatten()).reshape(ngx, ngy)

        # 3. Mass Flux
        u_phys = np.fft.irfft(u_hat, n=operators.Nz, axis=2)
        dp_dx = massflux_correction(u_phys, y_coords, U_bulk_target, dt)
        
        # 4. Assembly RHS
        rhs_u = u_hat + dt * (alpha[0]*nu*viscous_u[0] + gamma[0]*Nu_old)
        rhs_v = v_hat + dt * (alpha[0]*nu*viscous_v[0] + gamma[0]*Nv_old)
        rhs_w = w_hat + dt * (alpha[0]*nu*viscous_w[0] + gamma[0]*Nw_old)
        rhs_u[:,:,0] += dt * gamma[0] * dp_dx 

        # 5. Predictor - Corrector - Update
        u_til, v_til, w_til = predictor_step(rhs_u, rhs_v, rhs_w, A[0], A_solvers[0], wall_idx)
        phi = correction_step(u_til, v_til, w_til, operators, dt, Poisson_solvers, wall_idx)
        u_hat, v_hat, w_hat, p_hat = update_step(u_til, v_til, w_til, p_hat, phi, operators, dt)
        
        # Apply BCs in spectral space (Nyquist kill)
        if ngz > 1: u_hat[:,:,-1] = v_hat[:,:,-1] = w_hat[:,:,-1] = 0.0

        # --- Substep 2 ---
        # We use u_hat from end of substep 1
        Nu_curr, Nv_curr, Nw_curr = compute_nonlinear_term(u_hat, v_hat, w_hat, operators)
        nonlinear_u[1], nonlinear_v[1], nonlinear_w[1] = Nu_curr, Nv_curr, Nw_curr
        
        for k in range(Nz_fourier):
            viscous_u[1][:,:,k] = (Laplacian[k] @ u_hat[:,:,k].flatten()).reshape(ngx, ngy)
            viscous_v[1][:,:,k] = (Laplacian[k] @ v_hat[:,:,k].flatten()).reshape(ngx, ngy)
            viscous_w[1][:,:,k] = (Laplacian[k] @ w_hat[:,:,k].flatten()).reshape(ngx, ngy)

        u_phys = np.fft.irfft(u_hat, n=ops.Nz, axis=2)
        dp_dx = massflux_correction(u_phys, y_coords, U_bulk_target, dt)

        rhs_u = u_hat + dt * (alpha[1]*nu*viscous_u[1] + gamma[1]*Nu_curr + zeta[1]*nonlinear_u[0])
        rhs_v = v_hat + dt * (alpha[1]*nu*viscous_v[1] + gamma[1]*Nv_curr + zeta[1]*nonlinear_v[0])
        rhs_w = w_hat + dt * (alpha[1]*nu*viscous_w[1] + gamma[1]*Nw_curr + zeta[1]*nonlinear_w[0])
        rhs_u[:,:,0] += dt * gamma[1] * dp_dx

        u_til, v_til, w_til = predictor_step(rhs_u, rhs_v, rhs_w, A[1], A_solvers[1], wall_idx)
        phi = correction_step(u_til, v_til, w_til, operators, dt, Poisson_solvers, wall_idx)
        u_hat, v_hat, w_hat, p_hat = update_step(u_til, v_til, w_til, p_hat, phi, operators, dt)
        if ngz > 1: u_hat[:,:,-1] = v_hat[:,:,-1] = w_hat[:,:,-1] = 0.0

        # --- Substep 3 ---
        Nu_curr, Nv_curr, Nw_curr = compute_nonlinear_term(u_hat, v_hat, w_hat, operators)
        
        for k in range(Nz_fourier):
            viscous_u[2][:,:,k] = (Laplacian[k] @ u_hat[:,:,k].flatten()).reshape(ngx, ngy)
            viscous_v[2][:,:,k] = (Laplacian[k] @ v_hat[:,:,k].flatten()).reshape(ngx, ngy)
            viscous_w[2][:,:,k] = (Laplacian[k] @ w_hat[:,:,k].flatten()).reshape(ngx, ngy)

        u_phys = np.fft.irfft(u_hat, n=ops.Nz, axis=2)
        dp_dx = massflux_correction(u_phys, y_coords, U_bulk_target, dt)

        rhs_u = u_hat + dt * (alpha[2]*nu*viscous_u[2] + gamma[2]*Nu_curr + zeta[2]*nonlinear_u[1])
        rhs_v = v_hat + dt * (alpha[2]*nu*viscous_v[2] + gamma[2]*Nv_curr + zeta[2]*nonlinear_v[1])
        rhs_w = w_hat + dt * (alpha[2]*nu*viscous_w[2] + gamma[2]*Nw_curr + zeta[2]*nonlinear_w[1])
        rhs_u[:,:,0] += dt * gamma[2] * dp_dx

        u_til, v_til, w_til = predictor_step(rhs_u, rhs_v, rhs_w, A[2], A_solvers[2], wall_idx)
        phi = correction_step(u_til, v_til, w_til, operators, dt, Poisson_solvers, wall_idx)
        u_hat, v_hat, w_hat, p_hat = update_step(u_til, v_til, w_til, p_hat, phi, operators, dt)
        if ngz > 1: u_hat[:,:,-1] = v_hat[:,:,-1] = w_hat[:,:,-1] = 0.0

        # --- Diagnostics ---
        t1 = time.perf_counter()
        
        # Check Divergence
        div_check = np.zeros_like(u_hat)
        for k in range(Nz_fourier):
             div_check[:,:,k] = (operators.Dx @ u_hat[:,:,k]) + (v_hat[:,:,k] @ operators.Dy.T) + 1j*operators.kz[k]*w_hat[:,:,k]
        
        max_div = np.max(np.abs(np.fft.irfft(div_check, n=ops.Nz, axis=2)))
        div_history.append(max_div)
        timesteps.append(n)

        print(f"Step {n+1}: Time {t1-t0:.3f}s, Max Div {max_div:.2e}")
        
        if (n+1) % 10 == 0:
            u_slice = np.fft.irfft(u_hat, n=ops.Nz, axis=2)[:,:,ngz//2]
            plt.figure(figsize=(8,4))
            plt.pcolormesh(grid['X'][:,:,0], grid['Y'][:,:,0], u_slice, shading='gouraud', cmap='viridis')
            plt.colorbar()
            plt.title(f"U Velocity Step {n+1}")
            plt.savefig(f"figures/step_{n+1}.png")
            plt.close()
