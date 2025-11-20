import numpy as np
import scipy.sparse as sp 
import matplotlib.pyplot as plt
import h5py
from scipy.sparse.linalg import gmres, cg
from pypardiso import spsolve, PyPardisoSolver
import time
import os
import plotly.graph_objects as go 

from operators import build_A, build_P, apply_dirichlet, apply_dirichlet_rhs, apply_neumann, boundary_flag, pin_pressure
from bspline_ops import BSplineOperator 
from grid import create_channel_grid 
from solver import compute_nonlinear_term, predictor_step, correction_step, update_step, massflux_correction 

# Domain sizes 
H   = 1.0 # Minimal channel and H is the half height
Lx  = 1.0 * np.pi * H
Lz  = 0.4 * np.pi * H

# Parameters
# Re_bulk = 3300.0 # Kim & Moin 
Re_bulk = 6000.0 # for triggering turbulence
U_bulk = 1.0
nu = (U_bulk * H) / Re_bulk

# Grid Resolution
ngx = 64
ngy = 65
ngz = 64

# B-spline 
p = 7 
q = 7 


# Time 
dt = 0.001 
num_steps = 2500
save_step = 500

output_dir = "results"

# RK3 Coefficients
alpha   = [29/60, -3/40, 1/6]
beta    = [37/160, 5/24, 1/6]
gamma   = [8/15, 5/12, 3/4]
zeta    = [0.0, -17/60, -5/12]


def initialize_turbulence(grid):
    Nx = grid['Nx']
    Ny = grid['Ny']
    Nz = grid['Nz']
    
    y = grid['y_colloc']
    H = grid['H']
    X, Y, Z = grid['X'], grid['Y'], grid['Z']

    u = np.zeros((Nx, Ny, Nz))
    v = np.zeros((Nx, Ny, Nz))
    w = np.zeros((Nx, Ny, Nz))

    # parabolic velocity profile
    u_vel = 1.5 * 1.0 * (1 - (Y / H)**2) # 1.0 is U_bulk target and U_bulk = (2/3) * U_max
    u += u_vel 

    # add random noise 
    rng = np.random.default_rng(seed = 66) # fix random seed 
    noise_u = (rng.random((Nx, Ny, Nz)) - 0.5) * 2.0 # [-1, 1]
    noise_v = (rng.random((Nx, Ny, Nz)) - 0.5) * 2.0 # [-1, 1]
    noise_w = (rng.random((Nx, Ny, Nz)) - 0.5) * 2.0 # [-1, 1]

    u += 0.2 * noise_u * (1 - (Y / H)**2)
    v += 0.2 * noise_v * (1 - (Y / H)**2)
    w += 0.2 * noise_w * (1 - (Y / H)**2)

    # no-slip for wall 
    u[:, 0, :] = 0.0
    v[:, 0, :] = 0.0
    w[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    v[:, -1, :] = 0.0
    w[:, -1, :] = 0.0

    return u, v, w



def compute_statistics(u, v, w, p, grid, stats, step, t):

    ua = np.mean(u, axis = (0, 2)) # shape of (Ny, )
    va = np.mean(v, axis = (0, 2))
    wa = np.mean(w, axis = (0, 2))
    pa = np.mean(p, axis = (0, 2))

    u_prime = u - ua[None, :, None]
    v_prime = v - va[None, :, None]
    w_prime = w - wa[None, :, None]
    p_prime = p - pa[None, :, None]

    stats['ua'] += ua
    stats['va'] += va
    stats['wa'] += wa
    stats['pa'] += pa
    stats['uu'] += np.mean(u_prime**2, axis = (0, 2))
    stats['vv'] += np.mean(v_prime**2, axis = (0, 2))
    stats['ww'] += np.mean(w_prime**2, axis = (0, 2))
    stats['uv'] += np.mean(w_prime * v_prime, axis = (0, 2))
    stats['pp'] += np.mean(p_prime**2, axis = (0, 2))

    stats['count'] += 1


    # utau calculation 
    y = grid['y_colloc']
    # bottom wall 
    dy_bot = y[1] - y[0]
    du_bot = ua[1] - ua[0]
    dudy_bot = du_bot / dy_bot 

    tau_w_bot = nu * np.abs(dudy_bot) 

    dy_top = y[-1] - y[-2]
    du_top = ua[-1] - ua[-2]
    dudy_top = du_top / dy_top 

    tau_w_top  = nu * np.abs(dudy_top) 

    tau_w = 0.5 * (tau_w_top + tau_w_bot)

    u_tau = np.sqrt(tau_w)
    delta = H 
    Re_tau = (u_tau * delta) / nu 
    Cf = 2.0 * (u_tau / 1.0)**2 # 1.0 is Ubulk 

    stats['u_tau'] += u_tau 
    stats['Re_tau'] += Re_tau

    stats['time_history'].append(t)
    stats['u_tau_history'].append(u_tau)
    stats['Re_tau_history'].append(Re_tau)


    print(f"u_tau= {u_tau:.4f}, Re_tau={Re_tau:.4f}")


    return stats 


def save_statistics(stats, grid, step, output_dir):
    N = stats['count']
    if N == 0 : N = 1.0 
    
    filename = os.path.join(output_dir, f"stat_{step:05d}.h5")

    with h5py.File(filename, 'w') as f:
        f.attrs['timestep'] = step 
        f.attrs['samples'] = N 

        profiles_ = f.create_group("profiles")
        profiles_.create_dataset("ua", data = stats['ua'] / N)
        profiles_.create_dataset("va", data = stats['va'] / N)
        profiles_.create_dataset("wa", data = stats['wa'] / N)

        profiles_.create_dataset("uu", data = stats['uu'] / N)
        profiles_.create_dataset("vv", data = stats['vv'] / N)
        profiles_.create_dataset("ww", data = stats['ww'] / N)
        profiles_.create_dataset("uv", data = stats['uv'] / N)
        profiles_.create_dataset("pp", data = stats['pp'] / N)

        history_ = f.create_group("history")
        history_.create_dataset("time", data = np.array(stats['time_history']))
        history_.create_dataset("u_tau", data = np.array(stats['u_tau_history']))
        history_.create_dataset("Re_tau", data = np.array(stats['Re_tau_history']))

        grid_ = f.create_group("grid")
        grid_.create_dataset("y", data = grid['y_colloc'])
        grid_.create_dataset("x", data = grid['x_colloc'])
        grid_.create_dataset("z", data = grid['z_colloc'])

    

def save_field(filename, u, v, w, p, grid, step, t):
    with h5py.File(filename, 'w') as f:
        f.attrs['time'] = t
        f.attrs['timestep'] = step 

        f.create_dataset("u", data = u)
        f.create_dataset("v", data = v)
        f.create_dataset("w", data = w)
        f.create_dataset("p", data = p)

        grid_ = f.create_group("grid")
        grid_.create_dataset("y", data = grid['y_colloc'])
        grid_.create_dataset("x", data = grid['x_colloc'])
        grid_.create_dataset("z", data = grid['z_colloc'])



if __name__ == '__main__':


    grid = create_channel_grid(Nx = ngx, Ny = ngy, Nz = ngz, Lx = Lx, H = H, Lz = Lz, p = p, q = q, stretch_factor = 2.0, periodic_x=True)
    operators = BSplineOperator(grid, p = p, q = q, periodic_x = True)

    Nz_fourier = grid['Nz_fourier']


    print("Factorizing matrices...")
    _, _, bottom_idx, top_idx = boundary_flag(ngx, ngy)
    wall_idx = np.concatenate([bottom_idx, top_idx])

    # these matrices are not changing 
    Laplacian = [] #[k], this is for viscous term calculation 
    A = [[], [], []] # [substep][k], bc applied for solving 
    Poisson = [] # bc applied for solving
    A_solvers = [[], [], []] # [substep][k]
    Poisson_solvers = [] #[k]

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

    stats = {
            'ua' : np.zeros(ngy),
            'va' : np.zeros(ngy),
            'wa' : np.zeros(ngy),
            'pa' : np.zeros(ngy),
            'uu' : np.zeros(ngy),
            'vv' : np.zeros(ngy),
            'ww' : np.zeros(ngy),
            'uv' : np.zeros(ngy),
            'pp' : np.zeros(ngy),
            'count': 0,
            'u_tau': 0.0,
            'Re_tau': 0.0,
            'time_history': [],
            'u_tau_history': [],
            'Re_tau_history': []
    }

    # initialize turbulence field 
    u_n, v_n, w_n = initialize_turbulence(grid)
    p_n = np.zeros_like(u_n)
    print("Turbulence initialized. Initial divergence correction being applied")
    u_hat = np.fft.rfft(u_n, axis = 2)
    v_hat = np.fft.rfft(v_n, axis = 2)
    w_hat = np.fft.rfft(w_n, axis = 2)
    p_hat = np.fft.rfft(p_n, axis = 2)

    # solve poisson
    phi_hat = correction_step(u_hat, v_hat, w_hat, operators, 1.0, Poisson, Poisson_solvers, wall_idx)
    u_hat, v_hat, w_hat, p_hat = update_step(u_hat, v_hat, w_hat, p_hat, phi_hat, operators, 1.0)

    u_hat[:, :, -1] = 0.0
    v_hat[:, :, -1] = 0.0
    w_hat[:, :, -1] = 0.0

    u_n = np.fft.irfft(u_hat, n = ngz)
    v_n = np.fft.irfft(v_hat, n = ngz)
    w_n = np.fft.irfft(w_hat, n = ngz)



    # pre-allocation 
    viscous_u = [np.zeros((ngx, ngy, Nz_fourier), dtype = complex) for _ in range(3)] # this will create 3 arrays for each step. with this we can write the substeps inside a loop 
    viscous_v = [np.zeros((ngx, ngy, Nz_fourier), dtype = complex) for _ in range(3)] 
    viscous_w = [np.zeros((ngx, ngy, Nz_fourier), dtype = complex) for _ in range(3)]


    print(f"Starting Time-stepping...")

    y_coords = grid['Y'][0, :, 0]
    p_old_hat = p_hat.copy()

    for n in range(num_steps):

        start_time = time.perf_counter()
        print(f"Time-step {n + 1} / {num_steps}")
        dp_dx = massflux_correction(u_n, y_coords, U_bulk, dt) 

        u_hat = np.fft.rfft(u_n, axis = 2)
        v_hat = np.fft.rfft(v_n, axis = 2)
        w_hat = np.fft.rfft(w_n, axis = 2)
        p_hat = np.fft.rfft(p_n, axis = 2)

        Nu, Nv, Nw = compute_nonlinear_term(u_n, v_n, w_n, operators)
        if n == 0:
            Nu_old, Nv_old, Nw_old = Nu.copy(), Nv.copy(), Nw.copy()

        # RK3 substeps 
        for i in range(3):
            # calculate viscous term 
            for k in range(Nz_fourier):
                viscous_u[i][:, :, k] = (Laplacian[k] @ u_hat[:, :, k].flatten()).reshape(ngx, ngy)
                viscous_v[i][:, :, k] = (Laplacian[k] @ v_hat[:, :, k].flatten()).reshape(ngx, ngy)
                viscous_w[i][:, :, k] = (Laplacian[k] @ w_hat[:, :, k].flatten()).reshape(ngx, ngy)

            # RHS 
            rhs_u = u_hat + dt * (alpha[i] * nu * viscous_u[i] + gamma[i] * Nu + zeta[i] * Nu_old)
            rhs_v = v_hat + dt * (alpha[i] * nu * viscous_v[i] + gamma[i] * Nv + zeta[i] * Nv_old)
            rhs_w = w_hat + dt * (alpha[i] * nu * viscous_w[i] + gamma[i] * Nw + zeta[i] * Nw_old)

            rhs_u[:, :, 0] += dt * gamma[i] * (dp_dx * ngz)

            u_tilde_hat, v_tilde_hat, w_tilde_hat = predictor_step(rhs_u, rhs_v, rhs_w, A[i], A_solvers[i], wall_idx)
            phi_hat = correction_step(u_tilde_hat, v_tilde_hat, w_tilde_hat, operators, dt, Poisson, Poisson_solvers, wall_idx)
            u_hat, v_hat, w_hat, p_hat = update_step(u_tilde_hat, v_tilde_hat, w_tilde_hat, p_old_hat, phi_hat, operators, dt)

            if ngz > 1:
                u_hat[:, :, -1] = 0.0
                v_hat[:, :, -1] = 0.0
                w_hat[:, :, -1] = 0.0
                p_hat[:, :, -1] = 0.0


            if i < 2: # except last step 
                u_physical = np.fft.irfft(u_hat, n = ngz)
                v_physical = np.fft.irfft(v_hat, n = ngz)
                w_physical = np.fft.irfft(w_hat, n = ngz)
                Nu_old, Nv_old, Nw_old = Nu, Nv, Nw 
                Nu, Nv, Nw = compute_nonlinear_term(u_physical, v_physical, w_physical, operators)

        # final update 
        u_n = np.fft.irfft(u_hat, n = ngz)
        v_n = np.fft.irfft(v_hat, n = ngz)
        w_n = np.fft.irfft(w_hat, n = ngz)
        p_n = np.fft.irfft(p_hat, n = ngz)
        p_old_hat = p_hat.copy()
        Nu_old, Nv_old, Nw_old = Nu, Nv, Nw 

        stats = compute_statistics(u_n, v_n, w_n, p_n, grid, stats, n, n * dt)

        if n % save_step == 0 and n > 0:
            fname = os.path.join(output_dir, f"field_{n:05d}.npz")
            save_field(fname, u_n, v_n, w_n, p_n, grid, n, n * dt)
            save_statistics(stats, grid, n, output_dir)

        end_time = time.perf_counter()
        print(f"Timestep took :{end_time - start_time} seconds")








