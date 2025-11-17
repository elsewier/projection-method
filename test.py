import numpy as np 
from pypardiso import PyPardisoSolver

from operators import build_A, build_P, apply_dirichlet, apply_neumann, boundary_flag, pin_pressure
from bspline_ops import BSplineOperator 
from grid import create_channel_grid 
from solver import predictor_step, correction_step, update_step, compute_nonlinear_term


def compute_divergence(u, v, w, ops, grid):

    ngx, ngy, ngz = u.shape 

    du_dx   = (ops.Dx @ u.reshape(ngx, -1)).reshape(u.shape)
    dv_dy   = (v.transpose(0, 2, 1) @ ops.Dy.T).transpose(0, 2, 1)
    w_hat = np.fft.rfft(w, axis = 2)
    dw_dz   = np.fft.irfft(1j * grid['kz_3d'] * w_hat, n = ngz, axis = 2)

    div = du_dx + dv_dy + dw_dz 

    return div 


# Test1: div-free projection 

def test_divergence_free_projection(u_n, v_n, w_n, rhs_u, rhs_v, rhs_w,ops, grid, solvers):

    ngx, ngy, ngz = grid['Nx'], grid['Ny'], grid['Nz']
    X, Y, Z = grid['X'], grid['Y'], grid['Z']
    Lx, Ly, Lz = grid['Lx'], 2 * grid['H'], grid['Lz']

    div1 = compute_divergence(u_n, v_n, w_n, ops, grid)
    div1_norm = np.linalg.norm(div1)
    print(f"Initial divergence norm: {div1_norm:.4e}")

    u_hat = np.fft.rfft(u_n, axis = 2)
    v_hat = np.fft.rfft(v_n, axis = 2)
    w_hat = np.fft.rfft(w_n, axis = 2)
    p_hat = np.fft.rfft(p_n, axis = 2)


    # Prediction 
    u_tilde, v_tilde, w_tilde = predictor_step(rhs_u, rhs_v, rhs_w, ops, solvers['nu'], solvers['dt'], solvers['A_solvers'], solvers['A_matrices'], solvers['wall_idx'], dp_dx = 0.0)
    # Correction 
    phi_hat = correction_step(u_tilde, v_tilde, w_tilde, ops, solvers['dt'], solvers['Poisson_solvers'], solvers['Poisson_matrices'], solvers['wall_idx'])
    # Update 
    u1_hat, v1_hat, w1_hat, p1_hat = update_step(u_tilde, v_tilde, w_tilde, p_hat, phi_hat, ops, solvers['dt'])

    u_final = np.fft.irfft(u1_hat, n = ngz)
    v_final = np.fft.irfft(v1_hat, n = ngz)
    w_final = np.fft.irfft(w1_hat, n = ngz)

    # final divergence 
    div2 = compute_divergence(u_final, v_final, w_final, ops, grid)
    div2_max = np.max(np.abs(div2))
    print(f"Final max abs div: {div2_max:.4e}")
    if(div2_max < 1e-13) print("Divergence-free projection test PASSED")



if __name__ == '__main__':
    p, q = 5, 5 
    ngx, ngy, ngz = 128, 64, 4 

    grid = create_channel_grid(Nx = ngx, Ny = ngy, Nz = ngz, Lx = 2 * np.pi, H = 0.5, Lz = np.pi, p = p, q = q, stretch_factor = 1.0)
    ops = BSplineOperator(grid, p = p, q = q, periodic_x = True)

    Re = 180 
    dt = 0.005 
    nu = 1.0 / Re 
    Nz_fourier = grid['Nz_fourier']

    u_vel = 1.5 * 1.0 * (1 - (grid['Y']/grid['H'])**2)
    
    u_n = u_vel 
    v_n = np.zeros((ngx, ngy, ngz))
    w_n = np.zeros((ngx, ngy, ngz))
    p_n = np.zeros((ngx, ngy, ngz))
    

    alpha   = [29/60, -3/40, 1/6]
    beta    = [37/160, 5/24, 1/6]
    gamma   = [8/15, 5/12, 3/4]
    zeta    = [0.0, -17/60, -5/12]

    left, right, bottom, top = boundary_flag(ngx, ngy)
    wall_idx = np.concatenate([bottom, top])

    A_matrices, A_solvers = [], []
    Poisson_matrices, Poisson_solvers = [], []
    Laplacian = []

    for k in range(Nz_fourier):
        A_k = build_A(ops, nu, dt, beta[0], ops.kz[k])
        A_bc, _ = apply_dirichlet(A_k.copy(), np.zeros(ngx * ngy), wall_idx, val = 0.0)
        A_matrices.append(A_bc)

        A_solver = PyPardisoSolver()
        A_solver.factorize(A_bc.tocsr())
        A_solvers.append(A_solver)

        P_k = build_P(ops, ops.kz[k])
        Laplacian.append(P_k.tocsr())
        P_bc, _ = apply_neumann(P_k.copy(), np.zeros(ngx * ngy), wall_idx, ops)
        if k == 0:
            P_bc, _ = pin_pressure(P_bc, np.zeros(ngx * ngy), 0)
        Poisson_matrices.append(P_bc.tocsr())

        P_solver = PyPardisoSolver()
        P_solver.factorize(P_bc.tocsr())
        Poisson_solvers.append(P_solver)


    
    # pre-allocation 
    viscous_u_old   = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_v_old   = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)
    viscous_w_old   = np.zeros((ngx, ngy, Nz_fourier), dtype = np.complex128)

    Nu, Nv, Nw = compute_nonlinear_term(u_n, v_n, w_n, ops)

    
    u_old_hat = np.fft.rfft(u_n, axis = 2)
    v_old_hat = np.fft.rfft(v_n, axis = 2)
    w_old_hat = np.fft.rfft(w_n, axis = 2)
    p_old_hat = np.fft.rfft(p_n, axis = 2)

    for k in range(Nz_fourier):
        viscous_u_old[:, :, k] = (Laplacian[k] @ u_old_hat[:, :, k].flatten()).reshape(ngx, ngy)
        viscous_v_old[:, :, k] = (Laplacian[k] @ v_old_hat[:, :, k].flatten()).reshape(ngx, ngy)
        viscous_w_old[:, :, k] = (Laplacian[k] @ w_old_hat[:, :, k].flatten()).reshape(ngx, ngy)



    rhs_u1_hat = u_old_hat + dt * (alpha[0] * nu * viscous_u_old + gamma[0] * Nu)
    rhs_v1_hat = v_old_hat + dt * (alpha[0] * nu * viscous_v_old + gamma[0] * Nv)
    rhs_w1_hat = w_old_hat + dt * (alpha[0] * nu * viscous_w_old + gamma[0] * Nw)





    solvers = {'A_matrices': A_matrices, 'A_solvers': A_solvers, 'Poisson_matrices': Poisson_matrices, 'Poisson_solvers': Poisson_solvers, 'wall_idx': wall_idx, 'nu': nu, 'dt': dt}

    test_divergence_free_projection(u_n, v_n, w_n, rhs_u1_hat, rhs_v1_hat, rhs_w1_hat, ops, grid, solvers)


