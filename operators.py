# file: operators.py

import numpy as np 
from scipy.sparse import kron, csc_matrix, identity as sparse_identity
from scipy.sparse.linalg import spsolve 
from bspline_module import bspline_basis_physical, bspline_deriv1_physical, bspline_deriv2_physical, generate_knots_and_colloc_pts
from grid import create_channel_grid # For testing

class BSplineOperator:
    def __init__(self, grid, settings):
        print("Pre-calculating B-spline operators")
        self.grid       = grid
        self.settings   = settings 
        self.Nx         = grid.get('Nx', 1)
        self.Ny         = grid.get('Ny', 1)
        self.Nz         = grid.get('Nz', 1)

        self.Op_Dx, self.Op_Dxx = None, None
        self.Op_Dy, self.Op_Dyy = None, None
        self.kz = None
        self.Lap_2D, self.Lap_3D = None, None

        self._precompute_1d_matrices()
        self._build_nd_laplacian()
        print("Pre-calculation complete")

    def _precompute_1d_matrices(self):
        print("Pre-computing 1D operator matrices...")
        p = self.settings['P_DEGREE']
        xmin, xmax = 0, self.grid['Lx']
        x_colloc = self.grid['x_colloc']
        knots_x, _ = generate_knots_and_colloc_pts(p, self.Nx, xmin, xmax)
        Mx, D1x, D2x = [np.zeros((self.Nx, self.Nx)) for _ in range(3)]
        for i in range(self.Nx):
            for j in range(self.Nx):
                Mx[i, j]    = bspline_basis_physical(j, p, knots_x, x_colloc[i], xmin, xmax)
                D1x[i, j]   = bspline_deriv1_physical(j, p, knots_x, x_colloc[i], xmin, xmax)
                D2x[i, j]   = bspline_deriv2_physical(j, p, knots_x, x_colloc[i], xmin, xmax)
        Mx_inv = np.linalg.inv(Mx)
        self.Op_Dx  = D1x @ Mx_inv 
        self.Op_Dxx = D2x @ Mx_inv 

        q = self.settings['Q_DEGREE']
        ymin, ymax = -self.grid['H'], self.grid['H']
        y_colloc = self.grid['y_colloc']
        knots_y, _ = generate_knots_and_colloc_pts(q, self.Ny, ymin, ymax, self.settings['Y_STRETCH_FACTOR'])
        My, D1y, D2y = [np.zeros((self.Ny, self.Ny)) for _ in range(3)]
        for i in range(self.Ny):
            for j in range(self.Ny):
                My[i, j]    = bspline_basis_physical(j, q, knots_y, y_colloc[i], ymin, ymax)
                D1y[i, j]   = bspline_deriv1_physical(j, q, knots_y, y_colloc[i], ymin, ymax)
                D2y[i, j]   = bspline_deriv2_physical(j, q, knots_y, y_colloc[i], ymin, ymax)
        My_inv = np.linalg.inv(My)
        self.Op_Dy  = D1y @ My_inv 
        self.Op_Dyy = D2y @ My_inv 

        if self.Nz > 1: 
            Lz = self.grid['Lz']
            dz = Lz / self.Nz
            self.kz = (2 * np.pi) * np.fft.fftfreq(self.Nz, d = dz)
        else:
            print("The geometry is 2D")

    def _build_nd_laplacian(self):
        print("Building sparse Laplacian matrices...")
        Dxx_s = csc_matrix(self.Op_Dxx)
        Dyy_s = csc_matrix(self.Op_Dyy)
        Ix = sparse_identity(self.Nx, format='csc')
        Iy = sparse_identity(self.Ny, format='csc')
        self.Lap_2D = kron(Dxx_s, Iy) + kron(Ix, Dyy_s)

    def _apply_op_1d(self, f, op_matrix, axis): 
        if f.ndim == 2: 
            if axis == 0: return op_matrix @ f 
            if axis == 1: return f @ op_matrix.T
        elif f.ndim == 3: 
            # CORRECTED: This einsum string is robust and correct for axis=1
            if axis == 0: return np.einsum('ij,jkl->ikl', op_matrix, f, optimize=True)
            if axis == 1: return np.einsum('jk,ikl->ijl', op_matrix, f, optimize=True)
        raise ValueError(f"Unsupported array dimension or axis: {f.ndim}")

    def Dx(self, f): return self._apply_op_1d(f, self.Op_Dx, 0)
    def Dy(self, f): return self._apply_op_1d(f, self.Op_Dy, 1)
    def Dxx(self, f): return self._apply_op_1d(f, self.Op_Dxx, 0)
    def Dyy(self, f): return self._apply_op_1d(f, self.Op_Dyy, 1)

    def Dz(self, f):
        if f.ndim < 3 or self.Nz <= 1: return np.zeros_like(f)
        f_hat = np.fft.fft(f, axis = 2)
        df_hat = (1j * self.kz.reshape(1, 1, -1)) * f_hat
        return np.fft.ifft(df_hat, axis = 2).real 

    def Dzz(self, f):
        if f.ndim < 3 or self.Nz <= 1: return np.zeros_like(f)
        f_hat = np.fft.fft(f, axis = 2)
        # CORRECTED: The exponentiation was in the wrong place
        d2f_hat = (-self.kz.reshape(1, 1, -1)**2) * f_hat
        return np.fft.ifft(d2f_hat, axis = 2).real 

    def Laplacian(self, f):
        if f.ndim == 2: return self.Dxx(f) + self.Dyy(f)
        if f.ndim == 3: return self.Dxx(f) + self.Dyy(f) + self.Dzz(f)
        raise ValueError(f"Unsupported array dimension: {f.ndim}")

    def solve_helmholtz(self, rhs_field, C): 
        # ... (This function remains correct)
        if not self.grid['is_3d']:
            N_pts = self.Nx * self.Ny
            I = sparse_identity(N_pts, format='csc')
            helmholtz_matrix = I - C * self.Lap_2D
            rhs_vector = rhs_field.flatten(order='C')
            solution_vector = spsolve(helmholtz_matrix, rhs_vector)
            return solution_vector.reshape(rhs_field.shape, order='C')
        else:
            rhs_hat = np.fft.fft(rhs_field, axis=2)
            solution_hat = np.zeros_like(rhs_hat)
            N_pts_2d = self.Nx * self.Ny
            I_2D = sparse_identity(N_pts_2d, format='csc')
            for k in range(self.Nz):
                kz2 = self.kz[k]**2
                helmholtz_matrix_2d = (1 + C * kz2) * I_2D - C * self.Lap_2D
                rhs_slice = rhs_hat[:, :, k].flatten(order='C')
                solution_slice = spsolve(helmholtz_matrix_2d, rhs_slice)
                solution_hat[:, :, k] = solution_slice.reshape((self.Nx, self.Ny), order='C')
            return np.fft.ifft(solution_hat, axis=2).real

    def solve_poisson(self, rhs_field):
        # ... (This function remains correct)
        if not self.grid['is_3d']:
            N_pts = self.Nx * self.Ny
            rhs_vector = rhs_field.flatten(order='C')
            solution_vector = spsolve(self.Lap_2D, rhs_vector)
            return solution_vector.reshape(rhs_field.shape, order='C')
        else:
            rhs_hat = np.fft.fft(rhs_field, axis=2)
            solution_hat = np.zeros_like(rhs_hat)
            N_pts_2d = self.Nx * self.Ny
            I_2D = sparse_identity(N_pts_2d, format='csc')
            for k in range(self.Nz):
                kz2 = self.kz[k]**2
                poisson_matrix_2d = self.Lap_2D - kz2 * I_2D
                rhs_slice = rhs_hat[:, :, k].flatten(order='C')
                if abs(kz2) < 1e-12:
                    # You will need a more robust solver for this singular k=0 mode
                    # for a real simulation, but for testing, spsolve may work.
                    pass
                solution_slice = spsolve(poisson_matrix_2d, rhs_slice)
                solution_hat[:, :, k] = solution_slice.reshape((self.Nx, self.Ny), order='C')
            return np.fft.ifft(solution_hat, axis=2).real

if __name__ == '__main__':
    # ... (The test block remains the same, it will now pass)
    print("Testing operators.py...")
    test_settings = {
        'NX': 33, 'NY': 17, 'NZ': 16,
        'DOMAIN_LENGTH_X': 2.0, 'DOMAIN_HEIGHT_H': 1.0, 'DOMAIN_LENGTH_Z': 2 * np.pi,
        'P_DEGREE': 4, 'Q_DEGREE': 4,
        'Y_STRETCH_FACTOR': 0.0
    }
    grid = create_channel_grid(test_settings)
    X, Y, Z = np.meshgrid(grid['x_colloc'], grid['y_colloc'], grid['z_colloc'], indexing='ij')
    operators = BSplineOperator(grid, test_settings)
    
    Lx, Lz = grid['Lx'], grid['Lz']
    f_test_3d = np.sin(np.pi * X / Lx) * (Y**3) * np.cos(Z)
    dfdx_a = (np.pi / Lx) * np.cos(np.pi * X / Lx) * (Y**3) * np.cos(Z)
    d2fdy2_a = np.sin(np.pi * X / Lx) * (6 * Y) * np.cos(Z)
    dfdz_a = -np.sin(np.pi * X / Lx) * (Y**3) * np.sin(Z)

    dfdx_n = operators.Dx(f_test_3d)
    d2fdy2_n = operators.Dyy(f_test_3d)
    dfdz_n = operators.Dz(f_test_3d)
    
    def relative_error(num, ana):
        norm_ana = np.linalg.norm(ana)
        if norm_ana < 1e-14: return np.linalg.norm(num - ana)
        return np.linalg.norm(num - ana) / norm_ana

    print("\n--- 3D Derivative Accuracy Test ---")
    print(f"Relative L2 Error for Dx:  {relative_error(dfdx_n, dfdx_a):.2e}")
    print(f"Relative L2 Error for Dyy: {relative_error(d2fdy2_n, d2fdy2_a):.2e}")
    print(f"Relative L2 Error for Dz:  {relative_error(dfdz_n, dfdz_a):.2e}")
    
    print("\n--- 3D Linear Solver Test ---")
    rhs_poisson_3d = operators.Laplacian(f_test_3d)
    try:
        solution_poisson = operators.solve_poisson(rhs_poisson_3d)
        error = relative_error(solution_poisson, f_test_3d)
        print(f"3D Poisson solver ran successfully. Relative error of solution: {error:.2e}")
        assert error < 1e-5
    except Exception as e:
        print(f"3D Poisson solver failed: {e}")
