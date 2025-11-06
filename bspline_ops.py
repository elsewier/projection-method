import numpy as np 
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
from bspline_module import bspline_basis_physical, bspline_deriv1_physical, bspline_deriv2_physical, generate_knots_and_colloc_pts
from grid import create_channel_grid 
from scipy.io import savemat
from pypardiso import spsolve






class BSplineOperator:
    def __init__(self, grid, p, q, stretch_factor = 0.0, periodic_x = False):
        self.grid   = grid 
        self.p      = p 
        self.q      = q 
        self.stretch_factor = stretch_factor 
        self.periodic_x = periodic_x

        self.Nx = grid.get('Nx', 1)
        self.Ny = grid.get('Ny', 1)
        self.Nz = grid.get('Nz', 1) # physical number of points 
        self.Nz_fourier = grid.get('Nz_fourier') # number of fourier mods 

        self.kz = grid.get('kz') # wavenumbers 

        self.Dx, self.Dxx = None, None
        self.Dy, self.Dyy = None, None

        self.precompute_matrices()

    def precompute_matrices(self):

        xmin, xmax = 0, self.grid['Lx']
        x_colloc = self.grid['x_colloc']
        knots_x = self.grid['x_knots']

        ymin, ymax = -self.grid['H'], self.grid['H']
        y_colloc = self.grid['y_colloc']
        knots_y = self.grid['y_knots']


        B0x = np.zeros((self.Nx, self.Nx))
        B1x = np.zeros((self.Nx, self.Nx))
        B2x = np.zeros((self.Nx, self.Nx))

        B0y = np.zeros((self.Ny, self.Ny))
        B1y = np.zeros((self.Ny, self.Ny))
        B2y = np.zeros((self.Ny, self.Ny))


        if not self.periodic_x:
            for i in range(self.Nx):
                for j in range(self.Nx):
                    B0x[i,j] = bspline_basis_physical(j, self.p, knots_x, x_colloc[i], xmin, xmax)
                    B1x[i,j] = bspline_deriv1_physical(j, self.p, knots_x, x_colloc[i], xmin, xmax)
                    B2x[i,j] = bspline_deriv2_physical(j, self.p, knots_x, x_colloc[i], xmin, xmax)
        else: 
            for i in range(self.Nx):
                xi = x_colloc[i]
                for j in range(self.Nx):
                    b0_val = bspline_basis_physical(j, self.p, knots_x, xi, xmin, xmax) \
                            + bspline_basis_physical(j, self.p, knots_x, xi - self.grid['Lx'], xmin, xmax) \
                            + bspline_basis_physical(j, self.p, knots_x, xi + self.grid['Lx'], xmin, xmax) 

                    b1_val = bspline_deriv1_physical(j, self.p, knots_x, xi, xmin, xmax) \
                            + bspline_deriv1_physical(j, self.p, knots_x, xi - self.grid['Lx'], xmin, xmax) \
                            + bspline_deriv1_physical(j, self.p, knots_x, xi + self.grid['Lx'], xmin, xmax) 

                    b2_val = bspline_deriv2_physical(j, self.p, knots_x, xi, xmin, xmax) \
                            + bspline_deriv2_physical(j, self.p, knots_x, xi - self.grid['Lx'], xmin, xmax) \
                            + bspline_deriv2_physical(j, self.p, knots_x, xi + self.grid['Lx'], xmin, xmax) 
                    
                    B0x[i, j] = b0_val
                    B1x[i, j] = b1_val
                    B2x[i, j] = b2_val

        for i in range(self.Ny):
            for j in range(self.Ny):
                B0y[i,j] = bspline_basis_physical( j, self.q, knots_y, y_colloc[i], ymin, ymax)
                B1y[i,j] = bspline_deriv1_physical(j, self.q, knots_y, y_colloc[i], ymin, ymax)
                B2y[i,j] = bspline_deriv2_physical(j, self.q, knots_y, y_colloc[i], ymin, ymax)

        Dx_T    = np.linalg.solve(B0x.T, B1x.T) # calculating Dx = B1x B0x^-1 without takinv inverse 
        Dxx_T   = np.linalg.solve(B0x.T, B2x.T) # Dxx = B2x B0x^-1 
        self.Dx = Dx_T.T
        self.Dxx= Dxx_T.T

        Dy_T    = np.linalg.solve(B0y.T, B1y.T) 
        Dyy_T   = np.linalg.solve(B0y.T, B2y.T)
        self.Dy = Dy_T.T
        self.Dyy= Dyy_T.T

        # if self.Nz > 1:
        #     Lz = self.grid['Lz']
        #     dz = Lz / self.Nz 
        #     self.kz = (2 * np.pi) * np.fft.fftfreq(self.Nz, d = dz)

    def laplacian_2d(self):
        # build 2D laplacian using 1D operators Dxx, Dyy 
        # Laplacian = (Dxx kron Iy) + (Ix kron Dyy)
        
        Nx = self.Nx 
        Ny = self.Ny 
        Dxx_sparse = sp.csr_matrix(self.Dxx)
        Dyy_sparse = sp.csr_matrix(self.Dyy) 
        # Dxx = self.Dxx
        # Dyy = self.Dyy

        # Laplacian_sp = sp.csr_matrix(Laplacian_org)

        Ix_sparse  = sp.identity(self.Nx, format = 'csr')
        Iy_sparse  = sp.identity(self.Ny, format = 'csr')
        # Ix = sp.identity(self.Nx)
        # Iy = sp.identity(self.Ny)

        # Laplacian_2D = sp.kron(Dxx_sparse, Iy_sparse, format = 'csr') + sp.kron(Ix_sparse, Dyy_sparse, format = 'csr')
        Laplacian_2D = sp.kron(Dxx_sparse, Iy_sparse, format = 'csr') + sp.kron(Ix_sparse, Dyy_sparse, format = 'csr') 
        # Laplacian_2D = sp.kron(Dxx, Iy) + sp.kron(Ix, Dyy)

        # # manually building laplacian csr 
        # Nx = self.Nx 
        # Ny = self.Ny 
        # N = Nx * Ny 
        #
        # Dxx_s = sp.csr_matrix(self.Dxx)
        # Dyy_s = sp.csr_matrix(self.Dyy)
        #
        # nnz = (Dxx_s.nnz * Ny) + (Dyy_s.nnz * Nx)
        # # csr matrices 
        # data = np.zeros(nnz, dtype = np.float64)
        # indices = np.zeros(nnz, dtype = np.int32)
        # indptr = np.zeros(N + 1, dtype = np.int32)
        #
        # pos = 0 # this will hold the current location in the arrays
        #
        # for i in range(Nx):
        #     for j in range(Ny):
        #
        #         # Dxx part 
        #         start = Dxx_s.indptr[i]
        #         end = Dxx_s.indptr[i + 1]
        #         col_xx = Dxx_s.indices[start:end]
        #         data_xx = Dxx_s.data[start:end]
        #
        #         indices[pos : pos + (end - start)] = col_xx * Ny + j 
        #         data[pos : pos + (end - start)] = data_xx 
        #         pos += (end - start)
        #
        #         # Dyy part 
        #         start = Dyy_s.indptr[j]
        #         end = Dyy_s.indptr[j + 1]
        #         col_yy = Dyy_s.indices[start:end]
        #         data_yy= Dyy_s.data[start:end]
        #
        #         indices[pos : pos + (end - start)] = i * Ny + col_yy
        #         data[pos : pos + (end - start)] = data_yy 
        #         pos += (end - start)
        #
        #         indptr[i * Ny + j + 1] = pos # index pointer for next row
        #
        # Laplacian_2D = sp.csr_matrix((data, indices, indptr), shape = (N, N))
        # Laplacian_2D.sum_duplicates()
        # Laplacian_2D.eliminate_zeros()

        return Laplacian_2D



class PoissonSolver:
    def __init__(self, operators):
        self.operators = operators 
        self.Nx = operators.Nx 
        self.Ny = operators.Ny 
        self.laplacian = operators.laplacian_2d()

    def apply_bcs(self, L, rhs_vector, u_outlet):
        Nx = self.Nx 
        Ny = self.Ny 
        X = self.operators.grid['X']
        Y = self.operators.grid['Y']
        L_lil = L.tolil()

        kx = 4 * np.pi / self.operators.grid['Lx']
        ky = 4 * np.pi / (2 * self.operators.grid['H'])

        # bottom j = 0 and top j = Ny - 1 
        for i in range(Nx):
            # bottom 
            idx = i * Ny + 0
            L_lil.rows[idx] = [idx]
            L_lil.data[idx] = [1.0]
            rhs_vector[idx] = np.sin(kx * X[i, 0]) * np.cos(ky * Y[i, 0])

            # top 
            idx = i * Ny + (Ny - 1)
            L_lil.rows[idx] = [idx]
            L_lil.data[idx] = [1.0]
            rhs_vector[idx] = np.sin(kx * X[i, Ny - 1]) * np.cos(ky * Y[i, Ny - 1])

        # # inlet (i = 0) is zero (sinkx * 0) = 0
        # for j in range(1, Ny - 1): 
        #     idx = 0 * Ny + j 
        #     L_lil.rows[idx] = [idx]
        #     L_lil.data[idx] = [1.0]
        #     rhs_vector[idx] = 0.0
        #
        # # outlet
        # Dx_s = sp.csr_matrix(self.operators.Dx)
        # start = Dx_s.indptr[Nx - 1]
        # end = Dx_s.indptr[Nx]
        #
        # cols_x  = Dx_s.indices[start : end]
        # vals_x  = Dx_s.data[start : end]
        #
        # # i think i can mix this with dirichlet one
        # for j in range(1, Ny - 1): # except corners 
        #     L_lil.rows[(Nx - 1) * Ny + j] = list((cols_x * Ny) + j)
        #     L_lil.data[(Nx - 1) * Ny + j] = list(vals_x)
        #     rhs_vector[(Nx - 1) * Ny + j] = kx * np.cos(kx * X[i,j]) * np.cos(ky * Y[i,j])#float(u_outlet[j])

        return L_lil.tocsr(), rhs_vector 

    def solve(self, rhs_field, u_outlet):
        # this will solve u = L^-1 * rhs 
        rhs_vector = rhs_field.flatten(order = 'C')
        Laplacian_bc, rhs_bc = self.apply_bcs(self.laplacian.copy(), rhs_vector.copy(), u_outlet)

        # Laplacian_csc = Laplacian_bc.tocsc() # csc is faster for spsolve
        plt.spy(Laplacian_bc, precision=0, marker='s', markersize=1, aspect='equal', color='k')
        plt.show()

        # cond_number = np.linalg.cond(Laplacian_bc.todense())
        # print(f"\n Condition number: {cond_number:.4e}")

        solution_vector = spsolve(Laplacian_bc,rhs_bc)

        # check condition number
        # solution_vector, info = gmres(Laplacian_bc, rhs_bc, M = M, tol = 1e-12, restart = 300, maxiter = 300)
        solution_field = solution_vector.reshape((self.Nx, self.Ny), order = 'C')

        return solution_field

# imshow nonzero values it should be dominated diagonal component then convert to boolean 
# show the matrix. this is similar to matlab eye function

def plot_result(X, Y, data, title):
    plt.figure(figsize = (12, 5))
    plt.pcolormesh(X, Y, data, shading = 'gouraud', cmap = 'viridis')
    plt.title(title)
    

if __name__ == '__main__': # test poisson solver 
# test function: u(x,y) = sin(pi x / Lx) * (y^2 - H^2)

    p = 7 # order in x direction 
    q = 7 # order in y direction
    grid = create_channel_grid(Nx = 25, Ny = 25, Nz = 1, Lx = 1.0, H = 0.5, Lz = 0.0, p = p, q = p, stretch_factor = 0.0) 
    operators = BSplineOperator(grid, p = p, q = p, periodic_x = True)

    solver = PoissonSolver(operators)

    Lx  = grid['Lx']
    H   = grid['H']
    X   = grid['X']
    Y   = grid['Y']

    kx = 4 * np.pi / Lx 
    ky = 4 * np.pi / (2 * H)

    rhs_field = -(kx**2 + ky**2) * np.sin(kx * X) * np.cos(ky * Y)
    # rhs_field = -(np.pi / Lx)**2 * np.sin(np.pi * X / Lx) * (Y**2 - H**2) + 2 * np.sin(np.pi * X / Lx)

    # u_outlet = -(np.pi / Lx) * (Y[-1, :]**2 - H**2)
    u_outlet = kx * np.cos(ky * Y[-1, :])
    start_time = time.perf_counter()
    solution_numerical = solver.solve(rhs_field, u_outlet = None)
    end_time = time.perf_counter()
    print(solution_numerical.dtype)

    time1 = end_time - start_time
    print(f"Ax=b took {time1:.4f} seconds")
    # solution_exact = np.sin(np.pi * X / Lx) * (Y**2 - H**2)
    solution_exact = np.sin(kx * X) * np.cos(ky * Y)

    error = np.max(np.abs(solution_numerical - solution_exact)) #infinity norm 
    print(f"Infinity Norm of the solution {error}")

    plot_result(X, Y, solution_numerical, "Numerical Solution")
    plot_result(X, Y, solution_exact, "Exact Solution")
    plot_result(X, Y, solution_numerical - solution_exact, "Error")
    # plt.show()




    

