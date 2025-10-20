import numpy as np 
from bspline_module import bspline_basis_physical, bspline_deriv1_physical, bspline_deriv2_physical, generate_knots_and_colloc_pts
from grid import create_channel_grid 





class BSplineOperator:
    def __init__(self, grid, p, q, stretch_factor = 0.0):
        self.grid   = grid 
        self.p      = p 
        self.q      = q 
        self.stretch_factor = stretch_factor 

        self.Nx = grid.get('Nx', 1)
        self.Ny = grid.get('Ny', 1)
        self.Nz = grid.get('Nz', 1)

        self.Dx, self.Dxx = None, None
        self.Dy, self.Dyy = None, None
        self.kz = None 

        self._precompute_matrices()

def _precompute_matrices(self):
    print("Pre-computing necessary matrices to build operators")

    xmin, xmax  = 0, self.grid['Lx']
    x_colloc    = self.grid['x_colloc']
    knots_x, _  = generate_knots_and_colloc_pts(self.p, self.Nx, xmin, xmax)

    ymin, ymax  = 0, self.grid['Ly']
    y_colloc    = self.grid['y_colloc']
    knots_y, _  = generate_knots_and_colloc_pts(self.q, self.Ny, ymin, ymax)

    B0x = np.zeros((self.Nx, self.Nx))
    B1x = np.zeros((self.Nx, self.Nx))
    B2x = np.zeros((self.Nx, self.Nx))

    B0y = np.zeros((self.Ny, self.Ny))
    B1y = np.zeros((self.Ny, self.Ny))
    B2y = np.zeros((self.Ny, self.Ny))


    for i in range(self.Nx):
        for j in range(self.Nx):
            B0x[i,j] = bspline_basis_physical( j, p, knots_x, x_colloc[i], xmin, xmax)
            B1x[i,j] = bspline_deriv1_physical(j, p, knots_x, x_colloc[i], xmin, xmax)
            B2x[i,j] = bspline_deriv2_physical(j, p, knots_x, x_colloc[i], xmin, xmax)

    for i in range(self.Ny):
        for j in range(self.Ny):
            B0y[i,j] = bspline_basis_physical( j, q, knots_y, y_colloc[i], ymin, ymax)
            B1y[i,j] = bspline_deriv1_physical(j, q, knots_y, y_colloc[i], ymin, ymax)
            B2y[i,j] = bspline_deriv2_physical(j, q, knots_y, y_colloc[i], ymin, ymax)

    Dx_T    = np.linalg.solve(B0x.T, B1x.T) # calculating Dx = B1x B0x^-1 without takinv inverse 
    Dxx_T   = np.linalg.solve(B0x.T, B2x.T) # Dxx = B2x B0x^-1 
    self.Dx = Dx_T.T
    self.Dxx= Dxx_T.T

    Dy_T    = np.linalg.solve(B0y.T, B1y.T) 
    Dyy_T   = np.linalg.solve(B0y.T, B2y.T)
    self.Dy = Dy_T.T
    self.Dyy= Dyy_T.T

    if self.Nz > 1:
        Lz = self.grid['Lz']
        dz = Lz / self.Nz 
        self.kz = (2 * np.pi) * np.fft.fftfreq(self.Nz, d = dz)
    else:
        print("The geometry is 2D")



