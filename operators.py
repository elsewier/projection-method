# this will build matrices for per fourier mode 
import numpy as np 
import scipy.sparse as sp 
from scipy.sparse.linalg import gmres
from pypardiso import spsolve

def build_A(operators, nu, dt, beta, kz):
    # A0 = (I - \Deltat \beta * nu * (Dxx kron Dyy))
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
# Poisson pressure correction : (Dxx kron Dyy)
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

    
# find boundary indices and apply bcs 
def boundary_flag(Nx, Ny):
    # corners are included 
    left_idx = []
    right_idx = []
    bottom_idx = []
    top_idx = []

    for i in range(Nx):
        bottom_idx.append(i * Ny + 0)
        top_idx.append(i * Ny + (Ny - 1))
    for j in range(Ny):
        left_idx.append(i * Ny + 0)
        right_idx.append(i * Ny + (Ny - 1))

    left_idx = np.array(left_idx, dtype = np.int32)
    right_idx = np.array(right_idx, dtype = np.int32)
    bottom_idx = np.array(bottom_idx, dtype = np.int32)
    top_idx = np.array(top_idx, dtype = np.int32)

    
    return left_idx, right_idx, bottom_idx, top_idx

def apply_dirichlet(A, b, idx, val = 0.0):
    A_lil = A.tolil()

    is_scalar = np.isscalar(val)
    for i in range(idx.shape[0]):
        k = int(idx[i])
        A_lil.rows[k] = [k]
        A_lil.data[k] = [1.0]
        b[k] = val[i] if not is_scalar else val

    A_csr = A_lil.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()

    return A_csr, b 

def apply_dirichlet_rhs(b, idx, val = 0.0):
    # only modifies the rhs vector 
    
    for i in range(idx.shape[0]):
        k = int(idx[i])
        if np.isscalar(val) == 1:
            b[k] = val
        else:
            b[k] = val[i]

    return b 

def apply_neumann(A, b, boundary_indices, ops, rhs_vals = None):
    # du/dn = 0 
    Nx = ops.Nx
    Ny = ops.Ny

    Dx_s = sp.csr_matrix(ops.Dx)
    Dy_s = sp.csr_matrix(ops.Dy)

    A_lil = A.tolil()

    if rhs_vals is None:
        rhs_vals = np.zeros(len(boundary_indices))

    for i, idx in enumerate(boundary_indices): 
        row = idx // Ny 
        col = idx % Ny 

        # bottom or top wall, normal direction is y 
        if col == 0 or col == Ny - 1: 
            start   = Dy_s.indptr[col]
            end     = Dy_s.indptr[col + 1]
            cols_y  = Dy_s.indices[start:end]
            vals_y  = Dy_s.data[start:end]

            A_lil.rows[idx] = list(row * Ny + cols_y)
            A_lil.data[idx] = list(vals_y)

        # left or right, normal direction is x 
        elif row == 0 or row == Nx - 1:
            start   = Dx_s.indptr[row]
            end     = Dx_s.indptr[row + 1]
            cols_x  = Dx_s.indices[start:end]
            vals_x  = Dx_s.data[start:end]

            A_lil.rows[idx] = list(cols_x * Ny + j)
            A_lil.data[idx] = list(vals_x)

        b[idx] = 0.0


    A_csr = A_lil.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()

    return A_csr, b 

# pressure nullspace pin one dof 
def pin_pressure(A, b, idx):
    A_lil = A.tolil()
    A_lil.rows[idx] = [idx]
    A_lil.data[idx] = [1.0]
    b[idx] = 0.0 

    A_csr = A_lil.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()

    return A_csr, b 




    

