import numpy as np
from scipy.sparse.linalg import gmres
from ..field import Field

def diffusion_matrix_1d(xsize, centervalue, neighborvalue):
    """
    Creates a matrix for the solution of 1d implicit or crank nickolson discretizations
    Because the exact format changes between implicit and C-N, and this method is reused 
        in 2D and 3D cases, centervalue and neighbor value must be explicitly specified
    Matrix shows periodic boundary conditions!
    """
    matrix1d = np.zeros([xsize, xsize])
    np.fill_diagonal(matrix1d, centervalue)
    matrix1d = np.roll(matrix1d, 1, 0)
    np.fill_diagonal(matrix1d, neighborvalue)
    matrix1d = np.roll(matrix1d, -2, 0)
    np.fill_diagonal(matrix1d, neighborvalue)
    matrix1d = np.roll(matrix1d, 1, 0)
    return matrix1d

def diffusion_matrix_2d(ysize, xsize, centervalue, neighborvalue):
    """
    Creates a matrix for the solution of 2d implicit or crank nickolson discretizations
    Because the exact format changes between implicit and C-N, and this method is reused 
        in 3D cases, centervalue and neighbor value must be explicitly specified
    Parameter order is specified as ysize then xsize, because the dimensional order of 2d arrays is [y, x]
    Matrix shows periodic boundary conditions!
    """
    matrix2d = np.zeros([xsize*ysize, xsize*ysize])
    matrix1d = diffusion_matrix_1d(xsize, centervalue, neighborvalue)
    for i in range(ysize):
        matrix2d[xsize*i:xsize*(i+1), xsize*i:xsize*(i+1)] = matrix1d
    matrix2d = np.roll(matrix2d, xsize, 0)
    np.fill_diagonal(matrix2d, neighborvalue)
    matrix2d = np.roll(matrix2d, -2*xsize, 0)
    np.fill_diagonal(matrix2d, neighborvalue)
    matrix2d = np.roll(matrix2d, xsize, 0)
    return matrix2d

def diffusion_matrix_3d(zsize, ysize, xsize, centervalue, neighborvalue):
    """
    Creates a matrix for the solution of 3d implicit or crank nickolson discretizations
    Because the exact format changes between implicit and C-N, centervalue and neighbor 
        value must be explicitly specified
    Parameter order is specified as zsize then ysize then xsize, because the dimensional order of 3d arrays is [z, y, x]
    Matrix shows periodic boundary conditions!
    Warning: non-adi, non-gmres 3d implicit or C-N solvers will be **extremely** slow unless they are *very* small!
    """
    matrix3d = np.zeros([xsize*ysize*zsize, xsize*ysize*zsize])
    matrix2d = diffusion_matrix_2d(ysize, xsize, centervalue, neighborvalue)
    for i in range(zsize):
        matrix3d[xsize*ysize*i:xsize*ysize*(i+1), xsize*ysize*i:xsize*ysize*(i+1)] = matrix2d
    matrix3d = np.roll(matrix3d, xsize*ysize, 0)
    np.fill_diagonal(matrix3d, neighborvalue)
    matrix3d = np.roll(matrix3d, -2*xsize*ysize, 0)
    np.fill_diagonal(matrix3d, neighborvalue)
    matrix3d = np.roll(matrix3d, xsize*ysize, 0)
    return matrix3d
    

def engine_ExplicitDiffusion(sim):
    """
    Computes the discretization of the diffusion equation using a purely explicit scheme
    Valid for 1, 2, or 3D simulations
    """
    dt = sim._time_step_in_seconds
    c = sim.fields[0]
    dc = dt * (sim.D * c.laplacian())
    sim.fields[0].data += dc
    
def engine_ImplicitDiffusion1D(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 1D
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    c_final = np.linalg.solve(matrix1d, c.data)
    sim.fields[0].data = c_final
    
def engine_ImplicitDiffusion1D_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 1D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    c_final, exitCode = gmres(matrix1d, c.data, atol='legacy')
    sim.fields[0].data = c_final
    
def engine_ImplicitDiffusion2D(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 2D
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha)
    c_final = np.linalg.solve(matrix2d, c.data.flatten())
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_ImplicitDiffusion2D_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 2D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha)
    c_final, exitCode = gmres(matrix2d, c.data.flatten(), atol='legacy')
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_ImplicitDiffusion3D(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 3D
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha)
    c_final = np.linalg.solve(matrix3d, c.data.flatten())
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_ImplicitDiffusion3D_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 3D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha)
    c_final, exitCode = gmres(matrix3d, c.data.flatten(), atol='legacy')
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_CrankNicolsonDiffusion1D(sim):
    """
    Computes the discretization of the diffusion equation using the Crank-Nicolson method in 1D
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    explicit_c_half = (1-2*alpha)*c.data + alpha*(np.roll(c.data, 1, 0) + np.roll(c.data, -1, 0))
    c_final = np.linalg.solve(matrix1d, explicit_c_half)
    sim.fields[0].data = c_final
    
def engine_CrankNicolsonDiffusion1D_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Crank-Nicolson method in 1D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    explicit_c_half = (1-2*alpha)*c.data + alpha*(np.roll(c.data, 1, 0) + np.roll(c.data, -1, 0))
    c_final, exitCode = gmres(matrix1d, explicit_c_half, atol='legacy')
    sim.fields[0].data = c_final
    
def engine_CrankNicolsonDiffusion2D(sim):
    """
    Computes the discretization of the diffusion equation using the Crank-Nicolson method in 2D
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha)
    explicit_c_half = (1-4*alpha)*c.data + alpha*(np.roll(c.data, 1, 0) + np.roll(c.data, -1, 0) + np.roll(c.data, 1, 1) + np.roll(c.data, -1, 1))
    c_final = np.linalg.solve(matrix2d, explicit_c_half.flatten())
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_CrankNicolsonDiffusion2D_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Crank-Nicolson method in 2D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha)
    explicit_c_half = (1-4*alpha)*c.data + alpha*(np.roll(c.data, 1, 0) + np.roll(c.data, -1, 0) + np.roll(c.data, 1, 1) + np.roll(c.data, -1, 1))
    c_final, exitCode = gmres(matrix2d, explicit_c_half.flatten(), atol='legacy')
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_CrankNicolsonDiffusion3D(sim):
    """
    Computes the discretization of the diffusion equation using the Crank-Nicolson method in 3D
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha)
    explicit_c_half = (1-6*alpha)*c.data + alpha*(np.roll(c.data, 1, 0) + np.roll(c.data, -1, 0) + np.roll(c.data, 1, 1) + np.roll(c.data, -1, 1) +  + np.roll(c.data, 1, 2) + np.roll(c.data, -1, 2))
    c_final = np.linalg.solve(matrix3d, explicit_c_half.flatten())
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_CrankNicolsonDiffusion3D_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Crank-Nicolson method in 3D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha)
    explicit_c_half = (1-6*alpha)*c.data + alpha*(np.roll(c.data, 1, 0) + np.roll(c.data, -1, 0) + np.roll(c.data, 1, 1) + np.roll(c.data, -1, 1) +  + np.roll(c.data, 1, 2) + np.roll(c.data, -1, 2))
    c_final, exitCode = gmres(matrix3d, explicit_c_half.flatten(), atol='legacy')
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_ImplicitDiffusion2D_ADI(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D
    Uses the function np.linalg.inv(A) to compute A^-1 directly, since it is reused several times
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        c[i] = np.dot(inv_x, c[i])
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        c[:,i] = np.dot(inv_y, c[:,i])
    sim.fields[0].data = c
    
def engine_ImplicitDiffusion2D_ADI_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        c[i], exitCode = gmres(matrix1d_x, c[i], atol='legacy')
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        c[:,i], exitCode = gmres(matrix1d_y, c[:,i], atol='legacy')
    sim.fields[0].data = c
    
def engine_ImplicitDiffusion3D_ADI(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D
    Uses the function np.linalg.inv(A) to compute A^-1 directly, since it is reused several times
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    inv_z = np.linalg.inv(matrix1d_z)
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        for j in range(dim[1]):
            c[i, j] = np.dot(inv_x, c[i, j])
    for i in range(dim[0]): #then iterate through ADI method in the y direction
        for j in range(dim[2]):
            c[i, :, j] = np.dot(inv_y, c[i, :, j])
    for i in range(dim[1]): #finally, iterate through ADI method in the z direction
        for j in range(dim[2]):
            c[:, i, j] = np.dot(inv_z, c[:, i, j])
    sim.fields[0].data = c
    
def engine_ImplicitDiffusion3D_ADI_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        for j in range(dim[1]):
            c[i, j], exitCode = gmres(matrix1d_x, c[i, j], atol='legacy')
    for i in range(dim[0]): #then iterate through ADI method in the y direction
        for j in range(dim[2]):
            c[i, :, j], exitCode = gmres(matrix1d_y, c[i, :, j], atol='legacy')
    for i in range(dim[1]): #finally, iterate through ADI method in the z direction
        for j in range(dim[2]):
            c[:, i, j], exitCode = gmres(matrix1d_z, c[:, i, j], atol='legacy')
    sim.fields[0].data = c
    
def engine_CrankNicolsonDiffusion2D_ADI(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D, 
        extended to use the Crank-Nicolson scheme
    Uses the Peaceman-Rachford discretization (explicit x + implicit y, then explicit y + implicit x)
    Uses the function np.linalg.inv(A) to compute A^-1 directly, since it is reused several times
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    c_explicit = (1-2*alpha)*c + alpha*(np.roll(c, 1, 0) + np.roll(c, -1, 0)) #explicit in x
    for i in range(dim[1]): #iterate through ADI method in the y direction first in P-R
        c[:,i] = np.dot(inv_y, c_explicit[:,i])
    c_explicit = (1-2*alpha)*c + alpha*(np.roll(c, 1, 1) + np.roll(c, -1, 1)) #explicit in y
    for i in range(dim[0]): #then iterate through ADI method in the x direction
        c[i] = np.dot(inv_x, c_explicit[i])
    sim.fields[0].data = c
    
def engine_CrankNicolsonDiffusion2D_ADI_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D, 
        extended to use the Crank-Nicolson scheme
    Uses the Peaceman-Rachford discretization (explicit x + implicit y, then explicit y + implicit x)
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    c_explicit = (1-2*alpha)*c + alpha*(np.roll(c, 1, 0) + np.roll(c, -1, 0)) #explicit in x
    for i in range(dim[1]): #iterate through ADI method in the y direction first for P-R
        c[:,i], exitCode = gmres(matrix1d_y, c_explicit[:,i], atol='legacy')
    c_explicit = (1-2*alpha)*c + alpha*(np.roll(c, 1, 1) + np.roll(c, -1, 1)) #explicit in y
    for i in range(dim[0]): #then iterate through ADI method in the x direction
        c[i], exitCode = gmres(matrix1d_x, c_explicit[i], atol='legacy')
    sim.fields[0].data = c
    
def engine_CrankNicolsonDiffusion3D_ADI(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 3D, 
        extended to use the Crank-Nicolson scheme
    Uses an extended Peaceman-Rachford discretization (explicit x + implicit y, then explicit y + implicit z, 
        then explicit z + implicit x)
    Uses the function np.linalg.inv(A) to compute A^-1 directly, since it is reused several times
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    inv_z = np.linalg.inv(matrix1d_z)
    c = (1-2*alpha)*c + alpha*(np.roll(c, 1, 0) + np.roll(c, -1, 0)) #explicit in x
    for i in range(dim[0]): #iterate through ADI method in the y direction first in extended P-R
        for j in range(dim[2]):
            c[i, :, j] = np.dot(inv_y, c[i, :, j])
    c = (1-2*alpha)*c + alpha*(np.roll(c, 1, 1) + np.roll(c, -1, 1)) #explicit in y
    for i in range(dim[1]): #then iterate through ADI method in the z direction
        for j in range(dim[2]):
            c[:, i, j] = np.dot(inv_z, c[:, i, j])
    c = (1-2*alpha)*c + alpha*(np.roll(c, 1, 2) + np.roll(c, -1, 2)) #explicit in z
    for i in range(dim[0]): #finally, iterate through ADI method in the x direction
        for j in range(dim[1]):
            c[i, j] = np.dot(inv_x, c[i, j])
    sim.fields[0].data = c
    
def engine_CrankNicolsonDiffusion3D_ADI_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 3D, 
        extended to use the Crank-Nicolson scheme
    Uses an extended Peaceman-Rachford discretization (explicit x + implicit y, then explicit y + implicit z, 
        then explicit z + implicit x)
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
        the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    alpha = 0.5*sim.D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha)
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha)
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    c = (1-2*alpha)*c + alpha*(np.roll(c, 1, 0) + np.roll(c, -1, 0)) #explicit in x
    for i in range(dim[0]): #iterate through ADI method in the y direction first in extended P-R
        for j in range(dim[2]):
            c[i, :, j], exitCode = gmres(matrix1d_y, c[i, :, j], atol='legacy')
    c = (1-2*alpha)*c + alpha*(np.roll(c, 1, 1) + np.roll(c, -1, 1)) #explicit in y
    for i in range(dim[1]): #then iterate through ADI method in the z direction
        for j in range(dim[2]):
            c[:, i, j], exitCode = gmres(matrix1d_z, c[:, i, j], atol='legacy')
    c = (1-2*alpha)*c + alpha*(np.roll(c, 1, 2) + np.roll(c, -1, 2)) #explicit in z
    for i in range(dim[0]): #finally, iterate through ADI method in the x direction
        for j in range(dim[1]):
            c[i, j], exitCode = gmres(matrix1d_x, c[i, j], atol='legacy')
    sim.fields[0].data = c
        

def init_Diffusion(sim, dim, solver="explicit", gmres=False, adi=False):
    sim.D = 0.1
    sim.set_dimensions(dim)
    sim.set_cell_spacing(1.)
    c = np.zeros(dim)
    if(len(dim) == 1):
        length = dim[0]
        c[length // 4:3 * length // 4] = 1
    elif(len(dim) == 2):
        length = dim[0]
        width = dim[1]
        c[length // 4:3 * length // 4][width // 4:3 * width // 4] = 1
    elif(len(dim) == 3):
        length = dim[0]
        width = dim[1]
        depth = dim[2]
        c[length // 4:3 * length // 4][width // 4:3 * width // 4][depth // 4:3 * depth // 4] = 1
    c_field = Field(data=c, name="c", simulation=sim)
    sim.add_field(c_field)
    if (solver == "explicit"):
        sim.set_engine(engine_ExplicitDiffusion)
    elif (solver == "implicit"):
        if(len(dim) == 1):
            if(gmres):
                sim.set_engine(engine_ImplicitDiffusion1D_GMRES)
            else:
                sim.set_engine(engine_ImplicitDiffusion1D)
        elif(len(dim) == 2):
            if(gmres):
                if(adi):
                    sim.set_engine(engine_ImplicitDiffusion2D_ADI_GMRES)
                else:
                    sim.set_engine(engine_ImplicitDiffusion2D_GMRES)
            else:
                if(adi):
                    sim.set_engine(engine_ImplicitDiffusion2D_ADI)
                else:
                    sim.set_engine(engine_ImplicitDiffusion2D)
        elif(len(dim) == 3):
            if(gmres):
                if(adi):
                    sim.set_engine(engine_ImplicitDiffusion3D_ADI_GMRES)
                else:
                    sim.set_engine(engine_ImplicitDiffusion3D_GMRES)
            else:
                if(adi):
                    sim.set_engine(engine_ImplicitDiffusion3D_ADI)
                else:
                    sim.set_engine(engine_ImplicitDiffusion3D)
    elif (solver == "crank-nicolson"):
        if(len(dim) == 1):
            if(gmres):
                sim.set_engine(engine_CrankNicolsonDiffusion1D_GMRES)
            else:
                sim.set_engine(engine_CrankNicolsonDiffusion1D)
        elif(len(dim) == 2):
            if(gmres):
                if(adi):
                    sim.set_engine(engine_CrankNicolsonDiffusion2D_ADI_GMRES)
                else:
                    sim.set_engine(engine_CrankNicolsonDiffusion2D_GMRES)
            else:
                if(adi):
                    sim.set_engine(engine_CrankNicolsonDiffusion2D_ADI)
                else:
                    sim.set_engine(engine_CrankNicolsonDiffusion2D)
        elif(len(dim) == 3):
            if(gmres):
                if(adi):
                    sim.set_engine(engine_CrankNicolsonDiffusion3D_ADI_GMRES)
                else:
                    sim.set_engine(engine_CrankNicolsonDiffusion3D_GMRES)
            else:
                if(adi):
                    sim.set_engine(engine_CrankNicolsonDiffusion3D_ADI)
                else:
                    sim.set_engine(engine_CrankNicolsonDiffusion3D)