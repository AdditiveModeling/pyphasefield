import numpy as np
from scipy.sparse.linalg import gmres
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
        
try:
    from numba import cuda
except:
    import pyphasefield.jit_placeholder as cuda



def diffusion_matrix_1d(xsize, centervalue, neighborvalue):
    """
    Creates a matrix for the solution of 1d implicit or crank nickolson discretizations
    
    Because the exact format changes between implicit and C-N, and this method is reused 
    in 2D and 3D cases, centervalue and neighbor value must be explicitly specified
    
    Matrix shows periodic boundary conditions!
    
    Parameters
    ----------
    xsize : int
        Size of one dimension of the square NxN implicit matrix, equal to the number of elements in the 1D phase field model
    centervalue : float
        Value inserted into the central diagonal of the implicit matrix. 
    neighborvalue : float
        Value inserted into the two just-off-center diagonals of the implicit matrix.
        
    Returns
    -------
    2D NumPy ndarray representation of implicit matrix, with shape [xsize, xsize]
        
    Notes
    -----
    Consider the implicit 1D diffusion matrix with generic discretization term equal to the following:
    
    $$(c_{x}^{t+1} - c_{x}^{t})/dt = (D/(\\Delta x^2))(c_{x+1}^{t+1} + c_{x-1}^{t+1} - 2c_{x}^{t+1})$$
    
    This can be rearranged as to express c_{x}^{t} as a function of c_{x}^{t+1}, c_{x-1}^{t+1}, and c_{x+1}^{t+1}
    (Also, let a = D*dt/(\\Delta x^2) ):
    
    $$c_{x}^{t} = (-a)c_{x+1}^{t+1} + (-a)c_{x-1}^{t+1} + (1+2a)c_{x}^{t+1}$$
    
    The implicit matrix composed of these terms is defined as follows: 
    The central diagonal (centervalue) equals the coefficient of c_{x}^{t+1}: 1+2a, or 1+2*D*\\Delta t/(\\Delta x^2)
    The neighboring diagonals to the center (neighborvalue) equals the coefficient of c_{x-1}^{t+1} or c_{x+1}^{t+1}: 
    -a, or -D*\\Delta t/(\\Delta x^2)
    
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
    
    Parameters
    ----------
    ysize : int
        Equal to the number of elements along the y-axis in the 2D phase field model
        xsize*ysize is equal to the length of one dimension of the square NxN implicit matrix
    xsize : int
        Equal to the number of elements along the x-axis in the 2D phase field model
        xsize*ysize is equal to the length of one dimension of the square NxN implicit matrix
    centervalue : float
        Value inserted into the central diagonal of the implicit matrix. 
    neighborvalue : float
        Value inserted into the four just-off-center diagonals of the 2D implicit matrix.
        
    Returns
    -------
    2D NumPy ndarray representation of implicit matrix, with shape [xsize*ysize, xsize*ysize]
        
    Notes
    -----
    Consider the implicit 2D diffusion matrix with generic discretization term equal to the following:
    
    $$(c_{x, y}^{t+1} - c_{x, y}^{t})/dt = (D/(\\Delta x^2))(c_{x+1, y}^{t+1} + c_{x-1, y}^{t+1} 
    + c_{x, y+1}^{t+1} + c_{x, y-1}^{t+1} - 4c_{x, y}^{t+1})$$
    
    This can be rearranged as to express c_{x, y}^{t} as a function of c_{x, y}^{t+1}, c_{x-1, y}^{t+1}, 
    c_{x+1, y}^{t+1}, c_{x, y-1}^{t+1}, and c_{x, y+1}^{t+1}
    (Also, let a = D*dt/(\\Delta x^2) ):
    
    $$c_{x, y}^{t} = (-a)c_{x+1, y}^{t+1} + (-a)c_{x-1, y}^{t+1} + (-a)c_{x, y+1}^{t+1} 
    + (-a)c_{x, y-1}^{t+1} + (1+4a)c_{x, y}^{t+1}$$
    
    The implicit matrix composed of these terms is defined as follows: 
    The central diagonal (centervalue) equals the coefficient of c_{x, y}^{t+1}: 1+4a, or 1+4*D*\\Delta t/(\\Delta x^2)
    The neighboring diagonals to the center (neighborvalue) equals the coefficient of c_{x-1, y}^{t+1} (or other similar terms): 
    -a, or -D*\\Delta t/(\\Delta x^2)
    
    Note that two of the "neighboring" diagonals are separated by a significant number of cells in the matrix, however 
    they are still considered to be "neighbors" conceptually
    
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
    
    Parameters
    ----------
    zsize : int
        Equal to the number of elements along the z-axis in the 3D phase field model
        xsize*ysize*zsize is equal to the length of one dimension of the square NxN implicit matrix
    ysize : int
        Equal to the number of elements along the y-axis in the 3D phase field model
        xsize*ysize*zsize is equal to the length of one dimension of the square NxN implicit matrix
    xsize : int
        Equal to the number of elements along the x-axis in the 3D phase field model
        xsize*ysize*zsize is equal to the length of one dimension of the square NxN implicit matrix
    centervalue : float
        Value inserted into the central diagonal of the implicit matrix. 
    neighborvalue : float
        Value inserted into the six just-off-center diagonals of the 3D implicit matrix.
        
    Returns
    -------
    2D NumPy ndarray representation of implicit matrix, with shape [xsize*ysize*zsize, xsize*ysize*zsize]
    
    Warnings
    -----
    Non-ADI, non-GMRES 3d implicit or C-N solvers will be **extremely** slow unless they are *very* small!
    
    Notes
    -----
    Consider the implicit 3D diffusion matrix with generic discretization term equal to the following:
    
    $$(c_{x, y, z}^{t+1} - c_{x, y, z}^{t})/dt = (D/(\\Delta x^2))(c_{x+1, y, z}^{t+1} + c_{x-1, y, z}^{t+1} 
    + c_{x, y+1, z}^{t+1} + c_{x, y-1, z}^{t+1} + c_{x, y, z+1}^{t+1} + c_{x, y, z-1}^{t+1} - 6c_{x, y, z}^{t+1})$$
    
    This can be rearranged as to express c_{x, y, z}^{t} as a function of c_{x, y, z}^{t+1}, c_{x-1, y, z}^{t+1}, 
    c_{x+1, y, z}^{t+1}, c_{x, y-1, z}^{t+1}, and c_{x, y+1, z}^{t+1}
    (Also, let a = D*dt/(\\Delta x^2) ):
    
    $$c_{x, y, z}^{t} = (-a)c_{x+1, y, z}^{t+1} + (-a)c_{x-1, y, z}^{t+1} + (-a)c_{x, y+1, z}^{t+1} + (-a)c_{x, y-1, z}^{t+1} 
    + (-a)c_{x, y, z+1}^{t+1} + (-a)c_{x, y, z-1}^{t+1} + (1+6a)c_{x, y, z}^{t+1}$$
    
    The implicit matrix composed of these terms is defined as follows: 
    The central diagonal (centervalue) equals the coefficient of c_{x, y, z}^{t+1}: 1+6a, or 1+6*D*\\Delta t/(\\Delta x^2)
    The neighboring diagonals to the center (neighborvalue) equals the coefficient of c_{x-1, y, z}^{t+1} (or other similar terms): 
    -a, or -D*\\Delta t/(\\Delta x^2)
    
    Note that four of the "neighboring" diagonals are separated by a significant number of cells in the matrix, however 
    they are still considered to be "neighbors" conceptually
    
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
    dt = sim.dt
    c = sim.fields[0]
    D = sim.user_data["D"]
    dc = dt * (D * c.laplacian())
    sim.fields[0].data += dc
    
def engine_ImplicitDiffusion1D(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 1D
    
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
    dim = sim.get_dimensions()
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha)
    c_final, exitCode = gmres(matrix1d, c.data, atol='legacy')
    sim.fields[0].data = c_final
    
def engine_ImplicitDiffusion2D(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 2D
    
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
    dim = sim.get_dimensions()
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha)
    c_final, exitCode = gmres(matrix2d, c.data.flatten(), atol='legacy')
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_ImplicitDiffusion3D(sim):
    """
    Computes the discretization of the diffusion equation using a purely implicit scheme in 3D
    
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
    dim = sim.get_dimensions()
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha)
    c_final, exitCode = gmres(matrix3d, c.data.flatten(), atol='legacy')
    sim.fields[0].data = c_final.reshape(dim)
    
def engine_CrankNicolsonDiffusion1D(sim):
    """
    Computes the discretization of the diffusion equation using the Crank-Nicolson method in 1D
    
    Uses the function np.linalg.solve(A, b) to solve the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0].data
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
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
    
@cuda.jit
def diffusion_kernel_1D(fields, fields_out, D, dx, dt):
    startx = cuda.grid(1)      
    stridex = cuda.gridsize(1) 

    alpha = D*dt/(dx*dx) #laplacian coefficient in diffusion discretization

    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(startx, c.shape[1], stridex):
        c_out[i] = c[i]+alpha*(-2*c[i]+c[i+1]+c[i-1])

@cuda.jit
def diffusion_kernel_2D(fields, fields_out, D, dx, dt):
    startx, starty = cuda.grid(2)      
    stridex, stridey = cuda.gridsize(2) 

    alpha = D*dt/(dx*dx) #laplacian coefficient in diffusion discretization

    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(starty, c.shape[0], stridey):
        for j in range(startx, c.shape[1], stridex):
            c_out[i][j] = c[i][j]+alpha*(-4*c[i][j]+c[i+1][j]+c[i-1][j]+c[i][j+1]+c[i][j-1])

@cuda.jit
def diffusion_kernel_3D(fields, fields_out, D, dx, dt):
    startx, starty, startz = cuda.grid(3)      
    stridex, stridey, startz = cuda.gridsize(3) 

    alpha = D*dt/(dx*dx) #laplacian coefficient in diffusion discretization

    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(startz, c.shape[0], stridez):
        for j in range(starty, c.shape[1], stridey):
            for k in range(startx, c.shape[2], stridex):
                c_out[i][j][k] = c[i][j][k]+alpha*(-6*c[i][j][k]+c[i+1][j][k]+c[i-1][j][k]+c[i][j+1][k]+c[i][j-1][k]+c[i][j][k+1]+c[i][j][k-1])
            
def engine_DiffusionGPU(sim):
    cuda.synchronize()
    if(len(sim.dimensions) == 1):
        diffusion_kernel_1D[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._fields_gpu_device, sim._fields_out_gpu_device, 
                                                                  sim.user_data["D"], sim.dx, sim.dt)
    elif(len(sim.dimensions) == 2):
        diffusion_kernel_2D[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._fields_out_gpu_device, 
                                                                  sim.user_data["D"], sim.dx, sim.dt)
    elif(len(sim.dimensions) == 3):
        diffusion_kernel_3D[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._fields_out_gpu_device, 
                                                                  sim.user_data["D"], sim.dx, sim.dt)
    cuda.synchronize()

class Diffusion(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def init_fields(self):
        #initialization of fields code goes here
        #runs *after* tdb and thermal data is loaded/initialized
        #runs *before* boundary conditions are initialized
        if not ("D" in self.user_data):
            self.user_data["D"] = 0.1
        if not ("solver" in self.user_data):
            self.user_data["solver"] = "explicit"
        if not ("adi" in self.user_data):
            self.user_data["adi"] = False
        if not ("gmres" in self.user_data):
            self.user_data["gmres"] = False
            
        #create field using local dimensions, but modify the array using global dimensions
        #slicing the field will account for the global -> local conversion!
        dim_global = self._global_dimensions
        dim = self.dimensions
        c = np.zeros(dim)
        
        self.add_field(c, "c")
        field = self.fields[0]
        if(len(dim) == 1):
            for i in range(dim_global[0]//100 + 1):
                field[100*i:100*i+50] = 1
        elif(len(dim) == 2):
            for i in range(dim_global[0]//100 + 1):
                for j in range(dim_global[1]//100 + 1):
                    field[100*i:100*i+50, 100*j:100*j+50] = 1
                    field[100*i+50:100*i+100, 100*j+50:100*j+100] = 1
        elif(len(dim) == 3):
            for i in range(dim_global[0]//100 + 1):
                for j in range(dim_global[1]//100 + 1):
                    for k in range(dim_global[2]//100 + 1):
                        field[100*i:100*i+50, 100*j:100*j+50, 100*j:100*j+50] = 1
                        field[100*i+50:100*i+100, 100*j+50:100*j+100, 100*j+50:100*j+100] = 1
                        
    def just_before_simulating(self):
        super().just_before_simulating()
        #additional code to run just before beginning the simulation goes below
        #runs immediately before simulating, no manual changes permitted to changes implemented here
        
    def simulation_loop(self):
        solver = self.user_data["solver"]
        gmres = self.user_data["gmres"]
        adi = self.user_data["adi"]
        if(self._framework == "GPU_SERIAL" or self._framework == "GPU_PARALLEL"):
            engine_DiffusionGPU(self)
        else: #"CPU_SERIAL"
            if (solver == "explicit"):
                engine_ExplicitDiffusion(self)
            elif (solver == "implicit"):
                if(len(dim) == 1):
                    if(gmres):
                        engine_ImplicitDiffusion1D_GMRES(self)
                    else:
                        engine_ImplicitDiffusion1D(self)
                elif(len(dim) == 2):
                    if(gmres):
                        if(adi):
                            engine_ImplicitDiffusion2D_ADI_GMRES(self)
                        else:
                            engine_ImplicitDiffusion2D_GMRES(self)
                    else:
                        if(adi):
                            engine_ImplicitDiffusion2D_ADI(self)
                        else:
                            engine_ImplicitDiffusion2D(self)
                elif(len(dim) == 3):
                    if(gmres):
                        if(adi):
                            engine_ImplicitDiffusion3D_ADI_GMRES(self)
                        else:
                            engine_ImplicitDiffusion3D_GMRES(self)
                    else:
                        if(adi):
                            engine_ImplicitDiffusion3D_ADI(self)
                        else:
                            engine_ImplicitDiffusion3D(self)
            elif (solver == "crank-nicolson"):
                if(len(dim) == 1):
                    if(gmres):
                        engine_CrankNicolsonDiffusion1D_GMRES(self)
                    else:
                        engine_CrankNicolsonDiffusion1D(self)
                elif(len(dim) == 2):
                    if(gmres):
                        if(adi):
                            engine_CrankNicolsonDiffusion2D_ADI_GMRES(self)
                        else:
                            engine_CrankNicolsonDiffusion2D_GMRES(self)
                    else:
                        if(adi):
                            engine_CrankNicolsonDiffusion2D_ADI(self)
                        else:
                            engine_CrankNicolsonDiffusion2D(self)
                elif(len(dim) == 3):
                    if(gmres):
                        if(adi):
                            engine_CrankNicolsonDiffusion3D_ADI_GMRES(self)
                        else:
                            engine_CrankNicolsonDiffusion3D_GMRES(self)
                    else:
                        if(adi):
                            engine_CrankNicolsonDiffusion3D_ADI(self)
                        else:
                            engine_CrankNicolsonDiffusion3D(self)

