import numpy as np
from scipy.sparse.linalg import gmres
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
        
try:
    from cupyx import jit
    import cupy as cp
except:
    import pyphasefield.jit_placeholder as jit
    import pyphasefield.jit_placeholder as cp



def diffusion_matrix_1d(xsize, centervalue, neighborvalue, boundary_conditions=None):
    """
    Creates a matrix for the solution of 1d implicit or crank nickolson discretizations
    
    Because the exact format changes between implicit and C-N, and this method is reused 
    in 2D and 3D cases, centervalue and neighbor value must be explicitly specified
    
    Parameters
    ----------
    xsize : int
        Size of one dimension of the square NxN implicit matrix, equal to the number of elements in the 1D phase field model
    centervalue : float
        Value inserted into the central diagonal of the implicit matrix. 
    neighborvalue : float
        Value inserted into the two just-off-center diagonals of the implicit matrix.
    boundary_conditions : list of str, optional
        Boundary conditions for this dimension [left, right]. Options: "PERIODIC", "NEUMANN", "DIRICHLET"
        If None, defaults to periodic boundary conditions
        
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
    
    For different boundary conditions:
    - PERIODIC: Wraps around (default behavior)
    - NEUMANN: Zero flux at boundaries (modifies diagonal values)
    - DIRICHLET: Fixed values at boundaries (handled separately in solver)
    
    """
    matrix1d = np.zeros([xsize, xsize])
    
    # Fill main diagonal
    np.fill_diagonal(matrix1d, centervalue)
    
    # Fill off-diagonals
    if xsize > 1:
        # Upper diagonal (i, i+1)
        np.fill_diagonal(matrix1d[:-1, 1:], neighborvalue)
        # Lower diagonal (i, i-1)
        np.fill_diagonal(matrix1d[1:, :-1], neighborvalue)
    
    # Handle boundary conditions
    if boundary_conditions is None:
        boundary_conditions = ["PERIODIC", "PERIODIC"]
    
    # Left boundary
    if boundary_conditions[0] == "PERIODIC":
        matrix1d[0, -1] = neighborvalue  # Connect first to last
    elif boundary_conditions[0] == "NEUMANN":
        # For Neumann BC, the ghost cell equals the boundary cell
        # This effectively removes one neighbor contribution
        matrix1d[0, 0] = centervalue + neighborvalue
    # DIRICHLET: No special handling needed in matrix
    
    # Right boundary  
    if boundary_conditions[1] == "PERIODIC":
        matrix1d[-1, 0] = neighborvalue  # Connect last to first
    elif boundary_conditions[1] == "NEUMANN":
        # For Neumann BC, the ghost cell equals the boundary cell
        matrix1d[-1, -1] = centervalue + neighborvalue
    # DIRICHLET: No special handling needed in matrix
        
    return matrix1d

def diffusion_matrix_2d(ysize, xsize, centervalue, neighborvalue, boundary_conditions=None):
    """
    Creates a matrix for the solution of 2d implicit or crank nickolson discretizations
    
    Because the exact format changes between implicit and C-N, and this method is reused 
    in 3D cases, centervalue and neighbor value must be explicitly specified
    
    Parameter order is specified as ysize then xsize, because the dimensional order of 2d arrays is [y, x]
    
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
    boundary_conditions : list of lists of str, optional
        Boundary conditions for each dimension [[x_left, x_right], [y_left, y_right]]
        Options: "PERIODIC", "NEUMANN", "DIRICHLET"
        If None, defaults to periodic boundary conditions
        
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
    if boundary_conditions is None:
        boundary_conditions = [["PERIODIC", "PERIODIC"], ["PERIODIC", "PERIODIC"]]
    
    matrix2d = np.zeros([xsize*ysize, xsize*ysize])
    
    # Build the matrix row by row
    for j in range(ysize):
        for i in range(xsize):
            row = j * xsize + i
            
            # Diagonal element
            matrix2d[row, row] = centervalue
            
            # X-direction neighbors
            # Left neighbor (i-1)
            if i > 0:
                matrix2d[row, row - 1] = neighborvalue
            elif boundary_conditions[0][0] == "PERIODIC":
                matrix2d[row, row + xsize - 1] = neighborvalue  # Connect to rightmost
            elif boundary_conditions[0][0] == "NEUMANN":
                matrix2d[row, row] += neighborvalue  # Modify diagonal
            
            # Right neighbor (i+1)
            if i < xsize - 1:
                matrix2d[row, row + 1] = neighborvalue
            elif boundary_conditions[0][1] == "PERIODIC":
                matrix2d[row, row - xsize + 1] = neighborvalue  # Connect to leftmost
            elif boundary_conditions[0][1] == "NEUMANN":
                matrix2d[row, row] += neighborvalue  # Modify diagonal
            
            # Y-direction neighbors
            # Bottom neighbor (j-1)
            if j > 0:
                matrix2d[row, row - xsize] = neighborvalue
            elif boundary_conditions[1][0] == "PERIODIC":
                matrix2d[row, row + xsize * (ysize - 1)] = neighborvalue  # Connect to top
            elif boundary_conditions[1][0] == "NEUMANN":
                matrix2d[row, row] += neighborvalue  # Modify diagonal
            
            # Top neighbor (j+1)
            if j < ysize - 1:
                matrix2d[row, row + xsize] = neighborvalue
            elif boundary_conditions[1][1] == "PERIODIC":
                matrix2d[row, row - xsize * (ysize - 1)] = neighborvalue  # Connect to bottom
            elif boundary_conditions[1][1] == "NEUMANN":
                matrix2d[row, row] += neighborvalue  # Modify diagonal
    
    return matrix2d

def diffusion_matrix_3d(zsize, ysize, xsize, centervalue, neighborvalue, boundary_conditions=None):
    """
    Creates a matrix for the solution of 3d implicit or crank nickolson discretizations
    
    Because the exact format changes between implicit and C-N, centervalue and neighbor 
    value must be explicitly specified
    
    Parameter order is specified as zsize then ysize then xsize, because the dimensional order of 3d arrays is [z, y, x]
    
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
    boundary_conditions : list of lists of str, optional
        Boundary conditions for each dimension [[x_left, x_right], [y_left, y_right], [z_left, z_right]]
        Options: "PERIODIC", "NEUMANN", "DIRICHLET"
        If None, defaults to periodic boundary conditions
        
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
    if boundary_conditions is None:
        boundary_conditions = [["PERIODIC", "PERIODIC"], ["PERIODIC", "PERIODIC"], ["PERIODIC", "PERIODIC"]]
    
    matrix3d = np.zeros([xsize*ysize*zsize, xsize*ysize*zsize])
    
    # Build the matrix row by row
    for k in range(zsize):
        for j in range(ysize):
            for i in range(xsize):
                row = k * xsize * ysize + j * xsize + i
                
                # Diagonal element
                matrix3d[row, row] = centervalue
                
                # X-direction neighbors
                # Left neighbor (i-1)
                if i > 0:
                    matrix3d[row, row - 1] = neighborvalue
                elif boundary_conditions[0][0] == "PERIODIC":
                    matrix3d[row, row + xsize - 1] = neighborvalue  # Connect to rightmost
                elif boundary_conditions[0][0] == "NEUMANN":
                    matrix3d[row, row] += neighborvalue  # Modify diagonal
                
                # Right neighbor (i+1)
                if i < xsize - 1:
                    matrix3d[row, row + 1] = neighborvalue
                elif boundary_conditions[0][1] == "PERIODIC":
                    matrix3d[row, row - xsize + 1] = neighborvalue  # Connect to leftmost
                elif boundary_conditions[0][1] == "NEUMANN":
                    matrix3d[row, row] += neighborvalue  # Modify diagonal
                
                # Y-direction neighbors
                # Bottom neighbor (j-1)
                if j > 0:
                    matrix3d[row, row - xsize] = neighborvalue
                elif boundary_conditions[1][0] == "PERIODIC":
                    matrix3d[row, row + xsize * (ysize - 1)] = neighborvalue  # Connect to top
                elif boundary_conditions[1][0] == "NEUMANN":
                    matrix3d[row, row] += neighborvalue  # Modify diagonal
                
                # Top neighbor (j+1)
                if j < ysize - 1:
                    matrix3d[row, row + xsize] = neighborvalue
                elif boundary_conditions[1][1] == "PERIODIC":
                    matrix3d[row, row - xsize * (ysize - 1)] = neighborvalue  # Connect to bottom
                elif boundary_conditions[1][1] == "NEUMANN":
                    matrix3d[row, row] += neighborvalue  # Modify diagonal
                
                # Z-direction neighbors
                # Front neighbor (k-1)
                if k > 0:
                    matrix3d[row, row - xsize * ysize] = neighborvalue
                elif boundary_conditions[2][0] == "PERIODIC":
                    matrix3d[row, row + xsize * ysize * (zsize - 1)] = neighborvalue  # Connect to back
                elif boundary_conditions[2][0] == "NEUMANN":
                    matrix3d[row, row] += neighborvalue  # Modify diagonal
                
                # Back neighbor (k+1)
                if k < zsize - 1:
                    matrix3d[row, row + xsize * ysize] = neighborvalue
                elif boundary_conditions[2][1] == "PERIODIC":
                    matrix3d[row, row - xsize * ysize * (zsize - 1)] = neighborvalue  # Connect to front
                elif boundary_conditions[2][1] == "NEUMANN":
                    matrix3d[row, row] += neighborvalue  # Modify diagonal
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points
    c_interior = c.data[c._slice]
    
    # Build matrix with boundary conditions
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    
    # Solve for interior points
    c_final = np.linalg.solve(matrix1d, c_interior)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points
    c_interior = c.data[c._slice]
    
    # Build matrix with boundary conditions
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    
    # Solve for interior points
    c_final, exitCode = gmres(matrix1d, c_interior, atol=1e-9)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points
    c_interior = c.data[c._slice].flatten()
    
    # Build matrix with boundary conditions
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final = np.linalg.solve(matrix2d, c_interior)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points
    c_interior = c.data[c._slice].flatten()
    
    # Build matrix with boundary conditions
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final, exitCode = gmres(matrix2d, c_interior, atol=1e-9)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points
    c_interior = c.data[c._slice].flatten()
    
    # Build matrix with boundary conditions
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final = np.linalg.solve(matrix3d, c_interior)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points
    c_interior = c.data[c._slice].flatten()
    
    # Build matrix with boundary conditions
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final, exitCode = gmres(matrix3d, c_interior, atol=1e-9)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points for calculation
    c_interior = c.data[c._slice]
    
    # Compute explicit part using laplacian on interior points
    explicit_c_half = c_interior + 0.5 * dt * D * c.laplacian()[c._slice]
    
    # Build matrix with boundary conditions
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    
    # Solve for interior points
    c_final = np.linalg.solve(matrix1d, explicit_c_half)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points for calculation
    c_interior = c.data[c._slice]
    
    # Compute explicit part using laplacian on interior points
    explicit_c_half = c_interior + 0.5 * dt * D * c.laplacian()[c._slice]
    
    # Build matrix with boundary conditions
    matrix1d = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    
    # Solve for interior points
    c_final, exitCode = gmres(matrix1d, explicit_c_half, atol=1e-9)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points for calculation
    c_interior = c.data[c._slice]
    
    # Compute explicit part using laplacian on interior points
    explicit_c_half = c_interior + 0.5 * dt * D * c.laplacian()[c._slice]
    
    # Build matrix with boundary conditions
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final = np.linalg.solve(matrix2d, explicit_c_half.flatten())
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points for calculation
    c_interior = c.data[c._slice]
    
    # Compute explicit part using laplacian on interior points
    explicit_c_half = c_interior + 0.5 * dt * D * c.laplacian()[c._slice]
    
    # Build matrix with boundary conditions
    matrix2d = diffusion_matrix_2d(dim[0], dim[1], 1+4*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final, exitCode = gmres(matrix2d, explicit_c_half.flatten(), atol=1e-9)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points for calculation
    c_interior = c.data[c._slice]
    
    # Compute explicit part using laplacian on interior points
    explicit_c_half = c_interior + 0.5 * dt * D * c.laplacian()[c._slice]
    
    # Build matrix with boundary conditions
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final = np.linalg.solve(matrix3d, explicit_c_half.flatten())
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
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
    bcs = sim._boundary_conditions_type
    
    # Extract interior points for calculation
    c_interior = c.data[c._slice]
    
    # Compute explicit part using laplacian on interior points
    explicit_c_half = c_interior + 0.5 * dt * D * c.laplacian()[c._slice]
    
    # Build matrix with boundary conditions
    matrix3d = diffusion_matrix_3d(dim[0], dim[1], dim[2], 1+6*alpha, -alpha, boundary_conditions=bcs)
    
    # Solve for interior points
    c_final, exitCode = gmres(matrix3d, explicit_c_half.flatten(), atol=1e-9)
    
    # Put solution back into interior points
    c.data[c._slice] = c_final.reshape(dim)
    
def engine_ImplicitDiffusion2D_ADI(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D
    
    Uses the function np.linalg.inv(A) to compute A^-1 directly, since it is reused several times
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Solve in x-direction for each row
    for i in range(dim[0]):
        c_interior[i] = np.dot(inv_x, c_interior[i])
    
    # Solve in y-direction for each column
    for i in range(dim[1]):
        c_interior[:,i] = np.dot(inv_y, c_interior[:,i])
    
    # Put solution back
    c.data[c._slice] = c_interior
    
def engine_ImplicitDiffusion2D_ADI_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D
    
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
    the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Solve in x-direction for each row
    for i in range(dim[0]):
        c_interior[i], exitCode = gmres(matrix1d_x, c_interior[i], atol=1e-9)
    
    # Solve in y-direction for each column
    for i in range(dim[1]):
        c_interior[:,i], exitCode = gmres(matrix1d_y, c_interior[:,i], atol=1e-9)
    
    # Put solution back
    c.data[c._slice] = c_interior
    
def engine_ImplicitDiffusion3D_ADI(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 3D
    
    Uses the function np.linalg.inv(A) to compute A^-1 directly, since it is reused several times
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[2])
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    inv_z = np.linalg.inv(matrix1d_z)
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Solve in x-direction
    for i in range(dim[0]):
        for j in range(dim[1]):
            c_interior[i, j] = np.dot(inv_x, c_interior[i, j])
    
    # Solve in y-direction
    for i in range(dim[0]):
        for j in range(dim[2]):
            c_interior[i, :, j] = np.dot(inv_y, c_interior[i, :, j])
    
    # Solve in z-direction
    for i in range(dim[1]):
        for j in range(dim[2]):
            c_interior[:, i, j] = np.dot(inv_z, c_interior[:, i, j])
    
    # Put solution back
    c.data[c._slice] = c_interior
    
def engine_ImplicitDiffusion3D_ADI_GMRES(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 3D
    
    Uses the function scipy.sparse.linalg.gmres(A, b) to **quickly but approximately** solve 
    the equation Ax=b for the matrix A and vectors x and b
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[2])
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Solve in x-direction
    for i in range(dim[0]):
        for j in range(dim[1]):
            c_interior[i, j], exitCode = gmres(matrix1d_x, c_interior[i, j], atol=1e-9)
    
    # Solve in y-direction
    for i in range(dim[0]):
        for j in range(dim[2]):
            c_interior[i, :, j], exitCode = gmres(matrix1d_y, c_interior[i, :, j], atol=1e-9)
    
    # Solve in z-direction
    for i in range(dim[1]):
        for j in range(dim[2]):
            c_interior[:, i, j], exitCode = gmres(matrix1d_z, c_interior[:, i, j], atol=1e-9)
    
    # Put solution back
    c.data[c._slice] = c_interior
    
def engine_CrankNicolsonDiffusion2D_ADI(sim):
    """
    Computes the discretization of the diffusion equation using the Alternating Direction Implicit method for 2D, 
    extended to use the Crank-Nicolson scheme
    
    Uses the Peaceman-Rachford discretization (explicit x + implicit y, then explicit y + implicit x)
    
    Uses the function np.linalg.inv(A) to compute A^-1 directly, since it is reused several times
    """
    dt = sim.dt
    dx = sim.get_cell_spacing()
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Step 1: explicit in x, implicit in y
    # Compute explicit part in x-direction
    laplacian_x = c.laplacian()[c._slice]
    c_explicit = c_interior + alpha * laplacian_x
    
    # Solve implicit in y-direction
    for i in range(dim[0]):
        c_interior[i] = np.dot(inv_y, c_explicit[i])
    
    # Step 2: explicit in y, implicit in x  
    # Compute explicit part in y-direction
    c.data[c._slice] = c_interior  # Update field for laplacian calculation
    laplacian_y = c.laplacian()[c._slice]
    c_explicit = c_interior + alpha * laplacian_y
    
    # Solve implicit in x-direction
    for i in range(dim[1]):
        c_interior[:,i] = np.dot(inv_x, c_explicit[:,i])
    
    # Put solution back
    c.data[c._slice] = c_interior
    
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
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Step 1: explicit in x, implicit in y
    # Compute explicit part in x-direction
    laplacian_x = c.laplacian()[c._slice]
    c_explicit = c_interior + alpha * laplacian_x
    
    # Solve implicit in y-direction
    for i in range(dim[0]):
        c_interior[i], exitCode = gmres(matrix1d_y, c_explicit[i], atol=1e-9)
    
    # Step 2: explicit in y, implicit in x  
    # Compute explicit part in y-direction
    c.data[c._slice] = c_interior  # Update field for laplacian calculation
    laplacian_y = c.laplacian()[c._slice]
    c_explicit = c_interior + alpha * laplacian_y
    
    # Solve implicit in x-direction
    for i in range(dim[1]):
        c_interior[:,i], exitCode = gmres(matrix1d_x, c_explicit[:,i], atol=1e-9)
    
    # Put solution back
    c.data[c._slice] = c_interior
    
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
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[2])
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    inv_z = np.linalg.inv(matrix1d_z)
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Step 1: explicit in x, implicit in y
    laplacian = c.laplacian()[c._slice]
    c_interior = c_interior + alpha * laplacian
    for i in range(dim[0]):
        for j in range(dim[2]):
            c_interior[i, :, j] = np.dot(inv_y, c_interior[i, :, j])
    
    # Step 2: explicit in y, implicit in z
    c.data[c._slice] = c_interior
    laplacian = c.laplacian()[c._slice]
    c_interior = c_interior + alpha * laplacian
    for i in range(dim[1]):
        for j in range(dim[2]):
            c_interior[:, i, j] = np.dot(inv_z, c_interior[:, i, j])
    
    # Step 3: explicit in z, implicit in x
    c.data[c._slice] = c_interior
    laplacian = c.laplacian()[c._slice]
    c_interior = c_interior + alpha * laplacian
    for i in range(dim[0]):
        for j in range(dim[1]):
            c_interior[i, j] = np.dot(inv_x, c_interior[i, j])
    
    # Put solution back
    c.data[c._slice] = c_interior
    
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
    c = sim.fields[0]
    D = sim.user_data["D"]
    alpha = 0.5*D*dt/dx**2
    dim = sim.get_dimensions()
    bcs = sim._boundary_conditions_type
    
    # Build matrices with boundary conditions
    matrix1d_x = diffusion_matrix_1d(dim[2], 1+2*alpha, -alpha, boundary_conditions=bcs[0])
    matrix1d_y = diffusion_matrix_1d(dim[1], 1+2*alpha, -alpha, boundary_conditions=bcs[1])
    matrix1d_z = diffusion_matrix_1d(dim[0], 1+2*alpha, -alpha, boundary_conditions=bcs[2])
    
    # Work on interior points only
    c_interior = c.data[c._slice].copy()
    
    # Step 1: explicit in x, implicit in y
    laplacian = c.laplacian()[c._slice]
    c_interior = c_interior + alpha * laplacian
    for i in range(dim[0]):
        for j in range(dim[2]):
            c_interior[i, :, j], exitCode = gmres(matrix1d_y, c_interior[i, :, j], atol=1e-9)
    
    # Step 2: explicit in y, implicit in z
    c.data[c._slice] = c_interior
    laplacian = c.laplacian()[c._slice]
    c_interior = c_interior + alpha * laplacian
    for i in range(dim[1]):
        for j in range(dim[2]):
            c_interior[:, i, j], exitCode = gmres(matrix1d_z, c_interior[:, i, j], atol=1e-9)
    
    # Step 3: explicit in z, implicit in x
    c.data[c._slice] = c_interior
    laplacian = c.laplacian()[c._slice]
    c_interior = c_interior + alpha * laplacian
    for i in range(dim[0]):
        for j in range(dim[1]):
            c_interior[i, j], exitCode = gmres(matrix1d_x, c_interior[i, j], atol=1e-9)
    
    # Put solution back
    c.data[c._slice] = c_interior   
    
@jit.rawkernel()
def diffusion_kernel_1D(fields, fields_out, D, dx, dt):
    startx = jit.grid(1)      
    stridex = jit.gridsize(1) 

    alpha = D*dt/(dx*dx) #laplacian coefficient in diffusion discretization

    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(startx, c.shape[1], stridex):
        c_out[i] = c[i]+alpha*(-2*c[i]+c[i+1]+c[i-1])

@jit.rawkernel()
def diffusion_kernel_2D(fields, fields_out, D, dx, dt):
    startx, starty = jit.grid(2)      
    stridex, stridey = jit.gridsize(2) 

    alpha = D*dt/(dx*dx) #laplacian coefficient in diffusion discretization

    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(starty, c.shape[0], stridey):
        for j in range(startx, c.shape[1], stridex):
            c_out[i][j] = c[i][j]+alpha*(-4*c[i][j]+c[i+1][j]+c[i-1][j]+c[i][j+1]+c[i][j-1])

@jit.rawkernel()
def diffusion_kernel_3D(fields, fields_out, D, dx, dt):
    startx, starty, startz = jit.grid(3)      
    stridex, stridey, stridez = jit.gridsize(3) 

    alpha = D*dt/(dx*dx) #laplacian coefficient in diffusion discretization

    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(startz, c.shape[0], stridez):
        for j in range(starty, c.shape[1], stridey):
            for k in range(startx, c.shape[2], stridex):
                c_out[i][j][k] = c[i][j][k]+alpha*(-6*c[i][j][k]+c[i+1][j][k]+c[i-1][j][k]+c[i][j+1][k]+c[i][j-1][k]+c[i][j][k+1]+c[i][j][k-1])
            
def engine_DiffusionGPU(sim):
    cp.cuda.runtime.deviceSynchronize()
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
                if(len(self.dimensions) == 1):
                    if(gmres):
                        engine_ImplicitDiffusion1D_GMRES(self)
                    else:
                        engine_ImplicitDiffusion1D(self)
                elif(len(self.dimensions) == 2):
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
                elif(len(self.dimensions) == 3):
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
                if(len(self.dimensions) == 1):
                    if(gmres):
                        engine_CrankNicolsonDiffusion1D_GMRES(self)
                    else:
                        engine_CrankNicolsonDiffusion1D(self)
                elif(len(self.dimensions) == 2):
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
                elif(len(self.dimensions) == 3):
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

