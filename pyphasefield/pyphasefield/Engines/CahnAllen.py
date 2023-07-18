import numpy as np
from scipy.sparse.linalg import gmres

try:
    #import from within Engines folder
    from ..field import Field
    from ..simulation import Simulation
    from ..ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
except:
    try:
        #import classes from pyphasefield library
        from pyphasefield.field import Field
        from pyphasefield.simulation import Simulation
        from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
    except:
        raise ImportError("Cannot import from pyphasefield library!")

def implicit_matrix_1d(xsize, centervalue, neighborvalue):
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
    2D NumPy ndarray representation of implicit matrix, with shape [xsize*ysize, xsize*ysize]
        
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

def implicit_matrix_2d(ysize, xsize, centervalue, neighborvalue):
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
    for i in range(ysize):
        matrix1d = implicit_matrix_1d(xsize, centervalue[i], neighborvalue)
        matrix2d[xsize*i:xsize*(i+1), xsize*i:xsize*(i+1)] = matrix1d
    matrix2d = np.roll(matrix2d, xsize, 0)
    np.fill_diagonal(matrix2d, neighborvalue)
    matrix2d = np.roll(matrix2d, -2*xsize, 0)
    np.fill_diagonal(matrix2d, neighborvalue)
    matrix2d = np.roll(matrix2d, xsize, 0)
    return matrix2d

def engine_CahnAllenExplicit(sim):
    dt = sim.dt
    phi = sim.fields[0]
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    deltaphi = dt * M * (epsilon**2 * phi.laplacian() - 16.*W*(4.*phi.data**3 - 6.*phi.data**2 + 2.*phi.data))
    sim.fields[0].data += deltaphi
    
def engine_CahnAllenImplicit1D(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = M*dt*(epsilon**2)/(dx**2)
    matrix_source_term = 16.*W*M*dt*(4.*phi**2 - 6.*phi + 2.)
    matrix1d = implicit_matrix_1d(dim[0], 1.+2.*alpha + matrix_source_term, -alpha)
    phi_final = np.linalg.solve(matrix1d, phi)
    sim.fields[0].data = phi_final
    
def engine_CahnAllenImplicit1D_GMRES(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = M*dt*(epsilon**2)/(dx**2)
    matrix_source_term = 16.*W*M*dt*(4.*phi**2 - 6.*phi + 2.)
    matrix1d = implicit_matrix_1d(dim[0], 1+2*alpha + matrix_source_term, -alpha)
    phi_final, exitCode = gmres(matrix1d, phi, atol='legacy')
    sim.fields[0].data = phi_final
    
def engine_CahnAllenImplicit2D_GMRES(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = M*dt*(epsilon**2)/(dx**2)
    matrix_source_term = 16.*W*M*dt*(4.*phi**2 - 6.*phi + 2.)
    matrix2d = implicit_matrix_2d(dim[0], dim[1], 1.+4.*alpha+matrix_source_term, -alpha)
    phi_final, exitCode = gmres(matrix2d, phi.flatten(), atol='legacy')
    sim.fields[0].data = phi_final.reshape(dim)
    
def engine_CahnAllenImplicit2D_ADI(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = M*dt*(epsilon**2)/(dx**2)
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        matrix_source_term = 8.*W*M*dt*(4.*phi[i]**2 - 6.*phi[i] + 2.)
        matrix1d_x = implicit_matrix_1d(dim[1], 1.+2.*alpha + matrix_source_term, -alpha)
        phi[i] = np.linalg.solve(matrix1d_x, phi[i])
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        matrix_source_term = 8.*W*M*dt*(4.*phi[:,i]**2 - 6.*phi[:,i] + 2.)
        matrix1d_y = implicit_matrix_1d(dim[0], 1.+2.*alpha + matrix_source_term, -alpha)
        phi[:,i] = np.linalg.solve(matrix1d_y, phi[:,i])
    sim.fields[0].data = phi
    
def engine_CahnAllenCrankNicolson2D_ADI(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = 0.5*M*dt*(epsilon**2)/(dx**2)
    phi = phi*(1-2*alpha) + alpha*(np.roll(phi, 1, 1) + np.roll(phi, -1, 1))
    #should dt in implicit here be halved? check this later
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        matrix_source_term = 8.*W*M*dt*(4.*phi[i]**2 - 6.*phi[i] + 2.)
        matrix1d_x = implicit_matrix_1d(dim[1], 1.+2.*alpha + matrix_source_term, -alpha)
        phi[i] = np.linalg.solve(matrix1d_x, phi[i])
    phi = phi*(1-2*alpha) + alpha*(np.roll(phi, 1, 0) + np.roll(phi, -1, 0))
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        matrix_source_term = 8.*W*M*dt*(4.*phi[:,i]**2 - 6.*phi[:,i] + 2.)
        matrix1d_y = implicit_matrix_1d(dim[0], 1.+2.*alpha + matrix_source_term, -alpha)
        phi[:,i] = np.linalg.solve(matrix1d_y, phi[:,i])
    sim.fields[0].data = phi
    
def engine_CahnAllenImplicit2D_ADI_GMRES(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = M*dt*(epsilon**2)/(dx**2)
    #should dt in implicit here be halved? check this later
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        matrix_source_term = 8.*W*M*dt*(4.*phi[i]**2 - 6.*phi[i] + 2.)
        matrix1d_x = implicit_matrix_1d(dim[1], 1.+2.*alpha + matrix_source_term, -alpha)
        phi[i], exitCode = gmres(matrix1d_x, phi[i], atol='legacy')
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        matrix_source_term = 8.*W*M*dt*(4.*phi[:,i]**2 - 6.*phi[:,i] + 2.)
        matrix1d_y = implicit_matrix_1d(dim[0], 1.+2.*alpha + matrix_source_term, -alpha)
        phi[:,i], exitCode = gmres(matrix1d_y, phi[:,i], atol='legacy')
    sim.fields[0].data = phi
    
def engine_CahnAllenImplicit3D_ADI(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = M*dt*(epsilon**2)/(dx**2)
    matrix_source_term = 16.*W*M*dt*(4.*phi**2 - 6.*phi + 2.)/3.
    matrix1d_x = implicit_matrix_1d(dim[2], 1.+2.*alpha+matrix_source_term, -alpha)
    matrix1d_y = implicit_matrix_1d(dim[1], 1.+2.*alpha+matrix_source_term, -alpha)
    matrix1d_z = implicit_matrix_1d(dim[0], 1.+2.*alpha+matrix_source_term, -alpha)
    inv_x = np.linalg.inv(matrix1d_x)
    inv_y = np.linalg.inv(matrix1d_y)
    inv_z = np.linalg.inv(matrix1d_z)
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        for j in range(dim[1]):
            phi[i, j] = np.dot(inv_x, phi[i, j])
    for i in range(dim[0]): #then iterate through ADI method in the y direction
        for j in range(dim[2]):
            phi[i, :, j] = np.dot(inv_y, phi[i, :, j])
    for i in range(dim[1]): #finally, iterate through ADI method in the z direction
        for j in range(dim[2]):
            phi[:, i, j] = np.dot(inv_z, phi[:, i, j])
    sim.fields[0].data = phi
    
def engine_CahnAllenIMEX1D(sim):
    dt = sim.dt
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    dim = sim.get_dimensions()
    M = user_data["M"]
    W = user_data["W"]
    epsilon = user_data["epsilon"]
    alpha = M*dt*(epsilon**2)/(dx**2)
    matrix_source_term = 16*W*M
    matrix1d = implicit_matrix_1d(dim[0], 1+2*alpha, -alpha)
    phi_init = phi-16*W*M*dt*(4*phi**3 - 6*phi**2 - 2*phi)
    phi_final = np.linalg.solve(matrix1d, phi_init)
    sim.fields[0].data = phi_final
    
def functional_CahnAllen():
    print("The functional for this engine is given by the following LaTeX expressions:")
    print("$$ F = \\int_\\Omega (\\frac{\\epsilon^2}{2}|\\nabla \\phi|^2 + \\phi^2(1-\\phi)^2) d\\Omega $$")
    print("$$ \\frac{\\phi^{t+1}-\\phi^{t}}{\\Delta t} = \\frac{\\epsilon^2M}{\\Delta x^2}(\\phi_{x+1} + \\phi_{x-1} - 2\\phi_{x}) - M(4\\phi_{x}^3 -6\\phi_{x}^2 + 2\\phi_{x}) $$")
    print("Explicit: $$ \\frac{\\phi^{t+1}_x-\\phi^{t}_x}{\\Delta t} = \\frac{\\epsilon^2M}{\\Delta x^2}(\\phi_{x+1}^{t} + \\phi_{x-1}^{t} - 2\\phi_{x}^{t}) - M(4(\\phi_{x}^{t})^3 -6(\\phi_{x}^{t})^2 + 2\\phi_{x}^{t}) $$")
    print("Semi-Implicit: $$ \\frac{\\phi^{t+1}_x-\\phi^{t}_x}{\\Delta t} = \\frac{\\epsilon^2M}{\\Delta x^2}(\\phi_{x+1}^{t+1} + \\phi_{x-1}^{t+1} - 2\\phi_{x}^{t+1}) - M\\phi_{x}^{t+1}(4\\phi_{x}^{t^2} -6\\phi_{x}^{t} + 2) $$")
    
class CahnAllen(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #additional initialization code goes below
        #runs *before* tdb, thermal, fields, and boundary conditions are loaded/initialized
        if not ("M" in self.user_data):
            self.user_data["M"] = 0.1
        if not ("epsilon" in self.user_data):
            self.user_data["epsilon"] = 1.
        if not ("W" in self.user_data):
            self.user_data["W"] = 1.
        if not ("solver" in self.user_data):
            self.user_data["solver"] = "explicit"
        if not ("gmres" in self.user_data):
            self.user_data["gmres"] = False
        if not ("adi" in self.user_data):
            self.user_data["adi"] = False
            
    def init_fields(self):
        #initialization of fields code goes here
        #runs *after* tdb and thermal data is loaded/initialized
        #runs *before* boundary conditions are initialized
        phi = 0.001*np.random.random(self.dimensions) + 0.4995
        self.add_field(phi, "phi")
                        
    def just_before_simulating(self):
        super().just_before_simulating()
        #additional code to run just before beginning the simulation goes below
        #runs immediately before simulating, no manual changes permitted to changes implemented here
        
    def simulation_loop(self):
        #code to run each simulation step goes here
        solver = self.user_data["solver"]
        gmres = self.user_data["gmres"]
        adi = self.user_data["adi"]
        if (solver == "explicit"):
            engine_CahnAllenExplicit(self)
        elif (solver == "implicit"):
            if(len(dim) == 1):
                if(gmres):
                    engine_CahnAllenImplicit1D_GMRES(self)
                else:
                    engine_CahnAllenImplicit1D(self)
            elif(len(dim) == 2):
                if(gmres):
                    if(adi):
                        engine_CahnAllenImplicit2D_ADI_GMRES(self)
                    else:
                        engine_CahnAllenImplicit2D_GMRES(self)
                else:
                    engine_CahnAllenImplicit2D_ADI(self)
            elif(len(dim) == 3):
                engine_CahnAllenImplicit3D_ADI(self)
        elif (solver == "imex"):
            if(len(dim) == 1):
                engine_CahnAllenIMEX1D(self)
            else:
                print("Higher dimensional non-explicit Cahn-Allen not yet implemented!")
                return
        elif (solver == "crank-nicolson"):
            if(len(dim) == 1):
                return
            elif(len(dim) == 2):
                engine_CahnAllenCrankNicolson2D_ADI(self)
            else:
                print("Higher dimensional non-explicit Cahn-Allen not yet implemented!")
                return
        