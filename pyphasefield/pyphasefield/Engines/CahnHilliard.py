import numpy as np
from scipy.sparse.linalg import gmres
from ..field import Field

def implicit_matrix_1d(xsize, centervalue, neighborvalue, farneighborvalue):
    """
    Creates a matrix for the solution of 1d implicit or crank nickolson discretizations
    Because the exact format changes between implicit and C-N, and this method is reused 
        in 2D and 3D cases, centervalue and neighbor value must be explicitly specified
    Matrix shows periodic boundary conditions!
    """
    matrix1d = np.zeros([xsize, xsize])
    np.fill_diagonal(matrix1d, centervalue)
    matrix1d = np.roll(matrix1d, 1, 0)
    np.fill_diagonal(matrix1d, np.roll(neighborvalue, 1, 0))
    matrix1d = np.roll(matrix1d, -2, 0)
    np.fill_diagonal(matrix1d, np.roll(neighborvalue, -1, 0))
    matrix1d = np.roll(matrix1d, -1, 0)
    np.fill_diagonal(matrix1d, farneighborvalue)
    matrix1d = np.roll(matrix1d, 4, 0)
    np.fill_diagonal(matrix1d, farneighborvalue)
    matrix1d = np.roll(matrix1d, -2, 0)
    return matrix1d

def implicit_matrix_2d(ysize, xsize, centervalue, neighborvalue, farneighborvalue):
    """
    Creates a matrix for the solution of 2d implicit or crank nickolson discretizations
    Because the exact format changes between implicit and C-N, and this method is reused 
        in 3D cases, centervalue and neighbor value must be explicitly specified
    Parameter order is specified as ysize then xsize, because the dimensional order of 2d arrays is [y, x]
    Matrix shows periodic boundary conditions!
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

def engine_CahnHilliardExplicit(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    phi = sim.fields[0]
    dfdphi = -sim.epsilon**2 * phi.laplacian() + (4*phi.data**3 - 6*phi.data**2 + 2*phi.data)
    deltaphi = 0
    for i in range(len(sim.get_dimensions())):
        deltaphi += (np.roll(dfdphi, 1, i) + np.roll(dfdphi, -1, i) - 2*dfdphi)
    deltaphi /= (dx**2)
    sim.fields[0].data += sim.M*deltaphi
    
def engine_CahnHilliardImplicit1D(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    idx2 = 1/dx**2
    e2 = sim.epsilon**2
    c = sim.fields[0].data
    dim = sim.get_dimensions()
    alpha = sim.M*dt*idx2
    matrix_source_term = (4*c**2 - 6*c + 2)
    matrix1d = implicit_matrix_1d(dim[0], 1+alpha*(2*matrix_source_term+6*e2*idx2), -alpha*(matrix_source_term+4*e2*idx2), alpha*e2*idx2)
    c_final = np.linalg.solve(matrix1d, c)
    sim.fields[0].data = c_final
    
def engine_CahnHilliardImplicit1D_GMRES(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    idx2 = 1/dx**2
    e2 = sim.epsilon**2
    c = sim.fields[0].data
    dim = sim.get_dimensions()
    alpha = sim.M*dt*idx2
    matrix_source_term = (4*c**2 - 6*c + 2)
    matrix1d = implicit_matrix_1d(dim[0], 1+alpha*(2*matrix_source_term+6*e2*idx2), -alpha*(matrix_source_term+4*e2*idx2), alpha*e2*idx2)
    c_final, exitCode = gmres(matrix1d, c, atol='legacy')
    sim.fields[0].data = c_final
    
def engine_CahnHilliardCrankNicolson1D(sim):
    dt = sim._time_step_in_seconds/2
    dx = sim.get_cell_spacing()
    phi = sim.fields[0]
    dfdphi = -sim.epsilon**2 * phi.laplacian() + (4*phi.data**3 - 6*phi.data**2 + 2*phi.data)
    deltaphi = 0
    for i in range(len(sim.get_dimensions())):
        deltaphi += (np.roll(dfdphi, 1, i) + np.roll(dfdphi, -1, i) - 2*dfdphi)
    deltaphi /= (dx**2)
    sim.fields[0].data += sim.M*deltaphi
    
    idx2 = 1/dx**2
    e2 = sim.epsilon**2
    c = sim.fields[0].data
    dim = sim.get_dimensions()
    alpha = sim.M*dt*idx2
    matrix_source_term = (4*c**2 - 6*c + 2)
    matrix1d = implicit_matrix_1d(dim[0], 1+alpha*(2*matrix_source_term+6*e2*idx2), -alpha*(matrix_source_term+4*e2*idx2), alpha*e2*idx2)
    c_final = np.linalg.solve(matrix1d, c)
    sim.fields[0].data = c_final
    
def engine_CahnHilliardCrankNicolson1D_GMRES(sim):
    dt = sim._time_step_in_seconds/2
    dx = sim.get_cell_spacing()
    phi = sim.fields[0]
    dfdphi = -sim.epsilon**2 * phi.laplacian() + (4*phi.data**3 - 6*phi.data**2 + 2*phi.data)
    deltaphi = 0
    for i in range(len(sim.get_dimensions())):
        deltaphi += (np.roll(dfdphi, 1, i) + np.roll(dfdphi, -1, i) - 2*dfdphi)
    deltaphi /= (dx**2)
    sim.fields[0].data += sim.M*deltaphi
    
    idx2 = 1/dx**2
    e2 = sim.epsilon**2
    c = sim.fields[0].data
    dim = sim.get_dimensions()
    alpha = sim.M*dt*idx2
    matrix_source_term = (4*c**2 - 6*c + 2)
    matrix1d = implicit_matrix_1d(dim[0], 1+alpha*(2*matrix_source_term+6*e2*idx2), -alpha*(matrix_source_term+4*e2*idx2), alpha*e2*idx2)
    c_final, exitCode = gmres(matrix1d, c, atol='legacy')
    sim.fields[0].data = c_final
    
def engine_CahnHilliardImplicit2D_GMRES(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    alpha = sim.M*dt*(sim.epsilon**2)/(dx**2)
    dim = sim.get_dimensions()
    matrix_source_term = 16*sim.W*sim.M*dt*(4*phi**2 - 6*phi + 2)
    matrix2d = implicit_matrix_2d(dim[0], dim[1], 1+4*alpha+matrix_source_term, -alpha)
    phi_final, exitCode = gmres(matrix2d, phi.flatten(), atol='legacy')
    sim.fields[0].data = phi_final.reshape(dim)
    
def engine_CahnHilliardImplicit2D_ADI(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    alpha = sim.M*dt*(sim.epsilon**2)/(dx**2)
    dim = sim.get_dimensions()
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        matrix_source_term = 8*sim.W*sim.M*dt*(4*phi[i]**2 - 6*phi[i] + 2)
        matrix1d_x = implicit_matrix_1d(dim[1], 1+2*alpha + matrix_source_term, -alpha)
        phi[i] = np.linalg.solve(matrix1d_x, phi[i])
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        matrix_source_term = 8*sim.W*sim.M*dt*(4*phi[:,i]**2 - 6*phi[:,i] + 2)
        matrix1d_y = implicit_matrix_1d(dim[0], 1+2*alpha + matrix_source_term, -alpha)
        phi[:,i] = np.linalg.solve(matrix1d_y, phi[:,i])
    sim.fields[0].data = phi
    
def engine_CahnHilliardCrankNicolson2D_ADI(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    alpha = 0.5*sim.M*dt*(sim.epsilon**2)/(dx**2)
    dim = sim.get_dimensions()
    phi = phi*(1-2*alpha) + alpha*(np.roll(phi, 1, 1) + np.roll(phi, -1, 1))
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        matrix_source_term = 8*sim.W*sim.M*dt*(4*phi[i]**2 - 6*phi[i] + 2)
        matrix1d_x = implicit_matrix_1d(dim[1], 1+2*alpha + matrix_source_term, -alpha)
        phi[i] = np.linalg.solve(matrix1d_x, phi[i])
    phi = phi*(1-2*alpha) + alpha*(np.roll(phi, 1, 0) + np.roll(phi, -1, 0))
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        matrix_source_term = 8*sim.W*sim.M*dt*(4*phi[:,i]**2 - 6*phi[:,i] + 2)
        matrix1d_y = implicit_matrix_1d(dim[0], 1+2*alpha + matrix_source_term, -alpha)
        phi[:,i] = np.linalg.solve(matrix1d_y, phi[:,i])
    sim.fields[0].data = phi
    
def engine_CahnHilliardImplicit2D_ADI_GMRES(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    alpha = sim.M*dt*(sim.epsilon**2)/(dx**2)
    dim = sim.get_dimensions()
    for i in range(dim[0]): #iterate through ADI method in the x direction first
        matrix_source_term = 8*sim.W*sim.M*dt*(4*phi[i]**2 - 6*phi[i] + 2)
        matrix1d_x = implicit_matrix_1d(dim[1], 1+2*alpha + matrix_source_term, -alpha)
        phi[i], exitCode = gmres(matrix1d_x, phi[i], atol='legacy')
    for i in range(dim[1]): #then iterate through ADI method in the y direction
        matrix_source_term = 8*sim.W*sim.M*dt*(4*phi[:,i]**2 - 6*phi[:,i] + 2)
        matrix1d_y = implicit_matrix_1d(dim[0], 1+2*alpha + matrix_source_term, -alpha)
        phi[:,i], exitCode = gmres(matrix1d_y, phi[:,i], atol='legacy')
    sim.fields[0].data = phi
    
def engine_CahnHilliardImplicit3D_ADI(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    alpha = sim.M*dt*(sim.epsilon**2)/(dx**2)
    dim = sim.get_dimensions()
    matrix_source_term = 16*sim.W*sim.M*dt*(4*phi**2 - 6*phi + 2)/3
    matrix1d_x = implicit_matrix_1d(dim[2], 1+2*alpha+matrix_source_term, -alpha)
    matrix1d_y = implicit_matrix_1d(dim[1], 1+2*alpha+matrix_source_term, -alpha)
    matrix1d_z = implicit_matrix_1d(dim[0], 1+2*alpha+matrix_source_term, -alpha)
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
    
def engine_CahnHilliardIMEX1D(sim):
    dt = sim._time_step_in_seconds
    dx = sim.get_cell_spacing()
    phi = sim.fields[0]
    dim = sim.get_dimensions()
    alpha = sim.M*dt*(sim.epsilon**2)/(dx**2)
    matrix_source_term = 16*sim.W*sim.M
    matrix1d = implicit_matrix_1d(dim[0], 1+2*alpha, -alpha)
    phi_init = phi.data-16*sim.W*sim.M*dt*(4*phi.data**3 - 6*phi.data**2 - 2*phi.data)
    phi_final = np.linalg.solve(matrix1d, phi_init)
    sim.fields[0].data = phi_final
    
def functional_CahnHilliard():
    print("Explicit: $$ \\frac{c^{t+1}_x-c^{t}_x}{\\Delta t} = -\\frac{\\epsilon^2M}{\\Delta x^4}(c_{x+2}^{t} + c_{x-2}^{t} - 4c_{x+1}^{t} - 4c_{x-1}^{t} + 6c_{x}^{t}) + \\frac{M}{\\Delta x^2}(4(c_{x+1}^{t})^3 -6(c_{x+1}^{t})^2 + 2c_{x+1}^{t} + 4(c_{x-1}^{t})^3 -6(c_{x-1}^{t})^2 + 2c_{x-1}^{t} - 8(c_{x}^{t})^3 +12(c_{x}^{t})^2 - 4c_{x}^{t}) $$" )
    print("Semi-Implicit: $$ \\frac{c^{t+1}_x-c^{t}_x}{\\Delta t} = -\\frac{\\epsilon^2M}{\\Delta x^4}(c_{x+2}^{t+1} + c_{x-2}^{t+1} - 4c_{x+1}^{t+1} - 4c_{x-1}^{t+1} + 6c_{x}^{t+1}) + \\frac{Mc_{x+1}^{t+1}}{\\Delta x^2}(4(c_{x+1}^{t})^3 -6(c_{x+1}^{t})^2 + 2c_{x+1}^{t})  + \\frac{Mc_{x-1}^{t+1}}{\\Delta x^2}(4(c_{x-1}^{t})^3 -6(c_{x-1}^{t})^2 + 2c_{x-1}^{t}) -\\frac{2Mc_{x}^{t+1}}{\\Delta x^2}(4(c_{x}^{t})^3 -6(c_{x}^{t})^2 + 2c_{x}^{t}) $$" )

def init_CahnHilliard(sim, dim, solver="explicit", gmres=False, adi=False):
    sim.M = 0.1
    sim.epsilon = 1
    sim.W = 1
    sim.set_dimensions(dim)
    sim.set_cell_spacing(1.)
    phi = 0.001*np.random.random(dim) + 0.4995
    phi_field = Field(data=phi, name="phi", simulation=sim)
    sim.add_field(phi_field)
    if (solver == "explicit"):
        sim.set_engine(engine_CahnHilliardExplicit)
    elif (solver == "implicit"):
        if(len(dim) == 1):
            if(gmres):
                sim.set_engine(engine_CahnHilliardImplicit1D_GMRES)
            else:
                sim.set_engine(engine_CahnHilliardImplicit1D)
        elif(len(dim) == 2):
            if(gmres):
                if(adi):
                    sim.set_engine(engine_CahnHilliardImplicit2D_ADI_GMRES)
                else:
                    sim.set_engine(engine_CahnHilliardImplicit2D_GMRES)
            else:
                sim.set_engine(engine_CahnHilliardImplicit2D_ADI)
        elif(len(dim) == 3):
            sim.set_engine(engine_CahnHilliardImplicit3D_ADI)
    elif (solver == "imex"):
        if(len(dim) == 1):
            sim.set_engine(engine_CahnHilliardIMEX1D)
        else:
            print("Higher dimensional non-explicit Cahn-Allen not yet implemented!")
            return
    elif (solver == "crank-nicolson"):
        if(len(dim) == 1):
            if(gmres):
                sim.set_engine(engine_CahnHilliardCrankNicolson1D_GMRES)
            else:
                sim.set_engine(engine_CahnHilliardCrankNicolson1D)
        elif(len(dim) == 2):
            sim.set_engine(engine_CahnHilliardCrankNicolson2D_ADI)
        else:
            print("Higher dimensional non-explicit Cahn-Allen not yet implemented!")
            return