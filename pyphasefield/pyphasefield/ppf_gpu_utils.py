from numba import cuda
import numba
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import meshio as mio
from . import ppf_utils

@cuda.jit
def periodic_bc_1D_x_kernel(fields):
    startx = cuda.grid(1)     
    stridex = cuda.gridsize(1) 

    for i in range(startx, fields.shape[0], stridex):
        fields[i][0] = fields[i][fields.shape[1]-2]
        fields[i][fields.shape[1]-1] = fields[i][1]
        
@cuda.jit
def dirchlet_bc_1D_x_kernel(fields, bcarray):
    startx = cuda.grid(1)     
    stridex = cuda.gridsize(1) 
    
    for i in range(startx, fields.shape[0], stridex):
        fields[i][0] = bcarray[i][0]
        fields[i][fields.shape[1]-1] = bcarray[i][fields.shape[1]-1]
            
@cuda.jit
def neumann_bc_1D_x_kernel(fields, bcarray, dx):
    startx = cuda.grid(1)     
    stridex = cuda.gridsize(1) 
    
    for i in range(startx, fields.shape[0], stridex):
        fields[i][0] = fields[i][1] - dx*bcarray[i][0]
        fields[i][fields.shape[1]-1] = fields[i][fields.shape[1]-2] + dx*bcarray[i][fields.shape[1]-1]

@cuda.jit
def periodic_bc_2D_x_kernel(fields):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty, fields.shape[1], stridey):
        for j in range(startx, fields.shape[0], stridex):
            fields[j][i][0] = fields[j][i][fields.shape[2]-2]
            fields[j][i][fields.shape[2]-1] = fields[j][i][1]
            
@cuda.jit
def periodic_bc_2D_y_kernel(fields):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, fields.shape[2], stridex):
        for j in range(starty, fields.shape[0], stridey):
            fields[j][0][i] = fields[j][fields.shape[1]-2][i]
            fields[j][fields.shape[1]-1][i] = fields[j][1][i]
            
@cuda.jit
def dirchlet_bc_2D_x_kernel(fields, bcarray):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty, fields.shape[1], stridey):
        for j in range(startx, fields.shape[0], stridex):
            fields[j][i][0] = bcarray[j][i][0]
            fields[j][i][fields.shape[2]-1] = bcarray[j][i][fields.shape[2]-1]
            
@cuda.jit
def dirchlet_bc_2D_y_kernel(fields, bcarray):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, fields.shape[2], stridex):
        for j in range(starty, fields.shape[0], stridey):
            fields[j][0][i] = bcarray[j][0][i]
            fields[j][fields.shape[1]-1][i] = bcarray[j][fields.shape[1]-1][i]
            
@cuda.jit
def neumann_bc_2D_x_kernel(fields, bcarray, dx):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty, fields.shape[1], stridey):
        for j in range(startx, fields.shape[0], stridex):
            fields[j][i][0] = fields[j][i][1] - dx*bcarray[j][i][0]
            fields[j][i][fields.shape[2]-1] = fields[j][i][fields.shape[2]-2] + dx*bcarray[j][i][fields.shape[2]-1]
            
@cuda.jit
def neumann_bc_2D_y_kernel(fields, bcarray, dx):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, fields.shape[2], stridex):
        for j in range(starty, fields.shape[0], stridey):
            fields[j][0][i] = fields[j][1][i] - dx*bcarray[j][0][i]
            fields[j][fields.shape[1]-1][i] = fields[j][fields.shape[1]-2][i] + dx*bcarray[j][fields.shape[1]-1][i]
            
@cuda.jit
def periodic_bc_3D_x_kernel(fields):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(starty, fields.shape[2], stridey):
        for j in range(startz, fields.shape[1], stridez):
            for k in range(startx, fields.shape[0], stridex):
                fields[k][j][i][0] = fields[k][j][i][fields.shape[3]-2]
                fields[k][j][i][fields.shape[3]-1] = fields[k][j][i][1]
            
@cuda.jit
def periodic_bc_3D_y_kernel(fields):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(startx, fields.shape[3], stridex):
        for j in range(startz, fields.shape[1], stridez):
            for k in range(starty, fields.shape[0], stridey):
                fields[k][j][0][i] = fields[k][j][fields.shape[2]-2][i]
                fields[k][j][fields.shape[2]-1][i] = fields[k][j][1][i]
            
@cuda.jit
def periodic_bc_3D_z_kernel(fields):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, fields.shape[3], stridex):
        for j in range(starty, fields.shape[2], stridey):
            for k in range(startz, fields.shape[0], stridez):
                fields[k][0][j][i] = fields[k][fields.shape[1]-2][j][i]
                fields[k][fields.shape[1]-1][j][i] = fields[k][1][j][i]
                
@cuda.jit
def dirchlet_bc_3D_x_kernel(fields, bcarray):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(starty, fields.shape[2], stridey):
        for j in range(startz, fields.shape[1], stridez):
            for k in range(startx, fields.shape[0], stridex):
                fields[k][j][i][0] = bcarray[k][j][i][0]
                fields[k][j][i][fields.shape[3]-1] = bcarray[k][j][i][fields.shape[3]-1]
            
@cuda.jit
def dirchlet_bc_3D_y_kernel(fields, bcarray):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(startx, fields.shape[3], stridex):
        for j in range(startz, fields.shape[1], stridez):
            for k in range(starty, fields.shape[0], stridey):
                fields[k][j][0][i] = bcarray[k][j][0][i]
                fields[k][j][fields.shape[2]-1][i] = bcarray[k][j][fields.shape[2]-1][i]
            
@cuda.jit
def dirchlet_bc_3D_z_kernel(fields, bcarray):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, fields.shape[3], stridex):
        for j in range(starty, fields.shape[2], stridey):
            for k in range(startz, fields.shape[0], stridez):
                fields[k][0][j][i] = bcarray[k][0][j][i]
                fields[k][fields.shape[1]-1][j][i] = bcarray[k][fields.shape[1]-1][j][i]
                
@cuda.jit
def neumann_bc_3D_x_kernel(fields):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(starty, fields.shape[2], stridey):
        for j in range(startz, fields.shape[1], stridez):
            for k in range(startx, fields.shape[0], stridex):
                fields[k][j][i][0] = fields[k][j][i][1] - dx*bcarray[k][j][i][0]
                fields[k][j][i][fields.shape[3]-1] = fields[k][j][i][fields.shape[3]-2] + dx*bcarray[k][j][i][fields.shape[3]-1]
            
@cuda.jit
def neumann_bc_3D_y_kernel(fields):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(startx, fields.shape[3], stridex):
        for j in range(startz, fields.shape[1], stridez):
            for k in range(starty, fields.shape[0], stridey):
                fields[k][j][0][i] = fields[k][j][1][i] - dx*bcarray[k][j][0][i]
                fields[k][j][fields.shape[2]-1][i] = fields[k][j][fields.shape[2]-2][i] + dx*bcarray[k][j][fields.shape[2]-1][i]
            
@cuda.jit
def neumann_bc_3D_z_kernel(fields):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, fields.shape[3], stridex):
        for j in range(starty, fields.shape[2], stridey):
            for k in range(startz, fields.shape[0], stridez):
                fields[k][0][j][i] = fields[k][1][j][i] - dx*bcarray[k][0][j][i]
                fields[k][fields.shape[1]-1][j][i] = fields[k][fields.shape[1]-2][j][i] + dx*bcarray[k][fields.shape[1]-1][j][i]
                
@cuda.jit
def periodic_bc_1D_x_single_kernel(field):
    startx = cuda.grid(1)     
    stridex = cuda.gridsize(1) 

    for i in range(startx, 1, stridex):
        field[0] = field[field.shape[0]-2]
        field[field.shape[0]-1] = field[1]
        
@cuda.jit
def neumann_bc_1D_x_single_kernel(field):
    startx = cuda.grid(1)     
    stridex = cuda.gridsize(1) 
    
    for i in range(startx, 1, stridex):
        field[0] = field[1]
        field[field.shape[0]-1] = field[field.shape[0]-2]

@cuda.jit
def periodic_bc_2D_x_single_kernel(field):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty, field.shape[0], stridey):
        for j in range(startx, 1, stridex):
            field[i][0] = field[i][field.shape[1]-2]
            field[i][field.shape[1]-1] = field[i][1]
            
@cuda.jit
def periodic_bc_2D_y_single_kernel(field):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, field.shape[1], stridex):
        for j in range(starty, 1, stridey):
            field[0][i] = field[field.shape[0]-2][i]
            field[field.shape[0]-1][i] = field[1][i]
            
@cuda.jit
def neumann_bc_2D_x_single_kernel(field):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty, field.shape[0], stridey):
        for j in range(startx, 1, stridex):
            field[i][0] = field[i][1]
            field[i][field.shape[1]-1] = field[i][field.shape[1]-2]
            
@cuda.jit
def neumann_bc_2D_y_single_kernel(field):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, field.shape[1], stridex):
        for j in range(starty, 1, stridey):
            field[0][i] = field[1][i]
            field[field.shape[0]-1][i] = field[field.shape[0]-2][i]
            
@cuda.jit
def periodic_bc_3D_x_single_kernel(field):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(starty, field.shape[1], stridey):
        for j in range(startz, field.shape[0], stridez):
            for k in range(startx, 1, stridex):
                field[j][i][0] = field[j][i][field.shape[2]-2]
                field[j][i][field.shape[2]-1] = field[j][i][1]
            
@cuda.jit
def periodic_bc_3D_y_single_kernel(field):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(startx, field.shape[2], stridex):
        for j in range(startz, field.shape[0], stridez):
            for k in range(starty, 1, stridey):
                field[j][0][i] = field[j][field.shape[1]-2][i]
                field[j][field.shape[1]-1][i] = field[j][1][i]
            
@cuda.jit
def periodic_bc_3D_z_single_kernel(field):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, field.shape[2], stridex):
        for j in range(starty, field.shape[1], stridey):
            for k in range(startz, 1, stridez):
                field[0][j][i] = field[field.shape[0]-2][j][i]
                field[field.shape[0]-1][j][i] = field[1][j][i]
                
@cuda.jit
def neumann_bc_3D_x_single_kernel(field):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(starty, field.shape[1], stridey):
        for j in range(startz, field.shape[0], stridez):
            for k in range(startx, 1, stridex):
                field[j][i][0] = field[j][i][1]
                field[j][i][field.shape[2]-1] = field[j][i][field.shape[2]-2]
            
@cuda.jit
def neumann_bc_3D_y_single_kernel(field):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    for i in range(startx, field.shape[2], stridex):
        for j in range(startz, field.shape[0], stridez):
            for k in range(starty, 1, stridey):
                field[j][0][i] = field[j][1][i]
                field[j][field.shape[1]-1][i] = field[j][field.shape[1]-2][i]
            
@cuda.jit
def neumann_bc_3D_z_single_kernel(field):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(startx, field.shape[2], stridex):
        for j in range(starty, field.shape[1], stridey):
            for k in range(startz, 1, stridez):
                field[0][j][i] = field[1][j][i]
                field[field.shape[0]-1][j][i] = field[field.shape[0]-2][j][i]

@cuda.jit
def update_thermal_gradient_1D_kernel(T, dTdt, dt):
    startx = cuda.grid(1)
    stridex = cuda.gridsize(1)
    
    dT = dTdt*dt

    # assuming x and y inputs are same length
    for i in range(startx, T.shape[0], stridex):
            T[i] += dT
            
@cuda.jit
def update_thermal_gradient_2D_kernel(T, dTdt, dt):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    dT = dTdt*dt

    # assuming x and y inputs are same length
    for i in range(starty, T.shape[0], stridey):
        for j in range(startx, T.shape[1], stridex):
            T[i][j] += dT

@cuda.jit
def update_thermal_gradient_3D_kernel(T, dTdt, dt):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, startz = cuda.gridsize(3)
    
    dT = dTdt*dt

    # assuming x and y inputs are same length
    for i in range(startz, T.shape[0], stridez):
        for j in range(starty, T.shape[1], stridey):
            for k in range(startx, T.shape[2], stridex):
                T[i][j][k] += dT
            
@cuda.jit
def update_thermal_file_1D_kernel(T, T0, T1, start, end, current):
    startx = cuda.grid(1)
    stridex = cuda.gridsize(1)
    
    ratio_T1 = (current-start)/(end-start)
    ratio_T0 = 1-ratio_T1

    # assuming x and y inputs are same length
    for i in range(startx+1, T.shape[0]-1, stridex):
        T[i] = T0[i-1]*ratio_T0 + T1[i-1]*ratio_T1

@cuda.jit
def update_thermal_file_2D_kernel(T, T0, T1, start, end, current):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    ratio_T1 = (current-start)/(end-start)
    ratio_T0 = 1-ratio_T1

    # assuming x and y inputs are same length
    for i in range(starty+1, T.shape[0]-1, stridey):
        for j in range(startx+1, T.shape[1]-1, stridex):
            T[i][j] = T0[i-1][j-1]*ratio_T0 + T1[i-1][j-1]*ratio_T1
            
@cuda.jit
def update_thermal_file_3D_kernel(T, T0, T1, start, end, current):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)
    
    ratio_T1 = (current-start)/(end-start)
    ratio_T0 = 1-ratio_T1

    # assuming x and y inputs are same length
    for i in range(startz+1, T.shape[0]-1, stridez):
        for j in range(starty+1, T.shape[1]-1, stridey):
            for j in range(startx+1, T.shape[2]-1, stridex):
                T[i][j][k] = T0[i-1][j-1][k-1]*ratio_T0 + T1[i-1][j-1][k-1]*ratio_T1
            
def send_fields_to_GPU(sim):
    if not (sim.temperature is None):
        sim._temperature_gpu_device = cuda.to_device(sim.temperature.data)
        if(sim._temperature_type == "XDMF_FILE"):
            sim._t_file_gpu_devices[0] = cuda.to_device(sim._t_file_arrays[0])
            sim._t_file_gpu_devices[1] = cuda.to_device(sim._t_file_arrays[1])
    fields = []
    for i in range(len(sim.fields)):
        fields.append(sim.fields[i].data)
    fields = np.array(fields)
    sim._fields_gpu_device = cuda.to_device(fields)
    sim._fields_out_gpu_device = cuda.device_array_like(fields)
    if not (sim._num_transfer_arrays is None):
        dim = sim.dimensions.copy()
        for i in range(len(dim)):
            dim[i] += 2
        dim.insert(0, sim._num_transfer_arrays)
        sim._fields_transfer_gpu_device = cuda.device_array_like(np.zeros(dim))
    if not (sim._tdb_path is None):
        dim = sim.dimensions.copy()
        for i in range(len(dim)):
            dim[i] += 2
        dim.append(len(sim._tdb_components)+1)
        sim._tdb_ufunc_gpu_device = cuda.device_array_like(np.zeros(dim))
    sim._boundary_conditions_gpu_device = cuda.to_device(sim._boundary_conditions_array)
        
        
def retrieve_fields_from_GPU(sim):
    if not (sim.temperature is None):
        sim.temperature.data = sim._temperature_gpu_device.copy_to_host()
    fields = sim._fields_gpu_device.copy_to_host()
    for i in range(len(sim.fields)):
        sim.fields[i].data = fields[i]
        
def apply_boundary_conditions(sim):
    bc = sim._boundary_conditions_type
    if not isinstance(bc, list):
        l = []
        for i in range(len(sim.dimensions)):
            l.append(bc)
        bc = l
    if(len(sim.dimensions) == 1):
        if(bc[0] == "PERIODIC"):
            periodic_bc_1D_x_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._fields_gpu_device)
            if not(sim._temperature_gpu_device is None):
                periodic_bc_1D_x_single_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._temperature_gpu_device)
        elif(bc[0] == "DIRCHLET"):
            dirchlet_bc_1D_x_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_1D_x_single_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._temperature_gpu_device)
        elif(bc[0] == "NEUMANN"):
            neumann_bc_1D_x_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device, sim.dx)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_1D_x_single_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._temperature_gpu_device)
    
    elif(len(sim.dimensions) == 2):
        if(bc[0] == "PERIODIC"):
            periodic_bc_2D_x_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device)
            if not(sim._temperature_gpu_device is None):
                periodic_bc_2D_x_single_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device)
        elif(bc[0] == "DIRCHLET"):
            dirchlet_bc_2D_x_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_2D_x_single_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device)
        elif(bc[0] == "NEUMANN"):
            neumann_bc_2D_x_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device, sim.dx)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_2D_x_single_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device)
        if(bc[1] == "PERIODIC"):
            periodic_bc_2D_y_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device)
            if not(sim._temperature_gpu_device is None):
                periodic_bc_2D_y_single_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device)
        elif(bc[1] == "DIRCHLET"):
            dirchlet_bc_2D_y_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_2D_y_single_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device)
        elif(bc[1] == "NEUMANN"):
            neumann_bc_2D_y_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device, sim.dx)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_2D_y_single_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device)
    
    elif(len(sim.dimensions) == 3):
        if(bc[0] == "PERIODIC"):
            periodic_bc_3D_x_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device)
            if not(sim._temperature_gpu_device is None):
                periodic_bc_3D_x_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        elif(bc[0] == "DIRCHLET"):
            dirchlet_bc_3D_x_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_3D_x_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        elif(bc[0] == "NEUMANN"):
            neumann_bc_3D_x_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device, sim.dx)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_3D_x_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        if(bc[1] == "PERIODIC"):
            periodic_bc_3D_y_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device)
            if not(sim._temperature_gpu_device is None):
                periodic_bc_3D_y_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        elif(bc[1] == "DIRCHLET"):
            dirchlet_bc_3D_y_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_3D_y_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        elif(bc[1] == "NEUMANN"):
            neumann_bc_3D_y_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device, sim.dx)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_3D_y_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        if(bc[2] == "PERIODIC"):
            periodic_bc_3D_z_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device)
            if not(sim._temperature_gpu_device is None):
                periodic_bc_3D_z_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        elif(bc[2] == "DIRCHLET"):
            dirchlet_bc_3D_z_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_3D_z_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
        elif(bc[2] == "NEUMANN"):
            neumann_bc_3D_z_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._boundary_conditions_gpu_device, sim.dx)
            if not(sim._temperature_gpu_device is None):
                neumann_bc_3D_z_single_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device)
    cuda.synchronize()
            
def update_temperature_field(sim):
    if(sim._temperature_type == "ISOTHERMAL"):
        return
    elif(sim._temperature_type == "LINEAR_GRADIENT"):
        if(len(sim.dimensions) == 1):
            update_thermal_gradient_1D_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._temperature_gpu_device, 
                                                                                    sim._dTdt,
                                                                                    sim.get_time_step_length())
        elif(len(sim.dimensions) == 2):
            update_thermal_gradient_2D_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device, 
                                                                                    sim._dTdt,
                                                                                    sim.get_time_step_length())
        elif(len(sim.dimensions) == 3):
            update_thermal_gradient_3D_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device, 
                                                                                    sim._dTdt,
                                                                                    sim.get_time_step_length())
    elif(sim._temperature_type == "XDMF_FILE"):
        current_time = sim.get_time_step_length()*sim.get_time_step_counter()
        while(current_time > sim._t_file_bounds[1]):
            with mio.xdmf.TimeSeriesReader(sim._temperature_path) as reader:
                reader.cells=[]
                sim._t_file_bounds[0] = sim._t_file_bounds[1]
                sim._t_file_arrays[0] = sim._t_file_arrays[1]
                sim._t_file_index += 1
                sim._t_file_bounds[1], point_data1, cell_data0 = reader.read_data(sim._t_file_index)
                sim._t_file_arrays[1] = np.squeeze(point_data1['T'])
                sim._t_file_gpu_devices[0], sim._t_file_gpu_devices[1] = sim._t_file_gpu_devices[1], sim._t_file_gpu_devices[0]
                sim._t_file_gpu_devices[1] = cuda.to_device(sim._t_file_arrays[1])
        if(len(sim.dimensions) == 1):
            update_thermal_file_1D_kernel[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._temperature_gpu_device, 
                                                                                    sim._t_file_gpu_devices[0], sim._t_file_gpu_devices[1], 
                                                                                    sim._t_file_bounds[0], sim._t_file_bounds[1], 
                                                                                    current_time)
        elif(len(sim.dimensions) == 2):
            update_thermal_file_2D_kernel[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device, 
                                                                                    sim._t_file_gpu_devices[0], sim._t_file_gpu_devices[1], 
                                                                                    sim._t_file_bounds[0], sim._t_file_bounds[1], 
                                                                                    current_time)
        elif(len(sim.dimensions) == 3):
            update_thermal_file_3D_kernel[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device, 
                                                                                    sim._t_file_gpu_devices[0], sim._t_file_gpu_devices[1], 
                                                                                    sim._t_file_bounds[0], sim._t_file_bounds[1], 
                                                                                    current_time)
