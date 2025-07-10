from cupyx import jit
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import math
from . import ppf_utils
import pathlib
import h5py
                
"""format is:
    x is the boundary condition type from sim._ngbc or sim._t_ngbc, 0 is periodic, 1 is neumann, 2 is dirichlet
    bc[x][0] : magnitude of source data
    bc[x][1] : number representing coordinate of target data point
    bc[x][2] : magnitude of dirichlet contribution
    bc[x][3] : magnitude of neumann contribution
    
    The arrays are reproduced here for easy reference:
    bcl_serial = cuda.to_device(np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]], dtype=int))
    bcr_serial = cuda.to_device(np.array([[1, 0, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=int))
    bcl_parallel = cuda.to_device(np.array([[1, -1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]], dtype=int))
    bcr_parallel = cuda.to_device(np.array([[1, 2, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=int))
"""


@jit.rawkernel()
def boundary_kernel_1D_x(fields, bcfields, bcs, l, r, dx):
    startx = jit.grid(1)
    stridex = jit.gridsize(1)
    bc1 = l[bcs[0][0]]
    bc2 = r[bcs[0][1]]
    fi1 = (fields.shape[1]-bc1[1])%fields.shape[1]
    fi2 = (fields.shape[1]-bc2[1])%fields.shape[1]
    
    for i in range(startx, fields.shape[0], stridex):
        fields[i][fi1] = bc1[0]*fields[i][1] + bc1[2]*bcfields[i][0] - bc1[3]*dx*bcfields[i][0]
        fields[i][fi2] = bc2[0]*fields[i][fields.shape[1]-2] + bc2[2]*bcfields[i][1] + bc2[3]*dx*bcfields[i][1]

@jit.rawkernel()
def boundary_kernel_2D_y(fields, bcfields, bcs, l, r, dx):
    starty, startx = jit.grid(2)
    stridey, stridex = jit.gridsize(2)
    bc1 = l[bcs[0][0]]
    bc2 = r[bcs[0][1]]
    fi1 = (fields.shape[1]-bc1[1])%fields.shape[1]
    fi2 = (fields.shape[1]-bc2[1])%fields.shape[1]
    
    for i in range(startx, fields.shape[2], stridex):
        for j in range(starty, fields.shape[0], stridey):
            fields[j][fi1][i] = bc1[0]*fields[j][1][i] + bc1[2]*bcfields[j][0][i] - bc1[3]*dx*bcfields[j][0][i]
            fields[j][fi2][i] = bc2[0]*fields[j][fields.shape[1]-2][i] + bc2[2]*bcfields[j][1][i] + bc2[3]*dx*bcfields[j][1][i]

@jit.rawkernel()
def boundary_kernel_2D_x(fields, bcfields, bcs, l, r, dx):
    starty, startx = jit.grid(2)
    stridey, stridex = jit.gridsize(2)
    bc1 = l[bcs[1][0]]
    bc2 = r[bcs[1][1]]
    fi1 = (fields.shape[2]-bc1[1])%fields.shape[2]
    fi2 = (fields.shape[2]-bc2[1])%fields.shape[2]
    
    for i in range(starty, fields.shape[1], stridey):
        for j in range(startx, fields.shape[0], stridex):
            fields[j][i][fi1] = bc1[0]*fields[j][i][1] + bc1[2]*bcfields[j][i][0] - bc1[3]*dx*bcfields[j][i][0]
            fields[j][i][fi2] = bc2[0]*fields[j][i][fields.shape[2]-2] + bc2[2]*bcfields[j][i][1] + bc2[3]*dx*bcfields[j][i][1]

@jit.rawkernel()
def boundary_kernel_3D_z(fields, bcfields, bcs, l, r, dx):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    bc1 = l[bcs[0][0]]
    bc2 = r[bcs[0][1]]
    fi1 = (fields.shape[1]-bc1[1])%fields.shape[1]
    fi2 = (fields.shape[1]-bc2[1])%fields.shape[1]
    
    
    for i in range(starty, fields.shape[2], stridey):
        for j in range(startx, fields.shape[3], stridex):
            for k in range(startz, fields.shape[0], stridez):
                fields[k][fi1][i][j] = bc1[0]*fields[k][1][i][j] + bc1[2]*bcfields[k][0][i][j] - bc1[3]*dx*bcfields[k][0][i][j]
                fields[k][fi2][i][j] = bc2[0]*fields[k][fields.shape[1]-2][i][j] + bc2[2]*bcfields[k][1][i][j] + bc2[3]*dx*bcfields[k][1][i][j]
    

@jit.rawkernel()
def boundary_kernel_3D_y(fields, bcfields, bcs, l, r, dx):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    bc1 = l[bcs[1][0]]
    bc2 = r[bcs[1][1]]
    fi1 = (fields.shape[2]-bc1[1])%fields.shape[2]
    fi2 = (fields.shape[2]-bc2[1])%fields.shape[2]
    
    for i in range(startz, fields.shape[1], stridez):
        for j in range(startx, fields.shape[3], stridex):
            for k in range(starty, fields.shape[0], stridey):
                fields[k][i][fi1][j] = bc1[0]*fields[k][i][1][j] + bc1[2]*bcfields[k][i][0][j] - bc1[3]*dx*bcfields[k][i][0][j]
                fields[k][i][fi2][j] = bc2[0]*fields[k][i][fields.shape[2]-2][j] + bc2[2]*bcfields[k][i][1][j] + bc2[3]*dx*bcfields[k][i][1][j]

@jit.rawkernel()
def boundary_kernel_3D_x(fields, bcfields, bcs, l, r, dx):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    bc1 = l[bcs[2][0]]
    bc2 = r[bcs[2][1]]
    fi1 = (fields.shape[3]-bc1[1])%fields.shape[3]
    fi2 = (fields.shape[3]-bc2[1])%fields.shape[3]
    
    for i in range(startz, fields.shape[1], stridez):
        for j in range(starty, fields.shape[2], stridey):
            for k in range(startx, fields.shape[0], stridex):
                fields[k][i][j][fi1] = bc1[0]*fields[k][i][j][1] + bc1[2]*bcfields[k][i][j][0] - bc1[3]*dx*bcfields[k][i][j][0]
                fields[k][i][j][fi2] = bc2[0]*fields[k][i][j][fields.shape[3]-2] + bc2[2]*bcfields[k][i][j][1] + bc2[3]*dx*bcfields[k][i][j][1]

@jit.rawkernel()
def boundary_kernel_1D_single_x(field, bcfield, bcs, l, r, dx):
    startx = jit.grid(1)
    stridex = jit.gridsize(1)
    bc1 = l[bcs[0][0]]
    bc2 = r[bcs[0][1]]
    fi1 = (field.shape[0]-bc1[1])%field.shape[0]
    fi2 = (field.shape[0]-bc2[1])%field.shape[0]
    
    for i in range(startx, 1, stridex):
        field[fi1] = bc1[0]*field[1] + bc1[2]*bcfield[0] - bc1[3]*dx*bcfield[0]
        field[fi2] = bc2[0]*field[field.shape[0]-2] + bc2[2]*bcfield[1] + bc2[3]*dx*bcfield[1]


@jit.rawkernel()
def boundary_kernel_2D_single_y(field, bcfield, bcs, l, r, dx):
    starty, startx = jit.grid(2)
    stridey, stridex = jit.gridsize(2)
    bc1 = l[bcs[0][0]]
    bc2 = r[bcs[0][1]]
    fi1 = (field.shape[0]-bc1[1])%field.shape[0]
    fi2 = (field.shape[0]-bc2[1])%field.shape[0]
    
    for i in range(startx, field.shape[1], stridex):
        for j in range(starty, 1, stridey):
            field[fi1][i] = bc1[0]*field[1][i] + bc1[2]*bcfield[0][i] - bc1[3]*dx*bcfield[0][i]
            field[fi2][i] = bc2[0]*field[field.shape[0]-2][i] + bc2[2]*bcfield[1][i] + bc2[3]*dx*bcfield[1][i]

@jit.rawkernel()
def boundary_kernel_2D_single_x(field, bcfield, bcs, l, r, dx):
    starty, startx = jit.grid(2)
    stridey, stridex = jit.gridsize(2)
    bc1 = l[bcs[1][0]]
    bc2 = r[bcs[1][1]]
    fi1 = (field.shape[1]-bc1[1])%field.shape[1]
    fi2 = (field.shape[1]-bc2[1])%field.shape[1]
    
    for i in range(starty, field.shape[0], stridey):
        for j in range(startx, 1, stridex):
            field[i][fi1] = bc1[0]*field[i][1] + bc1[2]*bcfield[i][0] - bc1[3]*dx*bcfield[i][0]
            field[i][fi2] = bc2[0]*field[i][field.shape[1]-2] + bc2[2]*bcfield[i][1] + bc2[3]*dx*bcfield[i][1]
            
@jit.rawkernel()
def boundary_kernel_3D_single_z(field, bcfield, bcs, l, r, dx):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    bc1 = l[bcs[0][0]]
    bc2 = r[bcs[0][1]]
    fi1 = (field.shape[0]-bc1[1])%field.shape[0]
    fi2 = (field.shape[0]-bc2[1])%field.shape[0]
    
    for i in range(starty, field.shape[1], stridey):
        for j in range(startx, field.shape[2], stridex):
            for k in range(startz, 1, stridez):
                field[fi1][i][j] = bc1[0]*field[1][i][j] + bc1[2]*bcfield[0][i][j] - bc1[3]*dx*bcfield[0][i][j]
                field[fi2][i][j] = bc2[0]*field[field.shape[0]-2][i][j] + bc2[2]*bcfield[1][i][j] + bc2[3]*dx*bcfield[1][i][j]
                
@jit.rawkernel()
def boundary_kernel_3D_single_y(field, bcfield, bcs, l, r, dx):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    bc1 = l[bcs[1][0]]
    bc2 = r[bcs[1][1]]
    fi1 = (field.shape[1]-bc1[1])%field.shape[1]
    fi2 = (field.shape[1]-bc2[1])%field.shape[1]
    
    for i in range(startz, field.shape[0], stridez):
        for j in range(startx, field.shape[2], stridex):
            for k in range(starty, 1, stridey):
                field[i][fi1][j] = bc1[0]*field[i][1][j] + bc1[2]*bcfield[i][0][j] - bc1[3]*dx*bcfield[i][0][j]
                field[i][fi2][j] = bc2[0]*field[i][field.shape[1]-2][j] + bc2[2]*bcfield[i][1][j] + bc2[3]*dx*bcfield[i][1][j]
                
@jit.rawkernel()
def boundary_kernel_3D_single_x(field, bcfield, bcs, l, r, dx):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    bc1 = l[bcs[2][0]]
    bc2 = r[bcs[2][1]]
    fi1 = (field.shape[2]-bc1[1])%field.shape[2]
    fi2 = (field.shape[2]-bc2[1])%field.shape[2]
    
    for i in range(startz, field.shape[0], stridez):
        for j in range(starty, field.shape[1], stridey):
            for k in range(startx, 1, stridex):
                field[i][j][fi1] = bc1[0]*field[i][j][1] + bc1[2]*bcfield[i][j][0] - bc1[3]*dx*bcfield[i][j][0]
                field[i][j][fi2] = bc2[0]*field[i][j][field.shape[2]-2] + bc2[2]*bcfield[i][j][1] + bc2[3]*dx*bcfield[i][j][1]

@jit.rawkernel()
def update_thermal_gradient_1D_kernel(T, dTdt, dt):
    startx = jit.grid(1)
    stridex = jit.gridsize(1)
    
    dT = dTdt*dt

    # assuming x and y inputs are same length
    for i in range(startx, T.shape[0], stridex):
            T[i] += dT
            
@jit.rawkernel()
def update_thermal_gradient_2D_kernel(T, dTdt, dt):
    starty, startx = jit.grid(2)
    stridey, stridex = jit.gridsize(2)
    
    dT = dTdt*dt

    # assuming x and y inputs are same length
    for i in range(starty, T.shape[0], stridey):
        for j in range(startx, T.shape[1], stridex):
            T[i][j] += dT

@jit.rawkernel()
def update_thermal_gradient_3D_kernel(T, dTdt, dt):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    
    dT = dTdt*dt

    # assuming x and y inputs are same length
    for i in range(startz, T.shape[0], stridez):
        for j in range(starty, T.shape[1], stridey):
            for k in range(startx, T.shape[2], stridex):
                T[i][j][k] += dT
            
@jit.rawkernel()
def update_thermal_file_1D_kernel(T, T0, T1, start, end, current):
    startx = jit.grid(1)
    stridex = jit.gridsize(1)
    
    ratio_T1 = (current-start)/(end-start)
    ratio_T0 = 1-ratio_T1

    # assuming x and y inputs are same length
    for i in range(startx+1, T.shape[0]-1, stridex):
        T[i] = T0[i]*ratio_T0 + T1[i]*ratio_T1

@jit.rawkernel()
def update_thermal_file_2D_kernel(T, T0, T1, start, end, current):
    starty, startx = jit.grid(2)
    stridey, stridex = jit.gridsize(2)
    
    ratio_T1 = (current-start)/(end-start)
    ratio_T0 = 1-ratio_T1

    # assuming x and y inputs are same length
    for i in range(starty+1, T.shape[0]-1, stridey):
        for j in range(startx+1, T.shape[1]-1, stridex):
            T[i][j] = T0[i][j]*ratio_T0 + T1[i][j]*ratio_T1
            
@jit.rawkernel()
def update_thermal_file_3D_kernel(T, T0, T1, start, end, current):
    startz, starty, startx = jit.grid(3)
    stridez, stridey, stridex = jit.gridsize(3)
    
    ratio_T1 = (current-start)/(end-start)
    ratio_T0 = 1-ratio_T1

    # assuming x and y inputs are same length
    for i in range(startz+1, T.shape[0]-1, stridez):
        for j in range(starty+1, T.shape[1]-1, stridey):
            for k in range(startx+1, T.shape[2]-1, stridex):
                T[i][j][k] = T0[i][j][k]*ratio_T0 + T1[i][j][k]*ratio_T1
                
def create_rand_states(num_threads, seed=3446621627):
    """
    Creates an array of initial states for the random number generator.

    Args:
        num_threads (int): The total number of states to create.
        seed (int): An integer seed to ensure different state sequences.

    Returns:
        A CuPy array containing the initial states.
    """
    # Create a unique starting value for each thread and add the seed.
    # We cast the seed to uint32 to ensure the addition is done with unsigned integers.
    return cp.arange(1, num_threads + 1, dtype=cp.uint32) + cp.uint32(seed)

@jit.rawkernel(device=True)
def rand_uniform(states, i):
    y = states[i]
    if y == 0:
        y = 1
    y ^= (y << 13)
    y ^= (y >> 17)
    y ^= (y << 5)
    states[i] = y
    return (y - 1) / np.uint32(0xFFFFFFFF)

@jit.rawkernel(device=True)
def rand_normal(states, i):
    u1 = rand_uniform(states, i)
    u2 = rand_uniform(states, i)
    u1 = 1e-9 if u1 == 0.0 else u1
    log_u1 = cp.log(u1)
    PI = 3.141592653589793
    z1 = cp.sqrt(-2.0 * log_u1) * cp.cos(2.0 * PI * u2)
    return z1
                
def create_GPU_devices(sim):
    #this should be run only once per simulation, to avoid memory leaks!
    if not (sim.temperature is None):
        sim._temperature_gpu_device = cp.empty(sim.temperature.data.shape, dtype=sim._gpu_dtype)
        temperature_shape = sim._temperature_boundary_field.data.shape
        if(sim._temperature_type == "THERMAL_HISTORY_FILE"):
            sim._t_file_gpu_devices[0] = cp.empty(sim._t_file_arrays[0].shape, dtype=sim._gpu_dtype)
            sim._t_file_gpu_devices[1] = cp.empty(sim._t_file_arrays[1].shape, dtype=sim._gpu_dtype)
    field_shape = list(sim.fields[0].data.shape)
    field_shape.insert(0, len(sim.fields))
    field_shape = tuple(field_shape)
    sim._fields_gpu_device = cp.empty(field_shape, dtype=sim._gpu_dtype)
    sim._fields_out_gpu_device = cp.empty(field_shape, dtype=sim._gpu_dtype)
    if not (sim._num_transfer_arrays is None):
        t_shape = list(sim.fields[0].data.shape)
        t_shape.insert(0, sim._num_transfer_arrays)
        sim._fields_transfer_gpu_device = cp.empty(tuple(t_shape), dtype=sim._gpu_dtype)
    if not (sim._tdb_path is None):
        tdb_shape = list(sim.fields[0].data.shape)
        tdb_shape.append(len(sim._tdb_components)+1)
        sim._tdb_ufunc_gpu_device = cp.empty(tuple(tdb_shape), dtype=sim._gpu_dtype)
    for i in range(len(sim.dimensions)):
        sim._bc_subarrays[i] = cp.array(sim._bc_subarrays[i])
        sim._temperature_bc_subarrays[i] = cp.array(sim._temperature_bc_subarrays[i])
    sim._ngbc = cp.array(np.array(sim._ngbc))
    sim._t_ngbc = cp.array(np.array(sim._t_ngbc))
    sim._bcl_serial = cp.array(np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]], dtype=int))
    sim._bcr_serial = cp.array(np.array([[1, 0, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=int))
    sim._bcl_parallel = cp.array(np.array([[1, -1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]], dtype=int))
    sim._bcr_parallel = cp.array(np.array([[1, 2, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=int))
            
def send_fields_to_GPU(sim):
    #03/28/2023 - tried to ensure device arrays are reused, avoid memory leaks
    if not (sim.temperature is None):
        sim._temperature_gpu_device.set(sim.temperature.data)
        if(sim._temperature_type == "THERMAL_HISTORY_FILE"):
            sim._t_file_gpu_devices[0].set(sim._t_file_arrays[0])
            sim._t_file_gpu_devices[1].set(sim._t_file_arrays[1])
    for i in range(len(sim.fields)):
        sim._fields_gpu_device[i].set(sim.fields[i].data)
    #send slices of full boundary condition array to gpu
    
def send_fields_to_GPU_minimal(sim):
    if not (sim._temperature_gpu_device is None):
        sim._temperature_gpu_device.set(sim.temperature.data)
    for i in range(len(sim.fields)):
        sim._fields_gpu_device[i].set(sim.fields[i].data)
        
        
def retrieve_fields_from_GPU(sim):
    if not (sim.temperature is None):
         sim._temperature_gpu_device.get(out=sim.temperature.data)
    for i in range(len(sim.fields)):
        sim._fields_gpu_device[i].get(out=sim.fields[i].data)

def retrieve_fields_from_GPU_minimal(sim):
    if not (sim._temperature_gpu_device is None):
         sim._temperature_gpu_device.get(out=sim.temperature.data)
    for i in range(len(sim.fields)):
        sim._fields_gpu_device[i].get(out=sim.fields[i].data)
        
def clean_GPU(sim):
    if not (sim._temperature_gpu_device is None):
        del(sim._temperature_gpu_device)
    while(len(sim._temperature_bc_subarrays) > 0):
        del(sim._temperature_bc_subarrays[0])
    if not (sim._fields_gpu_device is None):
        del(sim._fields_gpu_device)
    if not (sim._fields_out_gpu_device is None):
        del(sim._fields_out_gpu_device)
    if not (sim._fields_transfer_gpu_device is None):
        del(sim._fields_transfer_gpu_device)
    if not (sim._tdb_ufunc_gpu_device is None):
        del(sim._tdb_ufunc_gpu_device)
    while(len(sim._bc_subarrays) > 0):
        del(sim._bc_subarrays[0])
    
def apply_parallel_bcs(sim):
    apply_boundary_conditions(sim)
        
def apply_boundary_conditions(sim):
    #TODO: apply new boundary kernels here
    #currently broken because of the change to 2N bc types, to allow for asymmetric bcs
    bc = sim._boundary_conditions_type
    l, r = sim._bcl_serial, sim._bcr_serial
    if(sim._parallel):
        l, r = sim._bcl_parallel, sim._bcr_parallel
    if(len(sim.dimensions) == 1):
        boundary_kernel_1D_x[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._fields_gpu_device, sim._bc_subarrays[0], sim._ngbc, l, r, 0.1)
        if not(sim._temperature_gpu_device is None):
            boundary_kernel_1D_single_x[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim._temperature_gpu_device, sim._temperature_bc_subarrays[0], sim._t_ngbc, l, r, sim.dx)
    
    elif(len(sim.dimensions) == 2):
        boundary_kernel_2D_y[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._bc_subarrays[0], sim._ngbc, l, r, sim.dx)
        if not(sim._temperature_gpu_device is None):
            boundary_kernel_2D_single_y[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device, sim._temperature_bc_subarrays[0], sim._t_ngbc, l, r, sim.dx)
        cp.cuda.runtime.deviceSynchronize()
        boundary_kernel_2D_x[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._bc_subarrays[1], sim._ngbc, l, r, sim.dx)
        if not(sim._temperature_gpu_device is None):
            boundary_kernel_2D_single_x[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._temperature_gpu_device, sim._temperature_bc_subarrays[1], sim._t_ngbc, l, r, sim.dx)
    elif(len(sim.dimensions) == 3):
        boundary_kernel_3D_z[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._bc_subarrays[0], sim._ngbc, l, r, sim.dx)
        if not(sim._temperature_gpu_device is None):
            boundary_kernel_3D_single_z[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device, sim._temperature_bc_subarrays[0], sim._t_ngbc, l, r, sim.dx)
        cp.cuda.runtime.deviceSynchronize()
        boundary_kernel_3D_y[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._bc_subarrays[1], sim._ngbc, l, r, sim.dx)
        if not(sim._temperature_gpu_device is None):
            boundary_kernel_3D_single_y[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device, sim._temperature_bc_subarrays[1], sim._t_ngbc, l, r, sim.dx)
        cp.cuda.runtime.deviceSynchronize()
        boundary_kernel_3D_x[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._fields_gpu_device, sim._bc_subarrays[2], sim._ngbc, l, r, sim.dx)
        if not(sim._temperature_gpu_device is None):
            boundary_kernel_3D_single_x[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim._temperature_gpu_device, sim._temperature_bc_subarrays[2], sim._t_ngbc, l, r, sim.dx)
    cp.cuda.runtime.deviceSynchronize()
            
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
    elif(sim._temperature_type == "THERMAL_HISTORY_FILE"):
        current_time = sim.get_time_step_length()*sim.get_time_step_counter()
        with h5py.File(sim._temperature_path) as f:
            times = f["times"][:]
            #assume the first time slice is less than the current time, if not, interpolate before first slice
            while(times[sim._t_file_index] < current_time):
                if(sim._t_file_index == len(times)-1):
                    break #interpolate past last time slice if necessary
                sim._t_file_index += 1
                sim._t_file_bounds[0] = sim._t_file_bounds[1]
                sim._t_file_bounds[1] = times[sim._t_file_index]
                sim._t_file_arrays[0] = sim._t_file_arrays[1]
                sim._t_file_arrays[1] = sim._build_interpolated_t_array(f, sim._t_file_index)
                sim._t_file_gpu_devices[0], sim._t_file_gpu_devices[1] = sim._t_file_gpu_devices[1], sim._t_file_gpu_devices[0]
                sim._t_file_gpu_devices[1] = cp.array(sim._t_file_arrays[1])
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
