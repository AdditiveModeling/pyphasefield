from numba import cuda
import numpy as np
from ..field import Field

@cuda.jit
def diffusion_kernel(fields, fields_out, D, dx):
    startx, starty = cuda.grid(2)      
    stridex, stridey = cuda.gridsize(2) 
    
    alpha = D/(dx*dx) #laplacian coefficient in diffusion discretization
    
    #fields[0] = c
    #fields_out[0] = c_out

    # assuming x and y inputs are same length
    for i in range(starty, fields[0].shape[0], stridey):
        for j in range(startx, fields[0].shape[1], stridex):
            fields_out[0][i][j] = fields[0][i][j]+alpha*(-4*fields[0][i][j]+fields[0][i+1][j]+fields[0][i-1][j]+fields[0][i][j+1]+fields[0][i][j-1])
            
def engine_DiffusionGPU(sim):
    cuda.synchronize()
    diffusion_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device, sim.fields_out_gpu_device, 
                                                                  sim.D, sim.get_cell_spacing())
    cuda.synchronize()
    sim.fields_gpu_device, sim.fields_out_gpu_device = sim.fields_out_gpu_device, sim.fields_gpu_device
    
def init_DiffusionGPU(sim, dim=[200, 200], cuda_blocks=(16,16), cuda_threads_per_block=(256,1)):
    sim.D = 0.1
    sim.uses_gpu = True
    for i in range(len(dim)):
        dim[i] += 2 #boundary cells
    sim.cuda_blocks = cuda_blocks
    sim.cuda_threads_per_block = cuda_threads_per_block
    sim.set_dimensions(dim)
    sim.set_cell_spacing(1.)
    sim.set_engine(engine_DiffusionGPU)
    c = np.zeros(dim)
    if(len(dim) == 1):
        length = dim[0]
        c[(length // 4):(3 * length // 4)] = 1
    elif(len(dim) == 2):
        length = dim[0]
        width = dim[1]
        c[(length // 4):(3 * length // 4),(width // 4):(3 * width // 4)] = 1
    elif(len(dim) == 3):
        length = dim[0]
        width = dim[1]
        depth = dim[2]
        c[(length // 4):(3 * length // 4),(width // 4):(3 * width // 4),(depth // 4):(3 * depth // 4)] = 1
    c_field = Field(data=c, name="c", simulation=sim)
    sim.add_field(c_field)
    out_dim = dim.copy()
    out_dim.insert(0, 1) #there is 1 field
    sim.fields_out_gpu_device = cuda.device_array(out_dim)
    