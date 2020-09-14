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
def neumann_boundary_conditions_x_kernel(out):
    startx, starty = cuda.grid(2)     
    stridex, stridey = cuda.gridsize(2) 

    for i in range(starty+startx*stridey, out.shape[1], stridey*stridex):
        for j in range(out.shape[0]):
            out[j][i][0] = out[j][i][1]
            out[j][i][out.shape[2]-1] = out[j][i][out.shape[2]-2]

@cuda.jit
def neumann_boundary_conditions_y_kernel(out):
    startx, starty = cuda.grid(2)     
    stridex, stridey = cuda.gridsize(2) 

    for i in range(starty+startx*stridey, out.shape[2], stridey*stridex):
        for j in range(out.shape[0]):
            out[j][0][i] = out[j][1][i]
            out[j][out.shape[1]-1][i] = out[j][out.shape[1]-2][i]

@cuda.jit
def periodic_boundary_conditions_x_kernel(out):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty+startx*stridey, out.shape[1], stridey*stridex):
        for j in range(out.shape[0]):
            out[j][i][0] = out[j][i][out.shape[2]-2]
            out[j][i][out.shape[2]-1] = out[j][i][1]
            
@cuda.jit
def periodic_boundary_conditions_y_kernel(out):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty+startx*stridey, out.shape[2], stridey*stridex):
        for j in range(out.shape[0]):
            out[j][0][i] = out[j][out.shape[1]-2][i]
            out[j][out.shape[1]-1][i] = out[j][1][i]

@cuda.jit
def update_thermal_gradient_kernel(T, dTdt, dt):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    dT = dTdt*dt

    # assuming x and y inputs are same length
    for i in range(starty, T.shape[0], stridey):
        for j in range(startx, T.shape[1], stridex):
            T[i][j] += dT

@cuda.jit
def update_thermal_file_kernel(T, T0, T1, start, end, current):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    ratio_T1 = (current-start)/(end-start)
    ratio_T0 = 1-ratio_T1

    # assuming x and y inputs are same length
    for i in range(starty, T.shape[0], stridey):
        for j in range(startx, T.shape[1], stridex):
            T[i][j] = T0[i][j]*ratio_T0 + T1[i][j]*ratio_T1
            
def send_fields_to_GPU(sim):
    if not (sim.temperature is None):
        sim.temperature_gpu_device = cuda.to_device(sim.temperature.data)
    fields = []
    for i in range(len(sim.fields)):
        fields.append(sim.fields[i].data)
    fields = np.array(fields)
    sim.fields_gpu_device = cuda.to_device(fields)
    if not (sim.temperature is None):
        if(sim._temperature_type == "file"):
            sim.T0_device = cuda.to_device(sim.T0)
            sim.T1_device = cuda.to_device(sim.T1)
        
def retrieve_fields_from_GPU(sim):
    if not (sim.temperature is None):
        sim.temperature.data = sim.temperature_gpu_device.copy_to_host()
    fields = sim.fields_gpu_device.copy_to_host()
    for i in range(len(sim.fields)):
        sim.fields[i].data = fields[i]
        
def apply_boundary_conditions(sim):
    if(sim._boundary_conditions_type[0] == "neumann"):
        neumann_boundary_conditions_x_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device)
    elif(sim._boundary_conditions_type[0] == "periodic"):
        periodic_boundary_conditions_x_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device)
    if(sim._boundary_conditions_type[1] == "neumann"):
        neumann_boundary_conditions_y_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device)
    elif(sim._boundary_conditions_type[1] == "periodic"):
        periodic_boundary_conditions_y_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device)
    cuda.synchronize()
            
def update_temperature_field(sim):
    if(sim._temperature_type == "isothermal"):
        return
    elif(sim._temperature_type == "gradient"):
        update_thermal_gradient_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.temperature_gpu_device, 
                                                                                    sim._cooling_rate_Kelvin_per_second,
                                                                                    sim.get_time_step_length())
    elif(sim._temperature_type == "file"):
        current_time = sim.get_time_step_length()*sim.get_time_step_counter()
        while(current_time > sim.t_end):
            nbc = [] #will be fully True for GPU code, as all boundary conditions use boundary cells
            for i in range(len(sim.get_dimensions)):
                nbc.append(True)
            with mio.xdmf.TimeSeriesReader(sim._save_path+"/T.xdmf") as reader:
                reader.cells=[]
                sim.t_start= self.t_end
                sim.T0 = self.T1
                sim.t_index += 1
                sim.t_end, point_data1, cell_data0 = reader.read_data(self.t_index)
                sim.T1 = ppf_utils.expand_T_array(point_data1['T'], nbc)
                sim.T0_device, sim.T1_device = sim.T1_device, sim.T0_device
                sim.T1_device = cuda.to_device(sim.T1)
        update_thermal_gradient_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.temperature_gpu_device, 
                                                                                    sim.T0_device, sim.T1_device, 
                                                                                    sim.t_start, sim.t_end, current_time)
