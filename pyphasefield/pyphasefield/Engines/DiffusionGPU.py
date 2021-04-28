from numba import cuda
import numpy as np

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

@cuda.jit
def diffusion_kernel_1D(fields, fields_out, D, dx):
    startx = cuda.grid(1)      
    stridex = cuda.gridsize(1) 
    
    alpha = D/(dx*dx) #laplacian coefficient in diffusion discretization
    
    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(startx, c.shape[1], stridex):
        c_out[i] = c[i]+alpha*(-2*c[i]+c[i+1]+c[i-1])
            
@cuda.jit
def diffusion_kernel_2D(fields, fields_out, D, dx):
    startx, starty = cuda.grid(2)      
    stridex, stridey = cuda.gridsize(2) 
    
    alpha = D/(dx*dx) #laplacian coefficient in diffusion discretization
    
    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(starty, c.shape[0], stridey):
        for j in range(startx, c.shape[1], stridex):
            c_out[i][j] = c[i][j]+alpha*(-4*c[i][j]+c[i+1][j]+c[i-1][j]+c[i][j+1]+c[i][j-1])
            
@cuda.jit
def diffusion_kernel_3D(fields, fields_out, D, dx):
    startx, starty, startz = cuda.grid(3)      
    stridex, stridey, startz = cuda.gridsize(3) 
    
    alpha = D/(dx*dx) #laplacian coefficient in diffusion discretization
    
    c = fields[0]
    c_out = fields_out[0]

    # assuming x and y inputs are same length
    for i in range(startz, c.shape[0], stridez):
        for j in range(starty, c.shape[1], stridey):
            for k in range(startx, c.shape[2], stridex):
                c_out[i][j][k] = c[i][j][k]+alpha*(-4*c[i][j][k]+c[i+1][j][k]+c[i-1][j][k]+c[i][j+1][k]+c[i][j-1][k]+c[i][j][k+1]+c[i][j][k-1])
            
def engine_DiffusionGPU(sim):
    cuda.synchronize()
    if(len(sim.dimensions) == 1):
        diffusion_kernel_1D[sim._gpu_blocks_per_grid_1D, sim._gpu_threads_per_block_1D](sim.fields_gpu_device, sim.fields_out_gpu_device, 
                                                                  sim.user_data["D"], sim.get_cell_spacing())
    elif(len(sim.dimensions) == 2):
        diffusion_kernel_2D[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim.fields_gpu_device, sim.fields_out_gpu_device, 
                                                                  sim.user_data["D"], sim.get_cell_spacing())
    elif(len(sim.dimensions) == 3):
        diffusion_kernel_3D[sim._gpu_blocks_per_grid_3D, sim._gpu_threads_per_block_3D](sim.fields_gpu_device, sim.fields_out_gpu_device, 
                                                                  sim.user_data["D"], sim.get_cell_spacing())
    cuda.synchronize()
    sim.fields_gpu_device, sim.fields_out_gpu_device = sim.fields_out_gpu_device, sim.fields_gpu_device
    
class DiffusionGPU(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #additional initialization code goes below
        #runs *before* tdb, thermal, fields, and boundary conditions are loaded/initialized
        if not ("D" in self.user_data):
            self.user_data["D"] = 0.1
        self._uses_gpu = True
            
    def init_fields(self):
        #initialization of fields code goes here
        #runs *after* tdb and thermal data is loaded/initialized
        #runs *before* boundary conditions are initialized
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
        self.add_field(c, "c")
        
    def initialize_fields_and_imported_data(self):
        super().initialize_fields_and_imported_data()
        #initialization of fields/imported data goes below
        #runs *after* tdb, thermal, fields, and boundary conditions are loaded/initialized
                        
    def just_before_simulating(self):
        super().just_before_simulating()
        #additional code to run just before beginning the simulation goes below
        #runs immediately before simulating, no manual changes permitted to changes implemented here
        
    def simulation_loop(self):
        #code to run each simulation step goes here
        engine_DiffusionGPU(self)