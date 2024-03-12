import numpy as np
import sympy as sp
import symengine as se
import meshio as mio
from .field import Field
from pathlib import Path
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from tinydb import where
from . import ppf_utils
from . import parallel_utils
import time
import h5py
from scipy.interpolate import RegularGridInterpolator

#DO NOT load mpi4py automatically, can crash jupyter notebooks if it isn't enabled.
#only load if parallelism is requested
#try:
#    from mpi4py import MPI
#except:
#    pass

#attempt to load GPU utils and numba, for GPU functionality
try:
    from . import ppf_gpu_utils
    import numba
except:
    pass

#attempt to load pycalphad, for tdb functionality
try:
    import pycalphad as pyc
except:
    pass

class Simulation:
    """
    Class used by pyphasefield to store data related to a given simulation
    
    Attributes
    ----------
    
    _framework : str
        A string which defines in what computational framework the simulation will run, as implemented in the Engine
        Values can be CPU_SERIAL, CPU_PARALLEL, GPU_SERIAL, or GPU_PARALLEL
        (SERIAL implies a single device (no MPI communication), PARALLEL implies multiple devices, where MPI is required)
        (CPU runs like normal python code, GPU runs on the GPU using numba/CUDA integration, requiring numba and cudatoolkit)
    _gpu_blocks_per_grid_1D : tuple of int, length 1, default = (256)
        Defines the number of blocks of threads that CUDA will use
        Similar private attributes exist for 2D/3D, of length 2 and 3, with defaults (16, 16) and (8, 8, 8), respectively
        For advanced usage, can be modified to change how many resources CUDA uses
    _gpu_threads_per_block_1D : tuple of int, length 1, default = (256)
        Defines the number of threads per block that CUDA will use
        Similar private attributes exist for 2D/3D, of length 2 and 3, with defaults (16, 16) and (8, 8, 8), respectively
        For advanced usage, can be modified to change how many resources CUDA uses
    
    fields : list of Field
        The list containing the fields of the phase field model
        Arbitrary convention is that the order of the fields is phase(s), then orientation(s), then composition(s)
        e.g. for the NCGPU model, Simulation.fields[0] is phase, 
            Simulation.fields[1] and Simulation.fields[2] are the 2D quaternion orientation fields
            Simulation.fields[3] onwards are the composition fields, sorted in alphabetical order by element (CALPHAD convention)
    dimensions : list of int
        defines the length, width (and possibly depth) of the simulation.
        Boundary conditions will be appended to these sizes as needed. 
        e.g. Submitting dimensions of [100, 100], with default boundary conditions of one cell in width, 
            will have an overall array size of [102, 102]. The "dimensions" attribute will remain [100, 100], however.
    dx : float
        dx is the cell spacing, the distance between neighboring cells, in arbitrary units (typically m or cm)
        This is the value used in finite difference equations, often specified as (delta)x in the math
        pyphasefield uses a structured square/cubic mesh, so this is used for all dimensions
    dt : float
        dt is the time step, the length of time between simulation states, in units of seconds
        This is the value used in finite difference equations, often specified as (delta)t in the math
        Some Engines may calculate a suitable value for dt automatically, using a variation of the Courant condition
    time_step_counter : int, default = 0
        Specifies the count of how many time steps have passed in the simulation when the simulation is initialized
        Defaults to zero (assumes a new simulation, where zero time steps have passed yet)
        This parameter affects the index used for save files, and the initial value of the frozen gradient thermal field
    
    temperature : Field
        A Field containing the value of the temperature field at every cell in the simulation
        (This field will have the same value everywhere if Simulation._temperature_type == "ISOTHERMAL")
        Defined separately from the other fields, as certain standard temperature types could work for any simulation
    _temperature_type : str, optional
        A string which defines the method used for the temperature field
        Values can be ISOTHERMAL, LINEAR_GRADIENT, or THERMAL_HISTORY_FILE
        ISOTHERMAL: Single, static temperature
        FROZEN_GRADIENT: Static ("frozen") thermal gradient, thermal field is not evolved like the other fields
        ACTIVE_GRADIENT: Dynamic thermal gradient, thermal field must be simulated just like the other fields
        THERMAL_HISTORY_FILE: Static temperature field 
            defined by a HDF5 file using h5py or by a XDMF file using meshio.xdmf.TimeSeriesReader
            linearly interpolates between defined thermal slices, as it is impractical to store thermal data for every PF time step
        (Here, static means the temperature field is not simulated by phase field equations, it is an independent variable)
        If undefined, the simulation will not use/initialize a temperature field
    _temperature_path : str, optional
        If Simulation._temperature_type == "THERMAL_HISTORY_FILE", the thermal history will be read from the file at this path
        REQUIRED if Simulation._temperature_type == "THERMAL_HISTORY_FILE"
    _temperature_units : str, default = "K"
        NOTE: pyphasefield always uses kelvin under the hood! This variable is only used for plotting graphs
    _initial_T : float, optional
        Defines the initial temperature of the simulation, in the corner where array indices are 0, at time step 0, in units of K
        (Specifying the corner is important, because values will change from there in the LINEAR_GRADIENT case)
        REQUIRED if Simulation._temperature_type == "ISOTHERMAL" or Simulation._temperature_type == "LINEAR_GRADIENT"
    _dTdx : float, default = 0
        Specifies the slope of the thermal gradient in the x-direction, in units of K/m or K/cm, depending on the units of dx
        similar attributes exist for the y and z directions
    _dTdt : float, default = 0
        Specifies how the thermal field will change with time, in units of K/s
        IMPORTANT: typically phase field solidification simulations will demand cooling to model liquid -> solid, and
            cooling is a negative value for dTdt!
    
    _tdb_container : TDBContainer
        A class which stores the phases, components, and codegen functions for a particular tdb file
        Large TDB files can have large overheads for loading (> 1min), this class ensures the process needs only happen once
        This can be useful for running multiple simulations in series using the same TDB file
    _tdb_path : str
        Alternate way to define the .tdb file used, pycalphad will load the database at this path if _tdb_container is undefined
    _tdb_components : list of str, optional
        Defines the components of the .tdb file which will be used in the simulation
        e.g. if the .tdb file is a Al-Cu-Ni database, specifying ["CU", "NI"] will only use the binary Cu-Ni portion of the database
        If undefined, all components of the .tdb file are used in the simulation
    _tdb_phases : list of str, optional
        Defines the phases of the .tdb file which will be used in the simulation
            e.g. if the .tdb file is a database containing the LIQUID, FCC_A1, and BCC_A2 phases, specifying ["FCC_A1", "LIQUID"] 
                will only use those two phases from the database
        If undefined, all phases in the .tdb file are used in the simulation
        
    _save_path : str
        Path where output files, checkpoints, etc. will be saved from the simulation
    _autosave_flag : Bool, default = False
        If true, checkpoints will automatically be saved
    _autosave_rate : int
        If Simulation._autosave_flag is True, checkpoints will be saved every time this many time steps pass
    _autosave_save_images_flag : Bool, default = False
        If true, also save images of the fields whenever the autosave function happens
    
    boundary_fields : list of Field
        list equal in length to fields, containing a complementary set of fields representing the boundary conditions
        Defaults to zero, Dirichlet bcs will therefore impose a value of zero, Neumann bcs will impose a slope of zero
        Only uses values at the border of the array (convenience). Inefficient, but manageable especially with parallelism
    _boundary_conditions_type : str OR list of str
        String that defines the boundary conditions in every dimension, or list that defines the boundary conditions 
            in each dimension separately (length of list == number of dimensions), or list that defines the boundary
            conditions separately in each dimension and each side of the domain (length of list == 2*number of dimensions)
        Options for values for string(s): PERIODIC, NEUMANN, or DIRICHLET, described conceptually as follows
        PERIODIC: boundary value on left edge = value on right edge
        NEUMANN: boundary value on left edge and value on left edge have slope defined by Simulation.boundary_fields
        DIRICHLET: boundary value on left edge has value equal to that in Simulation.boundary_fields
        Will be automatically converted into the list of size 2*number of dimensions when set through set_boundary_conditions!
            DON'T set this directly!
    _neighbors : list of int
        Set based on _boundary_conditions_type when using Simulation.set_boundary_conditions
        Specifically useful for parallelism (which core is the "periodic" boundary condition?) and GPUs (GPUs can't handle str)
    user_data : dict
        Arbitrary dictionary of parameters for the specific subclass engine
        e.g. Diffusion has the following params in the dict:
            D : float
                the diffusivity used in the simulation
            solver : str
                "explicit", "implicit", or "crank-nicolson", the method for solving the finite difference equation
            adi : Bool
                for True, use the alternating direction implicit method for fast computation, if 2D/3D
            gmres : Bool
                for True, use the gmres approximation method (implemented in scipy) for the implicit matrix equations
    """
    def __init__(self, dimensions, framework=None, dx=None, dt=None, initial_time_step=0, 
                 temperature_type=None, initial_T=None, dTdx=0, dTdy=0, dTdz=0, dTdt=0, 
                 temperature_path=None, temperature_units="K",
                 tdb_container=None, tdb_path=None, tdb_components=None, tdb_phases=None, 
                 save_path=None, autosave=False, save_images=False, autosave_rate=None, 
                 boundary_conditions=None, user_data={}):
        
        #framework (cpu, gpu, parallel) specific variables
        self._framework = framework
        self._uses_gpu = False
        self._parallel = False
        self._batched_simulations = True #if false, each core will run a unique simulation. Good for running many small simulations.
        self._gpu_blocks_per_grid_1D = (256)
        self._gpu_blocks_per_grid_2D = (16, 16)
        self._gpu_blocks_per_grid_3D = (8, 8, 4)
        self._gpu_threads_per_block_1D = (256)
        self._gpu_threads_per_block_2D = (16, 16)
        self._gpu_threads_per_block_3D = (8, 4, 8)
        self._gpu_dtype = "float64"
        self._MPI_COMM_WORLD = None
        self._MPI_rank = 0
        self._MPI_size = 1
        self._MPI_array_rank = None
        self._MPI_array_size = None
        self._global_dimensions = dimensions.copy()
        self._dim_sizes = []
        self._dim_offset = [0, 0, 0] #location in the global dimensions of this subgrid, defaults to zero
        self._ghost_rows = 1
        
        #variable for determining if class needs to be re-initialized before running simulation steps
        self._begun_simulation = False
        
        #variable for determining if simulation has been initialized. Prevents simulation from running if it has not.
        self._requires_initialization = True
        
        #core variables: fields, length of space/time steps, dimensions of simulation region
        self.fields = []
        self._fields_gpu_device = None
        self._fields_out_gpu_device = None
        self._num_transfer_arrays = None
        self._fields_transfer_gpu_device = None
        self.dimensions = dimensions.copy()
        self.dx = dx
        self._dx_units = "m"
        self.dt = dt
        self.time_step_counter = initial_time_step
        
        #temperature related variables
        self.temperature = None
        self._temperature_gpu_device = None
        self._temperature_out_gpu_device = None
        self._temperature_boundary_field = None
        self._temperature_bc_subarrays = []
        self._t_ngbc = None #specifically used for ACTIVE_GRADIENT, neumann bcs may be converted to dirichlet for T
        self._temperature_type = temperature_type
        self._temperature_path = temperature_path
        self._temperature_units = temperature_units
        self._initial_T = initial_T
        self._dTdx = dTdx
        self._dTdy = dTdy
        self._dTdz = dTdz
        self._dTdt = dTdt
        self._t_file_index = None
        self._t_file_bounds = [0, 0]
        self._t_file_arrays = [None, None]
        self._t_file_gpu_devices = [None, None]
        self._t_file_units = ["K", "m"]
        self._initialized_t_file_helper_arrays = False
        
        #tdb related variables
        self._tdb_container = tdb_container #TDBContainer class, for storing TDB info across simulation instances (load times...)
        self._tdb = None
        self._tdb_path = tdb_path
        self._tdb_components = tdb_components
        self._tdb_phases = tdb_phases
        self._tdb_ufuncs = []
        self._tdb_ufunc_input_size = None
        self._tdb_ufunc_gpu_device = None
        
        #progress saving related variables
        self._save_path = save_path
        self._autosave_flag = autosave
        self._autosave_rate = autosave_rate
        self._autosave_save_images_flag = save_images
        
        #boundary condition related variables
        self.boundary_fields = []
        self._boundary_conditions_type = boundary_conditions
        self._neighbors = []
        self._ngbc = []
        self._neighbors_gpu_device = None
        self._bc_subarrays = []
        
        #previous versions erroneously had dirichlet spelled as "dirchlet", correct this just in case
        if(self._boundary_conditions_type == "DIRCHLET"):
            self._boundary_conditions_type = "DIRICHLET"
        if(isinstance(self._boundary_conditions_type, list)):
            if(isinstance(self._boundary_conditions_type[0], list)):
                for i in range(len(self._boundary_conditions_type)):
                    for j in range(len(self._boundary_conditions_type[0])):
                        if(self._boundary_conditions_type[i][j] == "DIRCHLET"):
                            self._boundary_conditions_type[i][j] = "DIRICHLET"
            else:
                for i in range(len(self._boundary_conditions_type)):
                    if(self._boundary_conditions_type[i] == "DIRCHLET"):
                        self._boundary_conditions_type[i] = "DIRICHLET"
                        
        #previous versions used XDMF_FILE for temperature type, modify to new wording
        if(self._temperature_type == "XDMF_FILE"):
            self._temperature_type = "THERMAL_HISTORY_FILE"
        
        #debug mode flag, for verbose printing to track down errors
        self._debug_mode_flag = False
        
        #arbitrary subclass-specific data container (dictionary)
        self.user_data = user_data
        
    def init_fields(self):
        """
        Initializes the Field class instances for a given simulation. Deprecated, use the initialize_engine pathway instead!
        
        Notes
        -----
        
        Exclusively used by the subclass, base Simulation class does not initialize fields!
        This and initialize_fields_and_imported_data are both inherited by the subclass, with the second requiring 
            calling super.initialize_fields_and_imported_data()
        This function is meant to initialize the number (and potentially contents) of the field
        The second is meant for actions that must be taken after everything else is initialized (setting values in the boundary 
            condition array, as an example)
        """
        pass
        
        
    def init_temperature_field(self):
        """
        Initializes the temperature Field
        
        Notes
        -----
        
        This function will automatically adjust the temperature based on Simulation.time_step_counter, as would be expected
        E.g. if you begin the simulation with Simulation.time_step_counter == 1000, that will have the same temperature field
            as if you ran the simulation for 1000 timesteps, allowing the field to evolve with LINEAR_GRADIENT or XDMF_FILE
        """
        self._t_ngbc = self._ngbc.copy()
        ndims = len(self.dimensions)
        if not(self._temperature_type is None):
            self._temperature_boundary_field = Field(data=np.zeros(self.dimensions), name="T_BCs", simulation=self)
        if(self._temperature_type == "ACTIVE_GRADIENT"):
            #set temperature boundary conditions along gradient direction to dirichlet (index 2) - a must to apply temperature gradient!
            if(self._dTdx != 0 and self._dTdy == 0 and self._dTdz == 0): #x-direction gradient
                self._t_ngbc[ndims-1][0] = 2
                self._t_ngbc[ndims-1][1] = 2
            elif(self._dTdx == 0 and self._dTdy != 0 and self._dTdz == 0): #y-direction gradient
                self._t_ngbc[ndims-2][0] = 2
                self._t_ngbc[ndims-2][1] = 2
            elif(self._dTdx == 0 and self._dTdy == 0 and self._dTdz != 0): #z-direction gradient
                self._t_ngbc[ndims-3][0] = 2
                self._t_ngbc[ndims-3][1] = 2
            elif(self._dTdx == 0 and self._dTdy == 0 and self._dTdz == 0): #no gradient, but active t field
                #allow users to use any bcs for this case. Only dirichlet will apply dTdt however!
                pass
            else: #two or more gradient directions, cannot handle this with active gradient T type
                print("Cannot use ACTIVE_GRADIENT for non-orthogonal thermal gradient, switching to FROZEN_GRADIENT")
                self._temperature_type = "FROZEN_GRADIENT"
        if(self._temperature_type is None):
            pass
        elif(self._temperature_type == "ISOTHERMAL"):
            array = np.zeros(self.dimensions)
            array += self._initial_T
            t_field = Field(data=array, simulation=self, colormap="jet", name="Temperature ("+self._temperature_units+")")
            self.temperature = t_field
        elif(self._temperature_type == "LINEAR_GRADIENT" or self._temperature_type == "FROZEN_GRADIENT"):
            self._setup_linear_gradient()
        elif(self._temperature_type == "ACTIVE_GRADIENT"):
            if(self.get_time_step_counter() == 0):
                self._setup_linear_gradient()
            if(self._dTdx != 0 and self._dTdy == 0 and self._dTdz == 0): #x-direction gradient
                index = ndims-1
                grad = self._dTdx
            elif(self._dTdx == 0 and self._dTdy != 0 and self._dTdz == 0): #y-direction gradient
                index = ndims-2
                grad = self._dTdy
            elif(self._dTdx == 0 and self._dTdy == 0 and self._dTdz != 0): #z-direction gradient
                index = ndims-3
                grad = self._dTdz
            start, end = self._generate_T_slices(index)
            self._temperature_boundary_field[start] = self._initial_T
            self._temperature_boundary_field[end] = self._initial_T + grad*self.dx*self._global_dimensions[index]
        elif(self._temperature_type == "THERMAL_HISTORY_FILE"):
            if(self._temperature_path is None):
                #default to T.hdf5 first, then to T.xdmf if that doesn't exist
                if not(Path.cwd().joinpath("T.hdf5").exists()):
                    if not(Path.cwd().joinpath("T.xdmf").exists()):
                        raise FileNotFoundError("No default thermal history file found (T.hdf5 or T.xdmf). Please specify a path.")
                    else:
                        self._temperature_path = "T.xdmf"
                        
                else:
                    self._temperature_path = "T.hdf5"
            else:
                if not(Path.cwd().joinpath(self._temperature_path).exists()):
                    raise FileNotFoundError("No thermal history file found at the path specified!")
            #add code to handle hdf5 files directly here, in addition to xdmf (which won't be explicitly removed)
            self._t_file_index = 1
            dt = self.dt
            step = self.time_step_counter
            current_time = dt*step
            if(Path(self._temperature_path).suffix == ".xdmf"):
                with mio.xdmf.TimeSeriesReader(self._temperature_path) as reader:
                    points, cells = reader.read_points_cells()
                    self._t_file_bounds[0], point_data0, cell_data0 = reader.read_data(0)
                    self._t_file_arrays[0] = np.squeeze(point_data0['T'])
                    self._t_file_bounds[1], point_data1, cell_data0 = reader.read_data(self._t_file_index)
                    self._t_file_arrays[1] = np.squeeze(point_data1['T'])
                    while(current_time > self._t_file_bounds[1]):
                        self._t_file_bounds[0] = self._t_file_bounds[1]
                        self._t_file_arrays[0] = self._t_file_arrays[1]
                        self._t_file_index += 1
                        self._t_file_bounds[1], point_data1, cell_data0 = reader.read_data(self.t_file_index)
                        self._t_file_arrays[1] = np.squeeze(point_data1['T'])
            elif(Path(self._temperature_path).suffix == ".hdf5"):
                with h5py.File(self._temperature_path) as f:
                    times = f["times"][:]
                    #assume the first time slice is less than the current time, if not, interpolate before first slice
                    while(times[self._t_file_index] < current_time):
                        if(self.t_file_index == len(times)-1):
                            break #interpolate past last time slice if necessary
                        self._t_file_index += 1
                    self._t_file_bounds[0] = times[self._t_file_index-1]
                    self._t_file_bounds[1] = times[self._t_file_index]
                    self._t_file_arrays[0] = self._build_interpolated_t_array(f, self._t_file_index-1)
                    self._t_file_arrays[1] = self._build_interpolated_t_array(f, self._t_file_index)
            else:
                raise ValueError("Extension must be .hdf5 or .xdmf")
            array = self._t_file_arrays[0]*(self._t_file_bounds[1] - current_time)/(self._t_file_bounds[1]-self._t_file_bounds[0]) + self._t_file_arrays[1]*(current_time-self._t_file_bounds[0])/(self._t_file_bounds[1]-self._t_file_bounds[0])
            t_field = Field(data=array, simulation=self, colormap="jet", name="Temperature ("+self._temperature_units+")")
            self.temperature = t_field
                
    def _build_interpolated_t_array(self, f, index):
        if not(self._initialized_t_file_helper_arrays):
            self._build_t_file_helper_arrays() #creates self._t_interpolation_points just once
            print("test")
            self._initialized_t_file_helper_arrays = True
        dims_F = f["gridsize_F"][:]
        array = f["data"][:][index]
        if(len(dims_F) == 2):
            x = (np.arange(array.shape[1], dtype=float)*dims_F[0])
            y = (np.arange(array.shape[0], dtype=float)*dims_F[1])
            interp = RegularGridInterpolator([y,x], array, bounds_error=False, fill_value=None)
        elif(len(dims_F) == 3):
            x = (np.arange(array.shape[2], dtype=float)*dims_F[0])
            y = (np.arange(array.shape[1], dtype=float)*dims_F[1])
            z = (np.arange(array.shape[0], dtype=float)*dims_F[2])
            interp = RegularGridInterpolator([z,y,x], array, bounds_error=False, fill_value=None)
        interp_array = interp(self._t_interpolation_points, method="linear").reshape(*self.dimensions)
        return interp_array
            
        
    def _build_t_file_helper_arrays(self):
        aranges = []
        for i in range(len(self.dimensions)):
            aranges.append((np.arange(self.dimensions[i], dtype=float)+self._dim_offset[i])*self.dx)
        grid = np.meshgrid(*aranges, indexing='ij')
        print(len(grid))
        if(len(grid) == 2):
            self._t_interpolation_points = np.array([grid[0].ravel(), grid[1].ravel()]).T
        elif(len(grid) == 3):
            self._t_interpolation_points = np.array([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]).T
                
    def _generate_T_slices(self, index):
        ndims = len(self.dimensions)
        start = []
        end = []
        for i in range(ndims):
            if(i == index):
                start.append(slice(0, 1))
                end.append(self._global_dimensions[index], self._global_dimensions[index]+1)
            else:
                start.append(slice(None))
                end.append(slice(None))
        return start, end
                
    def _setup_linear_gradient(self):
        ndims = len(self.dimensions)
        array = np.zeros(self.dimensions)
        array += self._initial_T
        if(self._parallel):
            array += self._dTdx*self.dx*self._dim_offset[ndims-1]
            if(len(self.dimensions) > 1):
                array += self._dTdy*self.dx*self._dim_offset[ndims-2]
            if(len(self.dimensions) > 2):
                array += self._dTdz*self.dx*self._dim_offset[ndims-3]
        if not ((self._dTdx is None) or (self._dTdx == 0)):
            x_t_array = self.dx*self._dTdx*np.arange(array.shape[ndims-1])
            array += x_t_array
        if(len(self.dimensions) > 1):
            if not ((self._dTdy is None) or (self._dTdy == 0)):
                y_t_array = self.dx*self._dTdy*np.arange(array.shape[ndims-2])
                y_t_array = np.expand_dims(y_t_array, axis=1)
                array += y_t_array
        if(len(self.dimensions) > 2):
            if not ((self._dTdz is None) or (self._dTdz == 0)):
                z_t_array = self.dx*self._dTdz*np.arange(array.shape[ndims-3])
                z_t_array = np.expand_dims(z_t_array, axis=1)
                z_t_array = np.expand_dims(z_t_array, axis=2)
                array += z_t_array
        array += self.time_step_counter*self._dTdt
        t_field = Field(data=array, simulation=self, colormap="jet", name="Temperature ("+self._temperature_units+")")
        self.temperature = t_field
        
    def init_tdb_params(self):
        """
        Initializes the .tdb file, and associated codegen functions to compute the thermodynamics
        
        Notes
        -----
        
        The function will load from a given TDBContainer first. Only if one isn't given, will it try and load from the tdb_path.
        Order of phases and compositions is *always* alphabetical.
        Can be overridden (call super.init_tdb_params() first!) to define model-specific TDB functionality
        
        """
        if not(self._tdb_container is None):
            self._tdb = self._tdb_container._tdb
            self._tdb_path = self._tdb_container._tdb_path
            self._tdb_phases = self._tdb_container._tdb_phases
            self._tdb_components = self._tdb_container._tdb_components
            if(self._framework == "CPU_SERIAL" or self._framework == "CPU_PARALLEL"):
                self._tdb_ufuncs = self._tdb_container._tdb_cpu_ufuncs
            else:
                self._tdb_ufuncs = self._tdb_container._tdb_gpu_ufuncs
            return
        if(self._tdb_path is None):
            return
        #if tdb_path is specified, pycalphad *must* be installed
        if not ppf_utils.successfully_imported_pycalphad():
            raise ImportError
        import pycalphad as pyc
        self._tdb = pyc.Database(self._tdb_path)
        if self._tdb_phases is None:
            self._tdb_phases = list(self._tdb.phases)
        if self._tdb_components is None:
            self._tdb_components = list(self._tdb.elements)
        self._tdb_phases.sort()
        self._tdb_components.sort()
        for k in range(len(self._tdb_phases)):
            if(self._framework == "CPU_SERIAL" or self._framework == "CPU_PARALLEL"):
                #use numpy for CPUs
                sp_ufunc_numpy = ppf_utils.create_sympy_ufunc_from_tdb(self._tdb, self._tdb_phases[k], self._tdb_components, 'numpy')
                self._tdb_ufuncs.append(sp_ufunc_numpy)
            else: 
                #use numba for GPUs
                sp_ufunc_math = ppf_utils.create_sympy_ufunc_from_tdb(self._tdb, self._tdb_phases[k], self._tdb_components, 'math')
                nb_ufunc = nb_ufunc = ppf_utils.create_numba_ufunc_from_sympy(sp_ufunc_math)
                self._tdb_ufuncs.append(nb_ufunc)
                
    def _initialize_parallelism(self):
        #trying to import MPI *WILL* break things if MPI is not supported (even in a try-except block!)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self._MPI_COMM_WORLD = comm
        self._MPI_rank = comm.Get_rank()
        self._MPI_size = comm.Get_size()
        if(self._MPI_size > 1 and len(self.dimensions) > 1):
            #if you are here, assume MPI is functional AND there are multiple cores, AND parallelism is wanted
            #parallel 1D simulations aren't implemented yet... dunno why you'd need parallelism for 1D?
            self._parallel = True
            if(len(self.dimensions) == 2):
                dims0, dims1 = parallel_utils.region2d(self.dimensions, self._MPI_size)
                self._dim_sizes.append(dims0)
                self._dim_sizes.append(dims1)
                self._MPI_array_size = [len(dims0), len(dims1)]
                self._MPI_array_rank = []
                self._MPI_array_rank.append(self._MPI_rank//len(dims1))
                self._MPI_array_rank.append(self._MPI_rank%len(dims1))

                #C ordering, [y,x]
                self._dim_offset = []
                offset_y = 0
                offset_x = 0
                for i in range(self._MPI_array_rank[0]):
                    offset_y += dims0[i]
                self._dim_offset.append(offset_y)
                for i in range(self._MPI_array_rank[1]):
                    offset_x += dims1[i]
                self._dim_offset.append(offset_x)
                self.dimensions[0] = dims0[self._MPI_array_rank[0]]
                self.dimensions[1] = dims1[self._MPI_array_rank[1]]
            elif(len(self.dimensions) == 3):
                dims0, dims1, dims2 = parallel_utils.region3d(self.dimensions, self._MPI_size)
                self._dim_sizes.append(dims0)
                self._dim_sizes.append(dims1)
                self._dim_sizes.append(dims2)
                self._MPI_array_size = [len(dims0), len(dims1), len(dims2)]
                self._MPI_array_rank = []
                self._MPI_array_rank.append(self._MPI_rank//(len(dims1)*len(dims2)))
                self._MPI_array_rank.append((self._MPI_rank%(len(dims1)*len(dims2)))//len(dims2))
                self._MPI_array_rank.append(self._MPI_rank%len(dims2))

                #C ordering, [z,y,x]
                self._dim_offset = []
                offset_z = 0
                offset_y = 0
                offset_x = 0
                for i in range(self._MPI_array_rank[0]):
                    offset_z += dims0[i]
                self._dim_offset.append(offset_z)
                for i in range(self._MPI_array_rank[1]):
                    offset_y += dims1[i]
                self._dim_offset.append(offset_y)
                for i in range(self._MPI_array_rank[2]):
                    offset_x += dims2[i]
                self._dim_offset.append(offset_x)
                self.dimensions[0] = dims0[self._MPI_array_rank[0]]
                self.dimensions[1] = dims1[self._MPI_array_rank[1]]
                self.dimensions[2] = dims2[self._MPI_array_rank[2]]
                
    def _create_neighbors_list(self):
        """
        Creates a list of integers representing the boundary conditions
        Values:
            -2   - Dirichlet BCs, border cells are set according to the value in the boundary condition field
            -1   - Neumann BCs, border cells are set according to the slope defined by the BC field
            0    - Periodic BCs (if serial), border cells are set according to the value of the cell on the opposite side
            >= 0 - Parallel BCs (if parallel), border cells are set similar to periodic BCs, but from the specified MPI rank
        Uses the inverse of this (min value = 0) as indices for the boundary condition kernels
        """
        self._neighbors = []
        self._ngbc = [] #neighbor gpu boundary conditions, used as programmatic indices in the boundary condition kernel
        if(self._parallel):
            offset_quantity = self._MPI_size
            for i in range(len(self.dimensions)):
                self._neighbors.append([])
                self._ngbc.append([])
                offset_quantity //= self._MPI_array_size[i]
                if(self._MPI_array_rank[i] == 0):
                    index = self._get_bc_index(2*i)
                    self._neighbors[i].append(index)
                    self._ngbc[i].append(max(-index, 0))
                else:
                    self._neighbors[i].append(self._MPI_rank-offset_quantity)
                    self._ngbc[i].append(0)
                if(self._MPI_array_rank[i] == self._MPI_array_size[i]-1):
                    index = self._get_bc_index(2*i+1)
                    self._neighbors[i].append(index)
                    self._ngbc[i].append(max(-index, 0))
                else:
                    self._neighbors[i].append(self._MPI_rank+offset_quantity)
                    self._ngbc[i].append(0)
        else:
            bcs = self._boundary_conditions_type
            for i in range(len(self.dimensions)):
                self._neighbors.append([])
                self._ngbc.append([])
                for j in range(2):
                    if(bcs[i][j] == "DIRICHLET"):
                        self._neighbors[i].append(-2)
                        self._ngbc[i].append(2)
                    elif(bcs[i][j] == "NEUMANN"):
                        self._neighbors[i].append(-1)
                        self._ngbc[i].append(1)
                    else:
                        self._neighbors[i].append(0)
                        self._ngbc[i].append(0)
        
    def _get_bc_index(self, index):
        bcs = self._boundary_conditions_type
        bc_type = bcs[index//2][index%2]
            
        if(bc_type == "DIRICHLET"):
            return -2
        elif(bc_type == "NEUMANN"):
            return -1
        else: #periodic, actual thinking required
            loc = self._MPI_rank
            if(index//2 == 0): 
                offset = self._MPI_size - self._MPI_size//self._MPI_array_size[0]
            elif(index//2 == 1):
                offset = self._MPI_size//self._MPI_array_size[0] - self._MPI_size//self._MPI_array_size[0]//self._MPI_array_size[1]
            else:
                offset = self._MPI_size//self._MPI_array_size[0]//self._MPI_array_size[1] - 1
            if(index%2 == 1):
                offset *= -1
            return loc+offset
        
    def _create_bc_subarrays(self):
        self._bc_subarrays = []
        self._temperature_bc_subarrays = []
        field_shape = list(self.fields[0].data.shape)
        temperature_shape = field_shape.copy()
        field_shape.insert(0, len(self.fields))
        for i in range(len(self.dimensions)):
            bc_shape = field_shape.copy()
            bc_shape[i+1] = 2
            bc_array_i = np.zeros(bc_shape)
            t_bc_shape = temperature_shape.copy()
            t_bc_shape[i] = 2
            temperature_array_i = np.zeros(t_bc_shape)
            slice_left = self.boundary_fields[0]._sbc_in[i][0]
            slice_right = self.boundary_fields[0]._sbc_in[i][1]
            for j in range(len(self.fields)):
                #must use neighbors here, otherwise parallel boundaries will not be included!
                if not((self._neighbors[i][0] > -1) and self._parallel): #excludes parallel periodic conditions
                    bc_array_i[j][slice_left] = self.boundary_fields[j].data[slice_left]
                    if not (self.temperature is None): #acceptable to use this because parallel bcs shouldn't be replaced in t_bcs
                        #only do this for now if active gradient thermal type 
                        if(self._temperature_type == "ACTIVE_GRADIENT"):
                            temperature_array_i[slice_left] = self.temperature.data[slice_left]
                if not((self._neighbors[i][1] > -1) and self._parallel):
                    bc_array_i[j][slice_right] = self.boundary_fields[j].data[slice_right]
                    if not (self.temperature is None):
                        if(self._temperature_type == "ACTIVE_GRADIENT"):
                            temperature_array_i[slice_right] = self.temperature.data[slice_right]
            for j in range(len(self.dimensions)):
                if not(i == j):
                    for k in range(2):
                        s1 = self.boundary_fields[0]._sbc_in[j][k]
                        s2 = self.boundary_fields[0]._sbc_out[j][k]
                        temperature_array_i[s1] = temperature_array_i[s2]
                        for f in range(len(self.fields)):
                            bc_array_i[f][s1] = bc_array_i[f][s2]
            self._bc_subarrays.append(bc_array_i)
            self._temperature_bc_subarrays.append(temperature_array_i)
            

    def simulate(self, number_of_timesteps):
        """
        Evolves the simulation for a specified number of timesteps
        
        Parameters
        ----------
        
        number_of_timesteps : int
        
        
        Notes
        -----
        
        If the simulation has just been initialized, it will first call just_before_simulating(), which will deal with actions which
            only need to be taken once, like sending arrays to the GPU, applying boundary conditions for the first time, checking 
            validity of data, etc.. Other actions may be taken by the individual subclass engines through inheriting the method.
        """
        if(self._begun_simulation == False):
            self._begun_simulation = True
            self.just_before_simulating()
        for i in range(number_of_timesteps):
            self._increment_time_step_counter()
            self.simulation_loop()
            self.update_temperature_field()
            if(self._uses_gpu):
                self._swap_gpu_devices()
            self.apply_boundary_conditions()
            if(self._autosave_flag):
                if self.time_step_counter % self._autosave_rate == 0:
                    if(self._uses_gpu):
                        ppf_gpu_utils.retrieve_fields_from_GPU(self)
                    self.save_simulation()
                    if(self._autosave_save_images_flag):
                        self.plot_simulation(save_images=True, show_images=False)
                        
    def simulate_debug(self, number_of_timesteps):
        """
        Evolves the simulation for a specified number of timesteps. Prints timing information
        
        Parameters
        ----------
        
        number_of_timesteps : int
        
        
        Notes
        -----
        
        If the simulation has just been initialized, it will first call just_before_simulating(), which will deal with actions which
            only need to be taken once, like sending arrays to the GPU, applying boundary conditions for the first time, checking 
            validity of data, etc.. Other actions may be taken by the individual subclass engines through inheriting the method.
        """
        if(self._begun_simulation == False):
            t0 = time.time()
            self._begun_simulation = True
            self.just_before_simulating()
            t1 = time.time()
            print("Initialization time: {:.6f}".format(t1-t0))
        for i in range(number_of_timesteps):
            self._increment_time_step_counter()
            t0 = time.time()
            self.simulation_loop()
            t1 = time.time()
            self.update_temperature_field()
            t2 = time.time()
            if(self._uses_gpu):
                self._swap_gpu_devices()
            t3 = time.time()
            self.apply_boundary_conditions()
            t4 = time.time()
            if(self._autosave_flag):
                if self.time_step_counter % self._autosave_rate == 0:
                    if(self._uses_gpu):
                        ppf_gpu_utils.retrieve_fields_from_GPU(self)
                    self.save_simulation()
                    if(self._autosave_save_images_flag):
                        self.plot_simulation(save_images=True, show_images=False)
            print("Step: {}, Sim: {:.6f}, T: {:.6f}, Swap: {:.6f}, BCs: {:.6f}, total: {:.6f}".format(self.time_step_counter, t1-t0, t2-t1, t3-t2, t4-t3, t4-t0))
                
    def initialize_fields_and_imported_data(self):
        """
        Function to initialize the simulation, called after class attributes are set. 
        
        Deprecated name, use initialize_engine() instead.
        
        Notes
        -----
        
        This and init_fields are both inherited by the subclass, with the first requiring 
            calling super.initialize_fields_and_imported_data()
        This function is meant to initialize the number (and potentially contents) of the field
        The second is meant for actions that must be taken after everything else is initialized (setting values in the boundary 
            condition array, as an example)
        """
        self.initialize_engine()
        
    def initialize_engine(self):
        """
        Function to initialize the simulation. Called after class attributes are set.
        
        Notes
        -----
        
        This function should be overridden by the subclass (calling super.initialize_engine() first!)
        Afterwards, the subclass version of this function should do all necessary initialization for the model
        e.g. Create Fields, set boundary condition array values, import additional required tdb variables, etc.
        """
        self._requires_initialization = False
        #TODO: change GPU_SERIAL/PARALLEL to just use _uses_gpu, and automatically detect if sim can run in parallel
        if(self._framework == "GPU_SERIAL" or self._framework == "GPU_PARALLEL"): 
            self._uses_gpu = True
        if(self._framework == "CPU_PARALLEL" or self._framework == "GPU_PARALLEL"):
            self._initialize_parallelism()
        self._create_neighbors_list()
        self.init_tdb_params()
        self.init_temperature_field()
        self.init_fields()
        
    def _swap_gpu_devices(self):
        #TODO: add thermal field to here if it is actively evolved like other fields
        self._fields_gpu_device, self._fields_out_gpu_device = self._fields_out_gpu_device, self._fields_gpu_device
            
    def just_before_simulating(self): 
        self._parallel_swap(init=True)
        self._create_bc_subarrays()
        if(self._uses_gpu):
            self.create_GPU_devices()
            self.send_fields_to_GPU()
        self.apply_boundary_conditions(init=True)
        if not(self._temperature_type is None):
            assert(self.temperature.data.shape == self.fields[0].data.shape)
        self._check_temperature_file_bounds()
        
    def _check_temperature_file_bounds(self):
        if(self._temperature_type == "THERMAL_HISTORY_FILE"):
            if(self._MPI_rank == 0):
                f = h5py.File(self._temperature_path)
                shape = np.flip(f["data"][0].shape)
                size = f["gridsize_F"][:]
                thermal_size = shape*size
                sim_size = np.flip(self._global_dimensions)*self.dx
                for i in range(len(sim_size)):
                    if(np.abs(sim_size[i]-thermal_size[i])/sim_size[i] > 0.2): #if dimensions are mismatched by more than 20%
                        print(f"Sim dimensions (x, y(, z)) are: {sim_size} in {self._dx_units}")
                        print(f"Thermal file dimensions are: {thermal_size} in {self._t_file_units[1]}")
                        print("Mismatch detected, consider matching the sizes more closely!")
                        break
                
    def simulation_loop(self):
        """
        This function will run once every time step. Override this in the subclass to put simulation code here!
        """
        pass
    
    def add_field(self, array, array_name, full_grid=False, colormap="GnBu"):
        """
        Creates a Field from a numpy array, and adds it to the Simulation instance
        Also adds a complementary boundary condition field
        
        Parameters
        ----------
        
        array : ndarray
            A numpy array containing the values for the field at every point. 
            Don't include boundary cells in this array, they will be added automatically
        array_name : str
            A string naming the field, used for plotting graphs of the field
        colormap : Matplotlib colormap, str, etc.. Default = "GnBu"
            The colormap used for plotting the field. Defaults to a green blue sequential colormap
        """
        field = Field(data=array, name=array_name, simulation=self, full_grid=full_grid, colormap=colormap)
        self.fields.append(field)
        #we use the Field class for bcs for parallel splitting behavior and nearly nothing else
        bc = Field(data=np.zeros(array.shape), name=array_name+"_BCs", simulation=self, full_grid=full_grid, colormap=colormap)
        self.boundary_fields.append(bc)
        
    def default_value(self, key, value):
        """
        One line helper function for setting default values in the user_data for a subclass
        
        If the value is already defined, this does nothing, otherwise sets the value of the corresponding key to a default value
        
        Parameters
        ----------
        
        key : str
            Name of the key in the user_data dictionary 
        value : any
            Arbitrary value to be used as the default value of the previously specified key
        """
        original = getattr(self, key, None)
        if(original is None):
            setattr(self, key, value)

    def update_temperature_field(self, force_cpu=False):
        if(self._uses_gpu and (not(force_cpu))):
            #force_cpu used for initialization, before sending thermal field to GPU
            ppf_gpu_utils.update_temperature_field(self)
            return
        elif(self._temperature_type is None):
            return
        elif(self._temperature_type == "ISOTHERMAL"):
            return
        elif(self._temperature_type == "LINEAR_GRADIENT" or self._temperature_type == "FROZEN_GRADIENT"):
            self.temperature.data += self._dTdt*self.dt
            return
        elif(self._temperature_type == "THERMAL_HISTORY_FILE"):
            dt = self.get_time_step_length()
            step = self.get_time_step_counter()
            current_time = dt*step
            if(Path(self._temperature_path).suffix == ".xdmf"):
                while(current_time > self._t_file_bounds[1]):
                    with mio.xdmf.TimeSeriesReader(self._temperature_path) as reader:
                        reader.cells=[]
                        self._t_file_bounds[0] = self._t_file_bounds[1]
                        self._t_file_arrays[0] = self._t_file_arrays[1]
                        self._t_file_index += 1
                        self._t_file_bounds[1], point_data1, cell_data0 = reader.read_data(self._t_file_index)
                        self._t_file_arrays[1] = np.squeeze(point_data1['T'])
            elif(Path(self._temperature_path).suffix == ".hdf5"):
                with h5py.File(self._temperature_path) as f:
                    times = f["times"][:]
                    #assume the first time slice is less than the current time, if not, interpolate before first slice
                    while(times[self._t_file_index] < current_time):
                        if(self.t_file_index == len(times)-1):
                            break #interpolate past last time slice if necessary
                        self._t_file_index += 1
                        self._t_file_bounds[0] = self._t_file_bounds[1]
                        self._t_file_bounds[1] = times[self._t_file_index]
                        self._t_file_arrays[0] = self._t_file_arrays[1]
                        self._t_file_arrays[1] = self._build_interpolated_t_array(f, self._t_file_index)
            array = self.temperature.get_cells()
            array[:] = 0
            array += self._t_file_arrays[0]*(self._t_file_bounds[1] - current_time)/(self._t_file_bounds[1]-self._t_file_bounds[0]) + self._t_file_arrays[1]*(current_time-self._t_file_bounds[0])/(self._t_file_bounds[1]-self._t_file_bounds[0])
            return
        else:
            raise ValueError("Unknown temperature profile.")

    def load_simulation(self, file_path=None, step=-1):
        """
        Loads a simulation from a .npz file. 
        
        Either a filename, step, or both must be provided.
        If no step is specified, checks filename for step #.
        If no filename is specified, a file with the specified step number is loaded from
            the _save_path.
            
        Parameters
        ----------
        
        file_path : str, optional
            Load the simulation from a *different* place than specified in Simulation._save_path 
        step : int, optional
            If a particular .npz file is not specified in file_path, load the file corresponding to this value
            E.g. step==100, would load /path/to/file/step_100.npz
        """
        #if fields list already exists, try to copy colorbar info over
        colormaps = []
        for field in self.fields:
            colormaps.append(field.colormap)
        
        # Clear fields list
        self.fields = []
        
        if file_path is None:
            file_path = self._save_path
            if self._save_path is None:
                raise ValueError("Simulation needs a path to load data from!")
                
        # Check for file path inside cwd
        if Path.cwd().joinpath(file_path).exists():
            file_path = Path.cwd().joinpath(file_path)
        # Check if file exists in the save directory
        elif Path.cwd().joinpath(self._save_path).joinpath(file_path).exists():
            file_path = Path.cwd().joinpath(self._save_path).joinpath(file_path)
        else:
            file_path = Path(file_path)
            
        npz_filetype = False
            
        if file_path.is_dir():
            if step > -1:
                file_path = file_path.joinpath("step_"+str(step)+".hdf5")
                if not(Path.cwd().joinpath(file_path).exists()):
                    file_path = file_path.parents[0].joinpath("step_"+str(step)+".npz")
                    npz_filetype = True
                if not(Path.cwd().joinpath(file_path).exists()):
                    raise FileNotFoundError("No checkpoint exists for timestep = "+str(step)+"!")
            else:
                raise ValueError("Given path is a folder, must specify a timestep!")

        #propagate new path to the save path, the parent folder is the save path
        #only does so if the save path for the simulation is not set!
        if(self._save_path is None):
            self._save_path = str(file_path.parent)
            
        #if npz filetype, use deprecated loading mechanism
        if(npz_filetype):

            # Load array
            if(self._MPI_rank == 0):
                fields_dict = np.load(file_path, allow_pickle=True)

            #if not defined, set boundary conditions for the user (in case they just want to load the data and view it)
            if(self._boundary_conditions_type is None):
                #get first array
                array = list(fields_dict.items())[0][1]
                self.dimensions = list(array.shape)
                self.set_boundary_conditions("PERIODIC")

            if(self._parallel):
                pass
            else:
                # Add arrays self.fields as Field objects
                for key, value in fields_dict.items():
                    self.add_field(value, key, full_grid=True)

                # Set dimensions of simulation
                self.dimensions = list(self.fields[0].get_cells().shape)
        else:
            
            
            if(self._parallel):
                f = h5py.File(file_path, "r", driver='mpio', comm=self._MPI_COMM_WORLD)
                _names = f.attrs["names"]
                self.dimensions = list(f["fields"].shape[1:])
                self._global_dimensions = self.dimensions.copy()
                #reinitialize parallelism to create subarray dimensions (*SHOULD* modify self.dimensions to reflect this)
                self.initialize_parallelism()
                _slice = list(self._make_global_slice(self.dimensions, self._dim_offset))
                _slice.insert(0,0)
                for i, name in enumerate(_names):
                    _slice[0] = i
                    self.add_field(f["fields"][*_slice], name, full_grid=False)
            else:
                f = h5py.File(file_path, "r")
                _names = f.attrs["names"]
                for i, name in enumerate(_names):
                    if(f["fields"].shape[0] == len(colormaps)):
                        self.add_field(f["fields"][i], name, colormap=colormaps[i], full_grid=True)
                    else:
                        self.add_field(f["fields"][i], name, full_grid=True)
                self.dimensions = list(self.fields[0].get_cells().shape)
            f.close()
            
        # Time step set from parsing file name or manually --> defaults to 0
        if step < 0:
            filename = file_path.stem
            step_start_index = filename.find('step_') + len('step_')
            if step_start_index == -1:
                self.time_step_counter = 0
            else:
                i = step_start_index
                while i < len(filename) and filename[i].isdigit():
                    i += 1
                self.time_step_counter = int(filename[step_start_index:i])
        else:
            self.time_step_counter = int(step)
        if(self._temperature_type == "LINEAR_GRADIENT"):
            self.temperature.data += self.time_step_counter*self._dTdt*self.dt
        if(self._temperature_type == "THERMAL_HISTORY_FILE"):
            self.update_temperature_field(force_cpu=True)
        self._begun_simulation = False
        
        #if npz, save as hdf5 for next time to move away from npz
        if(npz_filetype):
            self.save_simulation()
        return 0
    
    def _collate_fields(self):
        #warning - this function will store the entire field data on a single core
        #if the simulation is *very* large, it could run out of memory
        #only used for saving checkpoints and plotting images from a *parallel* simulation
        
        global_fields = []
        for i in range(len(self.fields)):
            if(self._MPI_rank == 0):
                #leader - collect and return global_field list
                array = np.zeros(self._global_dimensions)
                _slice = self._make_global_slice(self.fields[i].get_cells().shape, self._dim_offset)
                array[_slice] = self.fields[i].get_cells()
                for j in range(1, self._MPI_size):
                    array_rank = []
                    rank = j
                    mod = self._MPI_size
                    for k in range(len(self._MPI_array_size)):
                        mod //= self._MPI_array_size[k]
                        array_rank.append(rank // mod)
                        rank = rank%mod
                    offset = []
                    shape = []
                    for k in range(len(array_rank)):
                        shape.append(self._dim_sizes[k][array_rank[k]])
                        _o = 0
                        for l in range(array_rank[k]):
                            _o += self._dim_sizes[k][l]
                        offset.append(_o)
                    array_recv = np.empty(shape).flatten()
                    self._MPI_COMM_WORLD.Recv(array_recv, source=j, tag=j)
                    _slice = self._make_global_slice(shape, offset)
                    array[_slice] = array_recv.reshape(shape)
                global_fields.append(Field(data=array, name=self.fields[i].name, simulation=self, colormap=self.fields[i].colormap))
            else:
                #worker - send field to leader, do NOT do anything in save_sim or plot_sim!
                array_send = self.fields[i].get_cells().flatten()
                self._MPI_COMM_WORLD.Send(array_send, dest=0, tag=self._MPI_rank)
        #empty for rank>0, contains complete fields for rank==0
        return global_fields
    
    def _disperse_fields(self):
        #warning - this function begins with the entire array on a single core
        #if the simulation is *very* large, it could run out of memory
        #only used for loading checkpoints in parallel simulations
        pass
                
    def _make_global_slice(self, shape, offset):
        _slice = []
        for i in range(len(shape)):
            start = offset[i]
            stop = offset[i]+shape[i]
            _slice.append(slice(start, stop))
        return tuple(_slice)
                
    
    def save_simulation(self):
        """
        Saves all fields in a .hdf5 in either the user-specified save path or a default path. 
        
        Step number is saved in the file name.
        TODO: save data for simulation instance in header file
        """
        if(self._uses_gpu and self._begun_simulation):
            self.retrieve_fields_from_GPU()
        
        # Save array with path
        if not self._save_path:
            #if save path is not defined, do not save, just return
            print("self._save_path not defined, aborting save!")
            return
        else:
            save_loc = Path(self._save_path)
        save_loc.mkdir(parents=True, exist_ok=True)
        
        _save_file_path = str(save_loc) + "/step_" + str(self.time_step_counter)+".hdf5"
        if(self._parallel):
            f = h5py.File(_save_file_path, "w", driver='mpio', comm=self._MPI_COMM_WORLD)
        else:
            f = h5py.File(_save_file_path, "w")
        fields_shape = list(self.fields[0].get_cells().shape)
        fields_shape.insert(0, len(self.fields))
        dset = f.create_dataset("fields", tuple(fields_shape), dtype='f')
        _slice = list(self._make_global_slice(self.fields[0].get_cells().shape, self._dim_offset))
        _slice.insert(0, 0)
        _names = []
        for i in range(len(self.fields)):
            _slice[0] = i
            dset[*_slice] = self.fields[i].get_cells()
            _names.append(self.fields[i].name)
        f.attrs["names"] = _names
        f.close()
    
    def save_images(self, fields=None, interpolation="bicubic", units="cells", size=None, norm=False):
        self.plot_simulation(fields, interpolation=interpolation, units=units, save_images=True, show_images=False, size=size, norm=norm)
    
    def plot_simulation(self, fields=None, interpolation="bicubic", units="cells", save_images=False, show_images=True, size=None, norm=False):
        """
        Plots/saves images of certain fields of the simulation
        
        Parameters
        ----------
        
        fields : list of int, optional
            Plot the fields corresponding to the indices in this list. If unspecified, plot all fields
        interpolation : str, default = "bicubic"
            Matplotlib string corresponding to an interpolation scheme. Other options are "nearest", "bilinear", etc.
        units : str, default = "cells"
            Units for the plots. Can be "cells", "m", or "cm". Plot assumes units for Simulation.dx are in m.
        save_images : Bool, default = False
            If true, save .png images of the fields
        show_images : Bool, default = True
            If true, plot images inline (jupyter notebook!)
        size : list of int, optional
            If defined, plot images of the defined size (matplotlib convention, inches I believe...)
        norm : Bool, default = False
            If true, plot the first field with a PowerNorm. Useful for seeing grain boundaries, as the first field is typically phase
        """
        if(self._uses_gpu and self._begun_simulation):
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
        _fields = self.fields
        if(self._parallel):
            _fields = self._collate_fields()
        if(len(_fields) == 0):
            #non rank 0 processes, go home
            return
        if fields is None:
            fields = range(len(_fields))
        if(len(_fields[0].get_cells().shape) == 1): #1d plot
            x = np.arange(0, len(_fields[0].get_cells()))
            for i in fields:
                plt.plot(x, _fields[i].get_cells())
            if(show_images):
                plt.show()
            else:
                plt.close()
        elif((len(_fields[0].get_cells()) == 1) or (len(_fields[0].get_cells()[0]) == 1)): #effective 1d plot, 2d but one dimension = 1
            x = np.arange(0, len(_fields[0].get_cells().flatten()))
            legend = []
            for i in fields:
                plt.plot(x, _fields[i].get_cells().flatten())
                legend.append(_fields[i].name)
            plt.legend(legend)
            if(show_images):
                plt.show()
            else:
                plt.close()
        elif(len(self.dimensions) == 3): #3D plot, just do an isosurface of the first field (usually phi) for now...
            points = np.argwhere((self.fields[0].get_cells() > 0.45) & (self.fields[0].get_cells() < 0.55)).T
            fig = plt.figure(figsize=(12,7))
            ax = fig.add_subplot(projection='3d')
            phi = self.fields[0].data[points[0], points[1], points[2]]
            pxp = self.fields[0].data[points[0], points[1], points[2]+1]-phi
            pyp = self.fields[0].data[points[0], points[1]+1, points[2]]-phi
            pzp = self.fields[0].data[points[0]+1, points[1], points[2]]-phi
            cs = []
            try:
                minx = min(pxp)
                maxx = max(pxp)
                miny = min(pyp)
                maxy = max(pyp)
                minz = min(pzp)
                maxz = max(pzp)
                for i in range(len(phi)):
                    r = (pxp[i]-minx)/(maxx-minx)
                    g = (pyp[i]-miny)/(maxy-miny)
                    b = (pzp[i]-minz)/(maxz-minz)
                    cs.append([r,g,b])
                img = ax.scatter(points[2], points[1], points[0], c=cs)
            except:
                #no points to show
                pass

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.show()
        else:
            for i in fields:
                if(units == "cells"):
                    extent = [0, _fields[i].get_cells().shape[1], 0, _fields[i].get_cells().shape[0]]
                elif(units == "cm"):
                    extent = [0, _fields[i].get_cells().shape[1]*self.get_cell_spacing()*100., 0, _fields[i].get_cells().shape[0]*self.get_cell_spacing()]
                elif(units == "m"):
                    extent = [0, _fields[i].get_cells().shape[1]*self.get_cell_spacing(), 0, _fields[i].get_cells().shape[0]*self.get_cell_spacing()/100.]
                if not (size is None):
                    plt.figure(figsize=size)
                if(norm):
                    if(i == 0):
                        plt.imshow(_fields[i].get_cells(), interpolation=interpolation, cmap=_fields[i].colormap, extent=extent, norm=PowerNorm(10, vmin=0, vmax=1))
                    else:
                        plt.imshow(_fields[i].get_cells(), interpolation=interpolation, cmap=_fields[i].colormap, extent=extent)
                else:
                    plt.imshow(_fields[i].get_cells(), interpolation=interpolation, cmap=_fields[i].colormap, extent=extent)
                plt.title(_fields[i].name)
                plt.colorbar()
                if(units == "cm"):
                    plt.xlabel("cm")
                    plt.ylabel("cm")
                elif(units == "m"):
                    plt.xlabel("m")
                    plt.ylabel("m")
                if(save_images):
                    #make sure parent path exists
                    if not self._save_path:
                        #if save path is not defined, do not save, just return
                        print("self._save_path not defined, aborting saving images!")
                        save_img = False
                    else:
                        save_loc = Path(self._save_path)
                        save_img = True
                    if(save_img):
                        save_loc.mkdir(parents=True, exist_ok=True)
                        plt.savefig(self._save_path+"/"+_fields[i].name+"_"+str(self.get_time_step_counter())+".png")
                if(show_images):
                    plt.show()
                else:
                    plt.close()
                

    def set_dimensions(self, dimensions):
        self.dimensions = dimensions
    def get_dimensions(self):
        return self.dimensions
    
    def set_framework(self, framework):
        self._framework = framework
    def get_framework(self):
        return self._framework
    
    def set_dx(self, dx):
        self.dx = dx
    def set_cell_spacing(self, dx):
        """See set_dx()"""
        self.dx = dx
    def get_dx(self):
        return self.dx
    def get_cell_spacing(self):
        """See get_dx()"""
        return self.dx
    
    """self.dt: length of a single timestep"""
    def set_dt(self, dt):
        self.dt = dt
    def set_time_step_length(self, dt):
        """See set_dt()"""
        self.dt = dt
    def get_dt(self):
        return self.dt
    def get_time_step_length(self):
        """See get_dt()"""
        return self.dt
    
    """self.time_step_counter: number of timesteps that have passed for a given simulation"""
    def set_time_step_counter(self, time_step_counter):
        self.time_step_counter = time_step_counter
    def get_time_step_counter(self):
        return self.time_step_counter
    def _increment_time_step_counter(self):
        """Increments the time step counter. The simulation class handles this, do not increment it yourself"""
        self.time_step_counter += 1
    
    def set_temperature_type(self, temperature_type):
        self._temperature_type = temperature_type
        if(self._temperature_type == "XDMF_FILE"):
            self._temperature_type = "THERMAL_HISTORY_FILE"

    def set_temperature_initial_T(self, initial_T):
        self._initial_T = initial_T

    #these will default to 0 if passed "None"
    def set_temperature_dTdx(self, dTdx):
        self._dTdx = dTdx
        if(dTdx is None):
            self._dTdx = 0.
    def set_temperature_dTdy(self, dTdy):
        self._dTdy = dTdy
        if(dTdy is None):
            self._dTdy = 0.
    def set_temperature_dTdz(self, dTdz):
        self._dTdz = dTdz
        if(dTdz is None):
            self._dTdz = 0.
    def set_temperature_dTdt(self, dTdt):
        self._dTdt = dTdt
        if(dTdt is None):
            self._dTdt = 0.

    def set_temperature_path(self, temperature_path):
        self._temperature_path = temperature_path
    def set_temperature_units(self, temperature_units):
        self._temperature_units = temperature_units

    def set_tdb_container(self, tdb_container):
        self._tdb_container = tdb_container
    def set_tdb_path(self, tdb_path):
        self._tdb_path = tdb_path
    def set_tdb_phases(self, tdb_phases):
        self._tdb_phases = tdb_phases
    def set_tdb_components(self, tdb_components):
        self._tdb_components = tdb_components
    
    def set_save_path(self, save_path):
        self._save_path = save_path
    def set_autosave_flag(self, autosave_flag):
        self._autosave_flag = autosave_flag
    def set_autosave_save_images_flag(self, autosave_save_images_flag):
        self._autosave_save_images_flag = autosave_save_images_flag
    def set_autosave_rate(self, autosave_rate):
        self._autosave_rate = autosave_rate
        
    def set_boundary_conditions(self, boundary_conditions_type):
        """
        Sets self._boundary_conditions_type according to the input
        Produces a 2D list:
            first axis is the number of dimensions of the simulation
            second axis has length 2, for left and right bcs
        Valid inputs (and resulting output in 3D):
            Single value: "PERIODIC"
                output: [["PERIODIC","PERIODIC"],["PERIODIC","PERIODIC"],["PERIODIC","PERIODIC"]]
            One value per dim: ["PERIODIC", "NEUMANN", "DIRICHLET"]
                output: [["PERIODIC","PERIODIC"],["NEUMANN","NEUMANN"],["DIRICHLET","DIRICHLET"]]
            Two values per dim: ["PERIODIC", "NEUMANN", "DIRICHLET", "PERIODIC", "NEUMANN", "DIRICHLET"]
                output: [["PERIODIC","NEUMANN"],["DIRICHLET","PERIODIC"],["NEUMANN","DIRICHLET"]]
            A 2D list: [["PERIODIC","NEUMANN"],["DIRICHLET","PERIODIC"],["NEUMANN","DIRICHLET"]]
                outputs itself
        """
        bc = boundary_conditions_type
        dims = len(self.dimensions)
        if not(type(boundary_conditions_type) is list): #single bc type
            self._boundary_conditions_type = []
            for i in range(dims):
                self._boundary_conditions_type.append([])
                self._boundary_conditions_type[i].append(bc)
                self._boundary_conditions_type[i].append(bc)
        elif(bc[0] is list): #double list, left and right bcs in each sublist, the "proper" format
            self._boundary_conditions_type = bc
        elif(len(bc) == len(self.dimensions)): #only 1 bc per dimension, symmetric
            self._boundary_conditions_type = []
            for i in range(dims):
                self._boundary_conditions_type.append([])
                self._boundary_conditions_type[i].append(bc[i])
                self._boundary_conditions_type[i].append(bc[i])
        else: #2 bcs per dimension, asymmetric
            self._boundary_conditions_type = []
            for i in range(dims):
                self._boundary_conditions_type.append([])
                self._boundary_conditions_type[i].append(bc[2*i])
                self._boundary_conditions_type[i].append(bc[2*i+1])
            
        
    def set_user_data(self, data):
        self.user_data = data

    def set_debug_mode_flag(self, debug_mode_flag):
        self._debug_mode_flag = debug_mode_flag
        return

    

    def _parallel_swap(self, init=False):
        sendrecv = self._uses_gpu and not(init)
        if(sendrecv):
            self.retrieve_fields_from_GPU_minimal()
        if(self._parallel):
            for i in range(len(self.fields)):
                self.fields[i]._swap()
            if not(self.temperature is None):
                self.temperature._swap()
        if(sendrecv):
            self.send_fields_to_GPU()
            
    def _apply_parallel_bcs(self):
        #currently this just calls gpu boundary conditions, as its the same for gpu serial and gpu parallel
        #TODO: make CPU parallel conditions
        ppf_gpu_utils.apply_parallel_bcs(self)

    def apply_boundary_conditions(self, init=False):
        #subtract 1 from time step counter, because "reasons". Dunno. It makes it work.
        #TODO: figure out why
        if(self._parallel):
            self._apply_parallel_bcs()
            if(((self.time_step_counter-1)%(self._ghost_rows) == 0) and not(init)):
                self._parallel_swap()
            return
        if(self._uses_gpu):
            ppf_gpu_utils.apply_boundary_conditions(self)
            return
        neumann_slices_target = [[(0), (slice(None), 0), (slice(None), slice(None), 0)], [(-1), (slice(None), -1), (slice(None), slice(None), -1)]]
        neumann_slices_source = [[(1), (slice(None), 1), (slice(None), slice(None), 1)], [(-2), (slice(None), -2), (slice(None), slice(None), -2)]]
        periodic_slices_target = [[(0), (slice(None), 0), (slice(None), slice(None), 0)], [(-1), (slice(None), -1), (slice(None), slice(None), -1)]]
        periodic_slices_source = [[(-2), (slice(None), -2), (slice(None), slice(None), -2)], [(1), (slice(None), 1), (slice(None), slice(None), 1)]]
        dirichlet_slices_target = [[(0), (slice(None), 0), (slice(None), slice(None), 0)],[(-1), (slice(None), -1), (slice(None), slice(None), -1)]]
        dims = len(self.fields[0].data.shape)
        for i in range(dims):
            for k in range(2):
                if(self._boundary_conditions_type[i][k] == "PERIODIC"):
                    if not(self.temperature is None):
                        self.temperature.data[periodic_slices_target[k][i]] = self.temperature.data[periodic_slices_source[k][i]]
                    for j in range(len(self.fields)):
                        self.fields[j].data[periodic_slices_target[k][i]] = self.fields[j].data[periodic_slices_source[k][i]]
                elif(self._boundary_conditions_type[i][k] == "NEUMANN"):
                    if not(self.temperature is None):
                        self.temperature.data[neumann_slices_target[k][i]] = self.temperature.data[neumann_slices_source[k][i]]
                    for j in range(len(self.fields)):
                        self.fields[j].data[neumann_slices_target[k][i]] = self.fields[j].data[neumann_slices_source[k][i]] - self.dx*self.boundary_fields[j].data[neumann_slices_target[k][i]]
                elif(self._boundary_conditions_type == "DIRICHLET"):
                    if not(self.temperature is None):
                        #use neumann boundary conditions for temperature field if using dirichlet boundary conditions
                        self.temperature.data[neumann_slices_target[k][i]] = self.temperature.data[neumann_slices_source[k][i]]
                    for j in range(len(self.fields)):
                        self.fields[j].data[dirichlet_slices_target[k][i]] = self.boundary_fields[j].data[dirichlet_slices_target[k][i]]
        return
    
    def send_fields_to_GPU(self):
        if(ppf_utils.successfully_imported_numba()): #unnecessary?
            ppf_gpu_utils.send_fields_to_GPU(self)
        return
    
    def retrieve_fields_from_GPU(self):
        if(ppf_utils.successfully_imported_numba()): #unnecessary?
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
        return
    
    def retrieve_fields_from_GPU_minimal(self):
        if(ppf_utils.successfully_imported_numba()): #unnecessary?
            ppf_gpu_utils.retrieve_fields_from_GPU_minimal(self)
        return
    
    def create_GPU_devices(self):
        if(ppf_utils.successfully_imported_numba()): #unnecessary?
            ppf_gpu_utils.create_GPU_devices(self)
        return
    
    def finish_simulation(self):
        if(ppf_utils.successfully_imported_numba()): #unnecessary?
            ppf_gpu_utils.clean_GPU(self)
        return

    def plot_all_fields(self):
        """
        Plots each field in self.fields and saves them to the save_path in a separate dir
        Recommended for when the number of fields used would clutter the data folder
        """
        image_folder = "images_step_" + str(self._time_step_counter) + "/"
        save_path = Path(self._save_path).joinpath(image_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        for i in range(len(self.fields)):
            self.plot_field(self.fields[i], save_path)
        return 0

    def plot_field(self, f, save_path=None):
        """
        Plots each field as a matplotlib 2d image. Takes in a field object as arg and saves
        the image to the data folder as namePlot_step_n.png
        """
        if(self._uses_gpu):
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
        if save_path is None:
            save_path = self._save_path
        fig, ax = plt.subplots()
        c = plt.imshow(f.data, interpolation='nearest', cmap=f.colormap)

        title = "Field: " + f.name + ", Step: " + str(self._time_step_counter)
        plt.title(title)
        fig.colorbar(c, ticks=np.linspace(np.min(f.data), np.max(f.data), 5))
        # Save image to save_path dir
        filename = f.name + "Plot_step_" + str(self._time_step_counter) + ".png"
        plt.savefig(Path(save_path).joinpath(filename))
        return 0

    def progress_bar(self):
        return

    def generate_python_script(self):
        return
    
    #import statements, specific to built-in Engines *TO BE REMOVED*

    def init_sim_Diffusion(self, dim=[200], solver="explicit", gmres=False, adi=False):
        Engines.init_Diffusion(self, dim, solver=solver, gmres=gmres, adi=adi)
        return
    
    def init_sim_DiffusionGPU(self, dim=[200, 200], cuda_blocks=(16,16), cuda_threads_per_block=(256,1)):
        if not ppf_utils.successfully_imported_numba():
            return
        Engines.init_DiffusionGPU(self, dim=dim, cuda_blocks=cuda_blocks, cuda_threads_per_block=cuda_threads_per_block)
        return
    
    def init_sim_CahnAllen(self, dim=[200], solver="explicit", gmres=False, adi=False):
        Engines.init_CahnAllen(self, dim, solver=solver, gmres=gmres, adi=adi)
        return
    
    def init_sim_CahnHilliard(self, dim=[200], solver="explicit", gmres=False, adi=False):
        Engines.init_CahnHilliard(self, dim, solver=solver, gmres=gmres, adi=adi)
        return

    def init_sim_Warren1995(self, dim=[200, 200], diamond_size=15):
        Engines.init_Warren1995(self, dim=dim, diamond_size=diamond_size)
        return

    def init_sim_NComponent(self, dim=[200, 200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb",
                            temperature_type="isothermal",
                            initial_temperature=1574, temperature_gradient=0, cooling_rate=0, temperature_file_path="T.xdmf",
                            initial_concentration_array=[0.40831], cell_spacing=0.0000046, d_ratio=1/0.94, solver="explicit", 
                            nbc=["periodic", "periodic"]):
        #initializes a Multicomponent simulation, using the NComponent model
        if not ppf_utils.successfully_imported_pycalphad():
            return
        Engines.init_NComponent(self, dim=dim, sim_type=sim_type, number_of_seeds=number_of_seeds, 
                                tdb_path=tdb_path, temperature_type=temperature_type, 
                                initial_temperature=initial_temperature, temperature_gradient=temperature_gradient, 
                                cooling_rate=cooling_rate, temperature_file_path=temperature_file_path, 
                                cell_spacing=cell_spacing, d_ratio=d_ratio, initial_concentration_array=initial_concentration_array, 
                                solver=solver, nbc=nbc)
        return
    
    def init_sim_NCGPU(self, dim=[200, 200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb",
                            temperature_type="isothermal",
                            initial_temperature=1574, temperature_gradient=0, cooling_rate=0, temperature_file_path="T.xdmf",
                            initial_concentration_array=[0.40831], cell_spacing=0.0000046, d_ratio=1/0.94, solver="explicit", 
                            nbc=["periodic", "periodic"], cuda_blocks = (16,16), cuda_threads_per_block = (256,1)):
        if not ppf_utils.successfully_imported_pycalphad():
            return
        if not ppf_utils.successfully_imported_numba():
            return
        
        Engines.init_NCGPU(self, dim=dim, sim_type=sim_type, number_of_seeds=number_of_seeds, 
                                tdb_path=tdb_path, temperature_type=temperature_type, 
                                initial_temperature=initial_temperature, temperature_gradient=temperature_gradient, 
                                cooling_rate=cooling_rate, temperature_file_path=temperature_file_path, 
                                cell_spacing=cell_spacing, d_ratio=d_ratio, initial_concentration_array=initial_concentration_array, 
                                solver=solver, nbc=nbc, cuda_blocks=cuda_blocks, cuda_threads_per_block=cuda_threads_per_block)
