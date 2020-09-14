import numpy as np
import meshio as mio
from .field import Field
from . import Engines
from pathlib import Path
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from . import ppf_utils
try:
    import pycalphad as pyc
    from . import ppf_gpu_utils
except:
    pass

#TODO: neumann boundary conditions breaks for "1d" sims (size 1 dimensions)
    
def expand_T_array(T, nbc):
    """Used by Simulation.set_thermal_file() to add boundary cells if not using periodic boundary conditions."""
    shape = list(T.shape)
    offset_x = 0
    offset_y = 0
    if(nbc[0]):
        shape[1] += 2
        offset_x = 1
    if(nbc[1]):
        shape[0] += 2
        offset_y = 1
    final = np.zeros(shape)
    #set center region equal to T
    final[offset_y:len(final)-offset_y, offset_x:len(final[0])-offset_x] += T
    #set edges to nbcs, if applicable
    final[0] = final[offset_y]
    final[len(final)-1] = final[len(final)-offset_y-1]
    final[:, 0] = final[:, offset_x]
    final[:, len(final[0])-1] = final[:, len(final[0])-offset_x-1]
    return np.squeeze(final)

def get_non_edge_cells_from_nbcs(array, nbc):
    offset_x = 0
    offset_y= 0
    if(nbc[0]):
        offset_x = 1;
    if(nbc[1]):
        offset_y = 1;
    return array[offset_y:len(array)-offset_y, offset_x:len(array[0])-offset_x]
        

class Simulation:
    def __init__(self, save_path=None):
        """
        Class used by pyphasefield to store data related to a given simulation
        Methods of Simulation are used to:
            * Run simulations
            * Set what engine is used to evolve the fields each timestep
            * Save and load checkpoints
            * Plot fields
            * Set and evolve the thermal profile of the simulation
            * Load TDB files used by certain engines (requires pycalphad!)
        
        Data specific to a particular field is stored within the Field class
        """
        self.fields = []
        self.uses_gpu = False
        self.temperature = None
        self.temperature_gpu_device = None
        self._dimensions_of_simulation_region = [200, 200]
        self._cell_spacing_in_meters = 1.
        self._time_step_in_seconds = 1.
        self._time_step_counter = 0
        self._temperature_type = "isothermal"
        self._initial_temperature_left_side = 1574.
        self._temperature_gradient_Kelvin_per_meter = 0.
        self._cooling_rate_Kelvin_per_second = 0.  # cooling is a negative number! this is dT/dt
        self._tdb = None
        self._tdb_path = ""
        self._components = []
        self._phases = []
        self._engine = None
        self._save_path = save_path
        self._time_steps_per_checkpoint = 500
        self._save_images_at_each_checkpoint = False
        self._boundary_conditions_type = ["periodic", "periodic"]

    def simulate(self, number_of_timesteps, dt=None):
        """
        Evolves the simulation for a specified number of timesteps
        If a length of timestep is not specified, uses the timestep length stored within the Simulation instance
        For each timestep, the method:
            * Increments the timestep counter by 1
            * Runs the engine function (a function which only takes the Simulation instance as a 
                parameter, and evolves the fields contained within the instance by one time step
            * Updates the thermal field of the simulation depending on which thermal type the simulation is:
                - Isothermal: Do nothing
                - Gradient: Add dT/dt, multiplied by the timestep length, to the thermal field
                - File: Use linear interpolation to find the thermal field of the new timestep
            * If the timestep counter is a multiple of time_steps_per_checkpoint, save a checkpoint of the simulation
        """
        if dt is None:
            dt = self._time_step_in_seconds
        self._time_step_in_seconds = dt
        for i in range(number_of_timesteps):
            self.increment_time_step_counter()
            self._engine(self)  # run engine on Simulation instance for 1 time step
            self.apply_boundary_conditions()
            self.update_temperature_field()
            if self._time_step_counter % self._time_steps_per_checkpoint == 0:
                if(self.uses_gpu):
                    ppf_gpu_utils.retrieve_fields_from_GPU(self)
                self.save_simulation()

    def load_tdb(self, tdb_path, phases=None, components=None):
        """
        Loads the tdb file using pycalphad. (Needless to say, this requires pycalphad!)
        The format for phases and components attributes of Simulation are a list of strings 
            that correspond to the terms within the tdb file
        Examples:
            * phases=["FCC_A1", "LIQUID"]
            * components=["CU", "NI"]
        Unless specified, method will load all phases and components contained within the tdb file.
        phases and components lists are always in alphabetical order, and will be automatically 
            sorted if not already done by the user
        """
        if not ppf_utils.successfully_imported_pycalphad():
            return
        import pycalphad as pyc
        self._tdb_path = tdb_path
        self._tdb = pyc.Database(tdb_path)
        if phases is None:
            self._phases = list(self._tdb.phases)
        else:
            self._phases = phases
        if components is None:
            self._components = list(self._tdb.elements)
        else:
            self._components = components
        self._phases.sort()
        self._components.sort()

    def get_time_step_length(self):
        """Returns the length of a single timestep"""
        return self._time_step_in_seconds

    def get_time_step_counter(self):
        """Returns the number of timesteps that have passed for a given simulation"""
        return self._time_step_counter

    def set_time_step_length(self, time_step):
        """Sets the length of a single timestep for a simulation instance"""
        self._time_step_in_seconds = time_step
        return

    def set_temperature_isothermal(self, temperature):
        """
        Sets the simulation to use an isothermal temperature profile
        The temperature variable is a Field instance
        Data stored within the Field instance is a numpy ndarray, with the same value 
            in each cell (defined by the parameter "temperature" to this method)
        (Could be a single value, but this way it won't break Engines that compute thermal gradients)
        """
        self._temperature_type = "isothermal"
        self._initial_temperature_left_side = temperature
        array = np.zeros(self._dimensions_of_simulation_region)
        array += temperature
        t_field = Field(data=array, name="Temperature (K)")
        self.temperature = t_field
        return

    def set_temperature_gradient(self, initial_T_left_side, dTdx, dTdt):
        """
        Sets the simulation to use a linear gradient temperature profile (frozen gradient approximation)
        The temperature variable is a Field instance, data stored within the Field instance is a numpy ndarray
        Thermal profile is defined by 3 parameters:
            * initial_T_left_side: The temperature of the left side of the 
                simulation region (in slicing notation, this is self.temperature.data[:, 0])
            * dTdx: Spacial derivative of temperature, which defines the gradient. The initial temperature 
                at a point x meters from the left side equals (initial_T_left_side + dTdx*x)
            * dTdt: Temporal derivative of temperature. Temperature at time t seconds from the start of the 
                simulation and a distance x meters from the left side equals 
                (initial_T_left_side + dTdx*x + dTdt*t)
        """
        self._temperature_type = "gradient"
        self._initial_temperature_left_side = initial_T_left_side
        self._temperature_gradient_Kelvin_per_meter = dTdx
        self._cooling_rate_Kelvin_per_second = dTdt
        array = np.zeros(self._dimensions_of_simulation_region)
        array += initial_T_left_side
        array += np.linspace(0, dTdx * self._dimensions_of_simulation_region[1] * self._cell_spacing_in_meters, self._dimensions_of_simulation_region[1])
        array += self.get_time_step_counter() * self.get_time_step_length() * dTdt
        t_field = Field(data=array, name="Temperature (K)")
        self.temperature = t_field
        return

    def set_temperature_file(self, temperature_file_path):
        """
        Sets the simulation to import the temperature from an xdmf file containing the temperature at given timesteps
        The temperature variable is a Field instance, data stored within the Field instance is a numpy ndarray
        Loads the file at the path "[temperature_file_path]/T.xdmf"
        Uses linear interpolation to find the temperature at times between stored timesteps
        E.g.: If we have T0 stored at t=1 second, T1 stored at t=2 seconds, the temperature
            profile at t=1.25 seconds = 0.75*T0 + 0.25*T1
        """
        self._temperature_type = "file"
        self.t_index = 1
        nbc = []
        for i in range(len(self._dimensions_of_simulation_region)):
            if(self._boundary_conditions_type[i] == "periodic"):
                nbc.append(False)
            else:
                nbc.append(True)
        with mio.xdmf.TimeSeriesReader(self._save_path+"/T.xdmf") as reader:
            dt = self.get_time_step_length()
            step = self.get_time_step_counter()
            points, cells = reader.read_points_cells()
            self.t_start, point_data0, cell_data0 = reader.read_data(0)
            self.T0 = expand_T_array(point_data0['T'], nbc)
            self.t_end, point_data1, cell_data0 = reader.read_data(self.t_index)
            self.T1 = expand_T_array(point_data1['T'], nbc)
            print(self.T0.shape, self.T1.shape)
            while(dt*step > self.t_end):
                self.t_start= self.t_end
                self.T0 = self.T1
                self.t_index += 1
                self.t_end, point_data1, cell_data0 = reader.read_data(self.t_index)
                self.T1 = expand_T_array(point_data1['T'], nbc)
                print(self.T0.shape, self.T1.shape)
            array = self.T0*(self.t_end - dt*step)/(self.t_end-self.t_start) + self.T1*(dt*step-self.t_start)/(self.t_end-self.t_start)
            t_field = Field(data=array, name="Temperature (K)")
            self.temperature = t_field
        return

    def update_temperature_field(self):
        """Updates the thermal field, method assumes only one timestep has passed"""
        if(self.uses_gpu):
            ppf_gpu_utils.update_temperature_field(self)
            return
        elif(self._temperature_type == "isothermal"):
            return
        elif(self._temperature_type == "gradient"):
            self.temperature.data += self._cooling_rate_Kelvin_per_second*self._time_step_in_seconds
            return
        elif(self._temperature_type == "file"):
            dt = self.get_time_step_length()
            step = self.get_time_step_counter()
            while(dt*step > self.t_end):
                nbc = []
                for i in range(len(self._dimensions_of_simulation_region)):
                    if(self._boundary_conditions_type[i] == "periodic"):
                        nbc.append(False)
                    else:
                        nbc.append(True)
                with mio.xdmf.TimeSeriesReader(self._save_path+"/T.xdmf") as reader:
                    reader.cells=[]
                    self.t_start= self.t_end
                    self.T0 = self.T1
                    self.t_index += 1
                    self.t_end, point_data1, cell_data0 = reader.read_data(self.t_index)
                    self.T1 = expand_T_array(point_data1['T'], nbc)
            self.temperature.data = self.T0*(self.t_end - dt*step)/(self.t_end-self.t_start) + self.T1*(dt*step-self.t_start)/(self.t_end-self.t_start)
            return
        if self._temperature_type not in ["isothermal", "gradient", "file"]:
            raise ValueError("Unknown temperature profile.")

    def load_simulation(self, file_path=None, step=-1):
        """
        Loads a simulation from a .npz file. Either a filename, step, or both must be provided.
            If no step is specified, checks filename for step #.
            If no filename is specified, a file with the specified step number is loaded from
            the _save_path.
        """
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
            
        if file_path.is_dir():
            if step > -1:
                file_path = file_path.joinpath("step_"+str(step)+".npz")
            else:
                raise ValueError("Given path is a folder, must specify a timestep!")

        #propagate new path to the save path, the parent folder is the save path
        #only does so if the save path for the simulation is not set!
        if(self._save_path is None):
            self._save_path = str(file_path.parent)

        # Load array
        fields_dict = np.load(file_path, allow_pickle=True)

        # Add arrays self.fields as Field objects
        for key, value in fields_dict.items():
            tmp = Field(value, self, key)
            self.fields.append(tmp)
            
        # Set dimensions of simulation
        self._dimensions_of_simulation_region = self.fields[0].data.shape

        # Time step set from parsing file name or manually --> defaults to 0
        if step < 0:
            filename = file_path.stem
            step_start_index = filename.find('step_') + len('step_')
            if step_start_index == -1:
                self._time_step_counter = 0
            else:
                i = step_start_index
                while i < len(filename) and filename[i].isdigit():
                    i += 1
                self._time_step_counter = int(filename[step_start_index:i])
        else:
            self._time_step_counter = int(step)
        if(self._temperature_type == "gradient"):
            self.temperature.data += self._time_step_counter*self._cooling_rate_Kelvin_per_second*self._time_step_in_seconds
        if(self._temperature_type == "file"):
            self.update_temperature_field()
        return 0
    
    def save_simulation(self):
        """
        Saves all fields in a .npz in either the user-specified save path or a default path. Step number is saved
        in the file name.
        TODO: save data for simulation instance in header file
        """
        save_dict = dict()
        for i in range(len(self.fields)):
            tmp = self.fields[i]
            save_dict[tmp.name] = tmp.data

        # Save array with path
        if not self._save_path:
            engine_name = self._engine.__name__
            print("Simulation.save_path not specified, saving to /data/"+engine_name)
            save_loc = Path.cwd().joinpath("data/", engine_name)
        else:
            save_loc = Path(self._save_path)
        save_loc.mkdir(parents=True, exist_ok=True)

        np.savez(str(save_loc) + "/step_" + str(self._time_step_counter), **save_dict)
        return 0
    
    def plot_simulation(self, fields=None, interpolation="bicubic"):
        if(self.uses_gpu):
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
        if fields is None:
            fields = range(len(self.fields))
        if(len(self.fields[0].data.shape) == 1): #1d plot
            x = np.arange(0, len(self.fields[0].data))
            for i in fields:
                plt.plot(x, self.fields[i].data)
            plt.show()
        elif((len(self.fields[0].data) == 1) or (len(self.fields[0].data[0]) == 1)): #effective 1d plot, 2d but one dimension = 1
            x = np.arange(0, len(self.fields[0].data.flatten()))
            legend = []
            for i in fields:
                plt.plot(x, self.fields[i].data.flatten())
                legend.append(self.fields[i].name)
            plt.legend(legend)
            plt.show()
        else:
            for i in fields:
                plt.imshow(self.fields[i].data, interpolation=interpolation, cmap=self.fields[i].colormap)
                plt.title(self.fields[i].name)
                plt.show()
                

    def set_dimensions(self, dimensions_of_simulation_region):
        self._dimensions_of_simulation_region = dimensions_of_simulation_region
        return
    
    def get_dimensions(self):
        return self._dimensions_of_simulation_region

    def set_cell_spacing(self, cell_spacing):
        self._cell_spacing_in_meters = cell_spacing
        return

    def get_cell_spacing(self):
        return self._cell_spacing_in_meters

    def add_field(self, field):
        # warn if field dimensions dont match simulation dimensions
        self.fields.append(field)
        return

    def set_engine(self, engine_function):
        self._engine = engine_function
        return

    def set_checkpoint_rate(self, time_steps_per_checkpoint):
        self._time_steps_per_checkpoint = time_steps_per_checkpoint
        return

    def set_automatic_plot_generation(self, plot_simulation_flag):
        self._save_images_at_each_checkpoint = plot_simulation_flag
        return

    def set_debug_mode(self, debug_mode_flag):
        self._debug_mode_flag = debug_mode_flag
        return

    def set_boundary_conditions(self, boundary_conditions_type):
        self._boundary_conditions_type = boundary_conditions_type

    def increment_time_step_counter(self):
        self._time_step_counter += 1
        return

    def apply_boundary_conditions(self):
        if(self.uses_gpu):
            ppf_gpu_utils.apply_boundary_conditions(self)
            return
        if(self._boundary_conditions_type[0] == "neumann"):
            for i in range(len(self.fields)):
                length=len(self.fields[i].data[0])
                self.fields[i].data[:,0] = self.fields[i].data[:,1]
                self.fields[i].data[:,(length-1)] = self.fields[i].data[:,(length-2)]
        if(self._boundary_conditions_type[1] == "neumann"):
            for i in range(len(self.fields)):
                length=len(self.fields[i].data)
                self.fields[i].data[0] = self.fields[i].data[1]
                self.fields[i].data[(length-1)] = self.fields[i].data[(length-2)]
        return

    def renormalize_quaternions(self):
        return

    def cutoff_order_values(self):
        return
    
    def send_fields_to_GPU(self):
        if(ppf_utils.successfully_imported_numba()):
            ppf_gpu_utils.send_fields_to_GPU(self)
        return
    
    def retrieve_fields_from_GPU(self):
        if(ppf_utils.successfully_imported_numba()):
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
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
        if(self.uses_gpu):
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
    
    #import statements, specific to built-in Engines

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
