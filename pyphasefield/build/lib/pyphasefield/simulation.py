import numpy as np
from .field import Field

class Simulation:
    def __init__(self, save_path):
        self.fields = []
        self.temperature = None
        self._dimensions_of_simulation_region = [200,200]
        self._cell_spacing_in_cm = 1.
        self._time_step_in_seconds = 1.
        self._simulation_time_step_reached = 0
        self._temperature_type = "isothermal"
        self._initial_temperature_left_side = 1574.
        self._thermal_gradient_K_per_cm = 0.
        self._cooling_rate_K_per_second = 0.
        self._tdb = ""
        self._components = []
        self._phases = []
        self._engine = None
        self._save_path = save_path
        self._steps_per_checkpoint = 500
        self._save_images_at_each_checkpoint = False
        self._boundary_conditions_type = "periodic"
    
    def simulate(self, number_of_timesteps, dt=None):
        if dt is None:
            dt=self._time_step_in_seconds
        self._time_step_in_seconds = dt
        for i in range(number_of_timesteps):
            self.engine(self) #run engine on Simulation instance
        if(self._simulation_time_step_reached%self._steps_per_checkpoint == 0):
            self.save_simulation()
    
    def set_thermal_isothermal(self, temperature):
        array = np.zeros(self._dimensions_of_simulation_region)
        array += temperature
        field = Field(array, cell_spacing=self._cell_spacing_in_cm, dimensions_of_simulation_region=self._dimensions_of_simulation_region, number_of_dimensions=len(self._dimensions_of_simulation_region))
        self.temperature = field
        return
    
    def set_thermal_gradient(self, initial_T_left_side, dTdx, dTdt):
        return
    
    def set_thermal_file(self, thermal_file_path):
        return
    
    def update_thermal_field(self):
        if(self.temperature_type == "isothermal"):
            return
        elif(self.temperature_type == "gradient"):
            return
        elif(self.temperature_type == "gradient"):
            return
        #if it gets this far, warn user about unexpected temperature_type
        return
    
    def set_tdb(self, tdb_file_path):
        return
    
    def load_tdb_parameters(self):
        return
    
    def load_tdb_components(self):
        return
    
    def load_tdb_phases(self):
        return
    
    def set_tdb_components(self):
        return
    
    def set_tdb_phases(self):
        return
    
    def load_simulation(self):
        return
    
    def save_simulation(self):
        return
    
    def set_dimensions(self, dimensions_of_simulation_region):
        self._dimensions_of_simulation_region = dimensions_of_simulation_region
        return
    
    def set_cell_spacing(self, cell_spacing):
        self._cell_spacing_in_cm = cell_spacing
        return
    
    def add_field(self, field):
        #warn if field dimensions dont match simulation dimensions
        self.fields.append(field)
        return
    
    def set_engine(self, engine_function):
        self.engine = engine_function
        return
    
    def set_checkpoint_rate(self, steps_per_checkpoint):
        self._steps_per_checkpoint = steps_per_checkpoint
        return
    
    def set_automatic_plot_generation(self, plot_simulation_flag):
        self._save_images_at_each_checkpoint = plot_simulation_flag
        return
    
    def set_debug_mode(self, debug_mode_flag):
        self._debug_mode_flag = debug_mode_flag
        return
    
    def set_boundary_conditions(self, boundary_conditions_type):
        self._boundary_conditions_type = boundary_conditions_type
    
    def increment_step_counter(self):
        self._simulation_time_step_reached += 1
        return
    
    def apply_boundary_conditions(self):
        return
    
    def renormalize_quaternions(self):
        return
    
    def cutoff_order_values(self):
        return
    
    def plot_fields(self):
        return
    
    def progress_bar(self):
        return
    
    def generate_python_script(self):
        return
    
    def init_sim_Diffusion(self, dim=[200]):
        def Diffusion_engine(sim):
            D = 0.1
            dt = sim._time_step_in_seconds
            c = sim.fields[0]
            dc = dt*(D*c.laplacian())
            sim.fields[0] += dc
            sim.increment_step_counter()
        self.set_dimensions(dim)
        self.set_cell_spacing(1.)
        c = np.zeros(dim)
        length = dim[0]
        c[length//4:3*length//4] = 1
        c_field = Field(c, name="c", cell_spacing=self._cell_spacing_in_cm)
        self.add_field(c_field)
        self.set_engine(Diffusion_engine)
        return