import numpy as np
# import meshio as mio
from .field import Field
from . import Engines

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
        # self.fields = []
        # self.temperature = None
        # self._dimensions_of_simulation_region = [200, 200]
        # self._cell_spacing_in_meters = 1.
        # self._time_step_in_seconds = 1.
        # self._time_step_counter = 0
        # self._temperature_type = "isothermal"
        # self._initial_temperature_left_side = 1574.
        # self._thermal_gradient_Kelvin_per_meter = 0.
        # self._cooling_rate_Kelvin_per_second = 0.  # cooling is a negative number! this is dT/dt
        # self._tdb = None
        # self._tdb_path = ""
        # self._components = []
        # self._phases = []
        # self._engine = None
        # self._save_path = save_path
        # self._time_steps_per_checkpoint = 500
        # self._save_images_at_each_checkpoint = False
        # self._boundary_conditions_type = ["periodic", "periodic"]
        # self._engine_data = {}

    def simulate(fields, engine, number_of_timesteps):
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
            self._time_step_counter += 1
            self._engine(self)  # run engine on Simulation instance for 1 time step
            apply_boundary_conditions()
            update_thermal_field()
            if self._time_step_counter % self._time_steps_per_checkpoint == 0:
                save_simulation(self)

    def init_sim_Diffusion(self, dim=[200]):
        Engines.init_Diffusion(self, dim)
        return

    def init_sim_NComponent(self, dim=[200, 200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb",
                            thermal_type="isothermal",
                            initial_temperature=1574, thermal_gradient=0, cooling_rate=0, thermal_file_path="T.xdmf",
                            initial_concentration_array=[0.40831], cell_spacing=0.0000046, d_ratio=1/0.94):
        #initializes a Multicomponent simulation, using the NComponent model
        Engines.init_NComponent(self, dim=dim, sim_type=sim_type, number_of_seeds=number_of_seeds,
                                tdb_path=tdb_path, thermal_type=thermal_type,
                                initial_temperature=initial_temperature, thermal_gradient=thermal_gradient,
                                cooling_rate=cooling_rate, thermal_file_path=thermal_file_path,
                                cell_spacing=cell_spacing, d_ratio=d_ratio, initial_concentration_array=initial_concentration_array)
        return
