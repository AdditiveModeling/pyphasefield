"""
Functions relating to updating fields during a simulation
"""
import numpy as np
import pyphasefield.io as io
import dataclasses


@dataclasses.dataclass(frozen=True)
class Component:
    """A data class that holds physical properties for a component"""
    T_m: float  # Melting temperature (K)
    L: float  # Latent heat (J/m^3)
    S: float  # Surface energy (J/m^2)
    B: float  # Linear kinetic coefficient (m/K*s)

    def W(self, d):
        return 3 * self.S / ((2 ** 0.5) * self.T_m * d)

    def M(self, d):
        return self.T_m * self.T_m * self.B / (6 * 2 ** 0.5 * self.L * d)


def run(fields, engine, steps, output=None, plot_fields=False):
    """
    A function for running a simple simulation. Mutates the inputted fields using the engine
    for a number of times specified by steps. Will output a certain number of checkpoints if specified.
    """
    save_loc = io.make_save_loc()
    for step in range(1, steps+1):
        engine(fields)
        if output and not step % output:
            io.save_fields(fields._asdict(), step, save_loc)
            if plot_fields:
                io.plot_all_fields(fields._asdict(), step, save_loc)
    return None


def expand_T_array(T, nbc):
    """Used by Simulation.set_thermal_file() to add boundary cells if not using periodic boundary conditions."""
    shape = list(T.shape)
    offset_x = 0
    offset_y = 0
    if nbc[0]:
        shape[1] += 2
        offset_x = 1
    if nbc[1]:
        shape[0] += 2
        offset_y = 1
    final = np.zeros(shape)
    # Set center region equal to T
    final[offset_y:len(final)-offset_y, offset_x:len(final[0])-offset_x] += T
    # Set edges to nbcs, if applicable
    final[0] = final[offset_y]
    final[len(final)-1] = final[len(final)-offset_y-1]
    final[:, 0] = final[:, offset_x]
    final[:, len(final[0])-1] = final[:, len(final[0])-offset_x-1]
    return final


def set_thermal_isothermal(sim, temperature):
    """
    Sets the simulation to use an isothermal temperature profile
    The temperature variable is a Field instance
    Data stored within the Field instance is a numpy ndarray, with the same value
        in each cell (defined by the parameter "temperature" to this method)
    (Could be a single value, but this way it won't break Engines that compute thermal gradients)
    """
    sim._temperature_type = "isothermal"
    sim._initial_temperature_left_side = temperature
    array = np.zeros(sim._dimensions_of_simulation_region)
    array += temperature
    t_field = Field(data=array, name="Temperature (K)")
    sim.temperature = t_field
    return


def set_thermal_gradient(self, initial_T_left_side, dTdx, dTdt):
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
    self._thermal_gradient_Kelvin_per_meter = dTdx
    self._cooling_rate_Kelvin_per_second = dTdt
    array = np.zeros(self._dimensions_of_simulation_region)
    array += self.temperature
    array += np.linspace(0, dTdx * self.shape[1] * self._cell_spacing_in_m, self.shape[1])
    array += self.get_time_step_reached() * self.get_time_step_length() * dTdt
    t_field = Field(data=array, name="Temperature (K)")
    self.temperature = t_field
    return


def set_thermal_file(sim, thermal_file_path):
    """
    Sets the simulation to import the temperature from an xdmf file containing the temperature at given timesteps
    The temperature variable is a Field instance, data stored within the Field instance is a numpy ndarray
    Loads the file at the path "[thermal_file_path]/T.xdmf"
    Uses linear interpolation to find the temperature at times between stored timesteps
    E.g.: If we have T0 stored at t=1 second, T1 stored at t=2 seconds, the temperature
        profile at t=1.25 seconds = 0.75*T0 + 0.25*T1
    """
    sim._temperature_type = "file"
    sim._
    sim.t_index = 1
    nbc = []
    for i in range(len(sim._dimensions_of_simulation_region)):
        if(boundary_conditions[i] == "periodic"):
            nbc.append(False)
        else:
            nbc.append(True)
    with mio.XdmfTimeSeriesReader(sim._save_path+"/T.xdmf") as reader:
        dt = sim.get_time_step_length()
        step = sim.get_time_step_counter()
        points, cells = reader.read_points_cells()
        sim.t_start, point_data0, cell_data0 = reader.read_data(0)
        sim.T0 = expand_T_array(point_data0['T'], nbc)
        sim.t_end, point_data1, cell_data0 = reader.read_data(sim.t_index)
        sim.T1 = expand_T_array(point_data1['T'], nbc)
        while(dt*step > t_end):
            sim.t_start= sim.t_end
            sim.T0 = sim.T1
            sim.t_index += 1
            sim.t_end, point_data1, cell_data0 = reader.read_data(sim.t_index)
            sim.T1 = expand_T_array(point_data1['T'], nbc)
        array = sim.T0*(sim.t_end - dt*step)/(sim.t_end-sim.t_start) + sim.T1*(dt*step-sim.t_start)/(sim.t_end-sim.t_start)
        t_field = Field(data=array, name="Temperature (K)")
        sim.temperature = t_field
    return


def update_thermal_field(temperature_field, temperature_type, file_path=None, dT=None):
    """Updates the thermal field, method assumes only one timestep has passed"""
    if sim._temperature_type == "isothermal":
        return
    elif sim._temperature_type == "gradient":
        sim.temperature.data += sim._cooling_rate_Kelvin_per_second*sim._time_step_in_seconds
        return
    elif sim._temperature_type == "file":
        dt = sim.get_time_step_length()
        step = sim.get_time_step_counter()
        if dt*step > t_end:
            nbc = []
            for i in range(len(sim._dimensions_of_simulation_region)):
                if sim._boundary_conditions_type[i] == "periodic":
                    nbc.append(False)
                else:
                    nbc.append(True)
            with mio.XdmfTimeSeriesReader(sim._save_path+"/T.xdmf") as reader:
                sim.t_start= sim.t_end
                sim.T0 = sim.T1
                sim.t_index += 1
                sim.t_end, point_data1, cell_data0 = reader.read_data(sim.t_index)
                sim.T1 = expand_T_array(point_data1['T'], nbc)
        sim.temperature.data = sim.T0*(sim.t_end - dt*step)/(sim.t_end-sim.t_start) + sim.T1*(dt*step-sim.t_start)/(sim.t_end-sim.t_start)
        return
    if sim._temperature_type not in ["isothermal", "gradient", "file"]:
        raise ValueError("Unknown temperature profile.")


def load_tdb(sim, tdb_path, phases=None, components=None):
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
    import pycalphad as pyc
    sim._tdb_path = tdb_path
    sim._tdb = pyc.Database(tdb_path)
    if phases is None:
        sim._phases = list(sim._tdb.phases)
    else:
        sim._phases = phases
    if components is None:
        sim._components = list(sim._tdb.elements)
    else:
        sim._components = components
    sim._phases.sort()
    sim._components.sort()


def apply_boundary_conditions(fields, bc_type):
    if bc_type[0] == "neumann":
        for i in range(len(fields)):
            length = len(fields[i].data[0])
            fields[i].data[:, 0] = fields[i].data[:, 1]
            fields[i].data[:, (length-1)] = fields[i].data[:, (length-2)]
    if bc_type[1] == "neumann":
        for i in range(len(fields)):
            length = len(fields[i].data)
            fields[i].data[0] = fields[i].data[1]
            fields[i].data[(length-1)] = fields[i].data[(length-2)]
            return 0
    else:
        raise ValueError("BC type not valid")
