"""
Functions that operate on a sim
"""
import numpy as np
from pyphasefield.field import Field
from pathlib import Path
from matplotlib import pyplot as plt


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


def update_thermal_field(sim):
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


def load_simulation(sim, file_path=None, step=-1):
    """
    Loads a simulation from a .npz file. Either a filename, step, or both must be provided.
        If no step is specified, checks filename for step #.
        If no filename is specified, a file with the specified step number is loaded from
        the _save_path.
    """
    if file_path is None:
        file_path = sim._save_path
        if sim._save_path is None:
            raise ValueError("Simulation needs a path to load data from!")

    # Check for file path inside cwd
    if Path.cwd().joinpath(file_path).exists():
        file_path = Path.cwd().joinpath(file_path)
    # Check if file exists in the save directory
    elif Path.cwd().joinpath(sim._save_path).joinpath(file_path).exists():
        file_path = Path.cwd().joinpath(sim._save_path).joinpath(file_path)
    else:
        file_path = Path(file_path)

    if file_path.is_dir():
        if step > -1:
            file_path = file_path.joinpath("step_" + str(step) + ".npz")
        else:
            raise ValueError("Given path is a folder, must specify a timestep!")

    # propagate new path to the save path, the parent folder is the save path
    # only does so if the save path for the simulation is not set!
    if sim._save_path is None:
        sim._save_path = str(file_path.parent)

    # Load array
    fields_dict = np.load(file_path, allow_pickle=True)

    # Add arrays sim.fields as Field objects
    for key, value in fields_dict.items():
        tmp = Field(value, sim, key)
        sim.fields.append(tmp)

    # Set dimensions of simulation
    sim._dimensions_of_simulation_region = sim.fields[0].data.shape

    # Time step set from parsing file name or manually --> defaults to 0
    if step < 0:
        filename = file_path.stem
        step_start_index = filename.find('step_') + len('step_')
        if step_start_index == -1:
            sim._time_step_counter = 0
        else:
            i = step_start_index
            while i < len(filename) and filename[i].isdigit():
                i += 1
            sim._time_step_counter = int(filename[step_start_index:i])
    else:
        sim._time_step_counter = int(step)
    return 0


def save_simulation(sim):
    """
    Saves all fields in a .npz in either the user-specified save path or a default path. Step number is saved
    in the file name.
    TODO: save data for simulation instance in header file
    """
    save_dict = dict()
    for i in range(len(sim.fields)):
        tmp = sim.fields[i]
        save_dict[tmp.name] = tmp.data

    # Save array with path
    if not sim._save_path:
        engine_name = sim._engine.__name__
        print("Simulation.save_path not specified, saving to /data/"+engine_name)
        save_loc = Path.cwd().joinpath("data/", engine_name)
    else:
        save_loc = Path(sim._save_path)
    save_loc.mkdir(parents=True, exist_ok=True)

    np.savez(str(save_loc) + "/step_" + str(sim._time_step_counter), **save_dict)
    return 0


def save_fields(fields, step, sim_name=None, overwrite=False):
    """
    Takes in fields as a dir and saves as a .npz
    Saves all fields in a .npz in a data folder which contains a folder for the simulation run.
    If overwrite is set to True, the function might write to a dir with existent data
        otherwise, a new directory will be found
    """
    # Save fields of a simulation in some default directory
    if not sim_name:
        n = 1
        save_loc = Path.cwd().joinpath("data/sim" + str(n))
        # Finds a unique dir name under data if overwrite is set to false
        if not overwrite:
            while save_loc.is_dir():
                n += 1
                save_loc = Path.cwd().joinpath("data/sim" + str(n))
    else:
        save_loc = Path.cwd().joinpath("data/" + str(sim_name))
    save_loc.mkdir(parents=True, exist_ok=overwrite)
    np.savez(str(save_loc) + "/step_" + str(step), **fields)
    return 0


def load_fields(file_path, step=-1):
    """
    Loads fields from either a .npz file or a directory containing a file with a name containing
    step_int(step). Returns a dict that can later be converted to the typical namedtuple
    """

    # Check for file path inside cwd
    if Path.cwd().joinpath(file_path).exists():
        file_path = Path.cwd().joinpath(file_path)

    if file_path.is_dir():
        if step > -1:
            file_path = file_path.joinpath("step_" + str(step) + ".npz")
        else:
            raise ValueError("Must specify step if path is directory")
    # Load array
    fields_dict = np.load(file_path, allow_pickle=True)

    # Time step set from parsing file name or manually --> defaults to 0
    if step < 0:
        filename = file_path.stem
        step_start_index = filename.find('step_') + len('step_')
        if step_start_index == -1:
            step = 0
        else:
            i = step_start_index
            while i < len(filename) and filename[i].isdigit():
                i += 1
            step = int(filename[step_start_index:i])
    else:
        step = int(step)
    return fields_dict, step


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


def plot_all_fields(fields, step, cell_spacing=1):
    """
    Plots each field in self.fields and saves them to the save_path in a separate dir
    Recommended for when the number of fields used would clutter the data folder
    """
    image_folder = "images_step_" + str(step) + "/"
    save_path = Path(self._save_path).joinpath(image_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(fields)):
        plot_field(fields[i], save_path)
    return 0


def plot_field(field, name, step, cell_spacing=1, save_path=None):
    """
    Plots a field as a matplotlib 2d image. Takes in a field object as arg and saves
    the image to the data folder as namePlot_step_n.png
    """
    fig, ax = plt.subplots()
    ex = len(field)*cell_spacing
    c = plt.imshow(field.data, interpolation='nearest', cmap="GnBu", extent=(0, ex, 0, ex))

    ax.ticklabel_format(axis='both', style='sci', scilimits=(0.01, 100))
    plt.title("Field: " + str(name) + ", Step: " + str(step))
    fig.colorbar(c, ticks=np.linspace(np.min(field.data), np.max(field.data), 5))
    plt.xlabel("meters")
    plt.ylabel("meters")
    # Save image to save_path dir
    # filename = field.name + "Plot_step_" + str(step) + ".png"
    # plt.savefig(Path(save_path).joinpath(filename))
    plt.show()
    return 0
