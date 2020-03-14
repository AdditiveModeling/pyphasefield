"""
Functions that operate on a sim
"""
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


def make_save_loc(sim_name=None, overwrite=False):
    """
    Creates a default directory under which all simulation steps will be saved.
    If overwrite is set to true, exist_ok is also true
    """
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
    return save_loc


def save_fields(fields, step, save_loc):
    """
    Takes in fields as a dict and saves as a .npz
    Saves all fields in a .npz in a data folder which contains a folder for the simulation run.
    """

    # Save fields of a simulation in some default directory
    np.savez(str(save_loc) + "/step_" + str(step), **fields)
    return 0


def load_fields(file_path, step=-1):
    """
    Loads fields from either a .npz file or a directory containing a file with a name containing
    step_int(step). Returns a dict that can later be converted to the typical namedtuple
    """

    # Check for file path inside cwd
    print("Path:", Path.cwd().joinpath(file_path))
    if Path.cwd().joinpath(file_path).exists():
        file_path = Path.cwd().joinpath(file_path)

    if file_path.is_dir():
        if step > -1:
            file_path = file_path.joinpath("step_" + str(step) + ".npz")
        else:
            raise ValueError("Must specify step if path is directory")

    # Load dict
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


def plot_all_fields(fields, step, save_loc=None, cell_spacing=None):
    """
    Plots each field in fields and saves them to the save_path in a separate dir
    Recommended for when the number of fields used would clutter the data folder
    """
    image_folder = "images_step_" + str(step) + "/"
    if not save_loc:
        save_loc = Path.cwd()
    save_path = Path(save_loc).joinpath(image_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    for name, field in fields.items():
        plot_field(field, name, step, cell_spacing, save_path)
    return 0


def plot_field(name, field, step, cell_spacing=None, save_loc=None, display=False):
    """
    Plots a field as a matplotlib 2d image. Takes in a numpy 2d array as arg and saves
    the image to the data folder as namePlot_step_n.png
    """
    fig, ax = plt.subplots()
    ex = len(field)*cell_spacing
    c = plt.imshow(field, interpolation='bicubic', cmap="GnBu", extent=(0, ex, 0, ex))
    plt.title("Field: " + str(name) + ", Step: " + str(step))
    fig.colorbar(c, ticks=np.linspace(np.min(field), np.max(field), 5))

    # Formats axes if a cell spacing is provided
    if cell_spacing:
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0.01, 100))
        plt.xlabel("meters")
        plt.ylabel("meters")

    # Save image to save_path dir
    filename = str(name) + "Plot_step_" + str(step) + ".png"
    if not save_loc:
        save_loc = Path.cwd()
    plt.savefig(Path(save_loc).joinpath(filename))
    if display: plt.show()
    return None
