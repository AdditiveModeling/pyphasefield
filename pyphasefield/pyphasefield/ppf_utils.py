from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import symengine as se
from tinydb import where
import h5py
from glob import glob
import pathlib
import linecache
import re
try:
    import cupy as cp
    from cupyx import jit
except:
    print("Cannot import cupy, therefore cannot create TDB ufuncs built for GPUs")

colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
COLORMAP_OTHER = LinearSegmentedColormap.from_list('rgb', colors)
colors2 = [(1, 0, 0), (1, 1, 0), (0, 0, 1)]
COLORMAP_PHASE = LinearSegmentedColormap.from_list('rgb', colors2)
colors2 = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]
COLORMAP_PHASE_INV = LinearSegmentedColormap.from_list('rgb', colors2)

#store seed nuclei masks for repeated use
seed_mask_p = None
seed_mask_q = None
r_p = None
r_q = None
seed_dim = None

#units dict for easy lookup
units_dict = {
    "m": 1.,
    "cm": 0.01,
    "mm": 0.001,
    "um": 0.000001,
    "nm": 0.000000001,
    "s": 1.,
    "ms": 0.001,
    "us": 0.000001,
    "ns": 0.000000001
}

def successfully_imported_pycalphad():
    """
    Checks if pycalphad is installed. 
    If not, warns the user that pycalphad-dependent features cannot be used
    Also tells the user how to install it (if the user has Anaconda)
    """
    try:
        import pycalphad as pyc
        from pycalphad.core.utils import instantiate_models
    except ImportError:
        print("The feature you are trying to use requires pycalphad")
        print("In Anaconda, use \'conda install -c pycalphad -c conda-forge pycalphad\' to install it")
        return False
    return True

def successfully_imported_cupy():
    """
    Checks if numba/cuda is installed. 
    If not, warns the user that gpu-dependent features cannot be used
    Also tells the user how to install it (if the user has Anaconda)
    """
    try:
        import cupy as cp
        from cupyx import jit
        from . import ppf_gpu_utils
    except ImportError as e:
        print("The feature you are trying to use requires cupy")
        print("Use \'pip install cupy\' to install, building against whichever GPU you have")
        print(e)
        return False
    return True

def make_seed_masks(r, q_extra, ndims):
    global seed_mask_p, seed_mask_q, r_p, r_q, seed_dim
    seed_dim = ndims
    r_p = r
    r_q = r+q_extra
    p_slices = []
    q_slices = []
    for i in range(ndims):
        p_slices.append(slice(0, 2*r_p))
        q_slices.append(slice(0, 2*r_q))
    p_grid = np.ogrid[tuple(p_slices)]
    q_grid = np.ogrid[tuple(q_slices)]
    dist2p = 0
    dist2q = 0
    for i in range(ndims):
        dist2p = dist2p + (p_grid[i]-r_p)**2 
        dist2q = dist2q + (q_grid[i]-r_q)**2 
    seed_mask_p = dist2p < r_p**2
    seed_mask_q = dist2q < r_q**2
    
def random_uniform_quaternion():
    
    u = 2*np.pi*np.random.rand(2)
    v = np.sqrt(np.random.rand(2))
    x = v*np.cos(u)
    y = v*np.sin(u)
    z = x*x+y*y
    s = np.sqrt((1-z[0]) / z[1])
    return [x[0], y[0], s*x[1], s*y[1]]

def make_seed(sim, p=0, q=[1, 2, 3, 4], c=[5], composition=None, x=None, y=None, z=None, angle=None, axis=[0, 0, 1], orientation=None, 
              seed_radius=5, q_extra = 5):
    """
    Fairly comprehensive method for adding seed nuclei to a simulation
    
    Parameters
    ----------
    
    sim : pyphasefield.Simulation
        The simulation to add a seed nuclei to
    p : int, default = 0
        Index of the order field ("phi")
    q : list of int, default = [1, 2, 3, 4]
        Indices of the quaternion orientation fields
        Either 0, 2, or 4 long. 0 means no quaternions, 2 is 2D, equivalent to complex rotation, 4 is 3D, full quaternion orientation
    c : list of int, default = [5]
        Indices of the composition fields 
    composition : list of float, default = None
        A list containing the values to set the composition field equal to
        If None, do not set the composition field at all (order-parameter only nucleation)
        If defined, must be the same length as c!
    x : int, optional
        Cell index on x axis to center the seed nuclei. If unspecified, choose a random location
    y : int, optional
        Cell index on y axis to center the seed nuclei. If unspecified, choose a random location
    z : int, optional
        Cell index on z axis to center the seed nuclei. If unspecified, choose a random location
    angle : float, optional
        Angle to rotate about the z-axis (2D) or about the defined axis (3D). 
        If unspecified in 2D, use a random rotation
        Used by default in 2D, or if orientation is not specified and angle is specified in 3D
    axis : list of float, default = [0, 0, 1]
        Axis about which to rotate in 3D. Will be normalized automatically. 
    orientation : list of float, optional
        Quaternion orientation to be used in 3D. If neither this nor angle is specified in 3D, use a random orientation
    seed_radius : int, default = 5
        Radius of the seed nuclei in the order field, in cells.
    q_extra : int, default = 5
        seed_radius+q_extra is the radius of the seed nuclei in the orientation fields 
        (to ensure the nuclei grows in a bubble of defined orientation until the quaternion evolution equations take over)
        
    Notes
    -----
    
    This function will account for parallelism (global vs. local coordinates) automatically. 
    
    """
    global seed_mask_p, seed_mask_q, r_p, r_q, seed_dim
    phi = sim.fields[p]
    shape = phi.data.shape
    gdims = sim._global_dimensions
    if not((seed_radius == r_p) and ((seed_radius+q_extra) == r_q) and (seed_dim == len(shape))):
        make_seed_masks(seed_radius, q_extra, len(shape))
    no_q = False
    q_2d = False
    if(q is None):
        no_q = True
    elif(len(q) == 0):
        no_q = True
    elif(len(q) == 2):
        q_2d = True
        q1 = sim.fields[q[0]]
        q4 = sim.fields[q[1]]
    else:
        q1 = sim.fields[q[0]]
        q2 = sim.fields[q[1]]
        q3 = sim.fields[q[2]]
        q4 = sim.fields[q[3]]
    qrad = seed_radius+q_extra
    if(angle is None):
        angle = 2*np.pi*np.random.rand()-np.pi #random angle between -pi and pi
    coordinates = []
    if(len(shape) > 2):
        if(z is None):
            coordinates.append(int(gdims[len(gdims)-3]*np.random.rand()))
        else:
            coordinates.append(int(z))
    if(len(shape) > 1):
        if(y is None):
            coordinates.append(int(gdims[len(gdims)-2]*np.random.rand()))
        else:
            coordinates.append(int(y))
    if(x is None):
        coordinates.append(int(gdims[len(gdims)-1]*np.random.rand()))
    else:
        coordinates.append(int(x))
    p_slices = []
    q_slices = []
    p_mask_slices = []
    q_mask_slices = []

    ghost_offset = []
    for i in range(len(shape)):
        offset = 1
        if(sim._parallel):
            if not((sim._MPI_array_rank[i] == 0) and (not sim._boundary_conditions_type[i][0] == "PERIODIC")): #if its not the first rank with non-periodic left boundary conditions, use ghost_rows
                offset = sim._ghost_rows
        ghost_offset.append(offset)
    
    for i in range(len(coordinates)):
        center = coordinates[i]-sim._dim_offset[i]+ghost_offset[i]
        
        p_slices.append(slice(max(0, center-r_p), max(0, center+r_p)))
        q_slices.append(slice(max(0, center-r_q), max(0, center+r_q)))
        p_mask_slices.append(slice(max(0, r_p-center), max(0, 2*r_p+shape[i]-r_p-center)))
        q_mask_slices.append(slice(max(0, r_q-center), max(0, 2*r_q+shape[i]-r_q-center)))
    
    p_slices = tuple(p_slices)
    q_slices = tuple(q_slices)
    p_mask_slices = tuple(p_mask_slices)
    q_mask_slices = tuple(q_mask_slices)
    
    phi.data[p_slices][seed_mask_p[p_mask_slices]] = 1
    if not(composition is None):
        for i in range(len(c)):
            sim.fields[c[i]].data[p_slices][seed_mask_p[p_mask_slices]] = composition[i]
    if not(no_q):
        if(q_2d):
            q1.data[q_slices][seed_mask_q[q_mask_slices]] = np.cos(0.5*angle)
            q4.data[q_slices][seed_mask_q[q_mask_slices]] = np.sin(0.5*angle)
        else: #3D case
            if(orientation is None):
                if(angle is None):
                    orientation = random_uniform_quaternion()
                else:
                    axis_magnitude = np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
                    for i in range(3):
                        axis[i] /= axis_magnitude
                    s = np.sin(0.5*angle)
                    orientation = [np.cos(0.5*angle), s*axis[0], s*axis[1], s*axis[2]]
            q1.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[0]
            q2.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[1]
            q3.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[2]
            q4.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[3]
    
    # Convert slices to tuples
    p_slices = tuple(p_slices)
    q_slices = tuple(q_slices)
    p_mask_slices = tuple(p_mask_slices)
    q_mask_slices = tuple(q_mask_slices)
    
    # Apply the seed masks to fields
    phi.data[p_slices][seed_mask_p[p_mask_slices]] = 1
    
    if composition is not None:
        for i in range(len(c)):
            sim.fields[c[i]].data[p_slices][seed_mask_p[p_mask_slices]] = composition[i]
            
    if not(no_q):
        if(q_2d):
            q1.data[q_slices][seed_mask_q[q_mask_slices]] = np.cos(0.5*angle)
            q4.data[q_slices][seed_mask_q[q_mask_slices]] = np.sin(0.5*angle)
        else:  # 3D case
            if(orientation is None):
                if(angle is None):
                    orientation = random_uniform_quaternion()
                else:
                    axis_magnitude = np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
                    for i in range(3):
                        axis[i] /= axis_magnitude
                    s = np.sin(0.5*angle)
                    orientation = [np.cos(0.5*angle), s*axis[0], s*axis[1], s*axis[2]]
            q1.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[0]
            q2.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[1]
            q3.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[2]
            q4.data[q_slices][seed_mask_q[q_mask_slices]] = orientation[3]
            
def TSVtoHDF5(tsv_folder_path, files=None, times=None, file_t_units="us", target_t_units="s", file_x_units="cm", target_x_units="cm",
              cutoff_x = [-np.inf, np.inf], cutoff_y = [-np.inf, np.inf], cutoff_z = [-np.inf, 0.00001], relative_t = False):
    #converts a series of tsv files contained in [tsv_folder_path] into a single time-series HDF5 file
    #assumes entries are in [x, y, z, T] order
    #ignores lines that do not have four entries split by tabs
    #if files and/or times are None, assume all files in the folder are of the format [name]time###, where ### is the time used
    #file_t_units and target_t_units can be "ns", "us", "ms", or "s"
    #file_x_units and target_x_units can be "nm", "um", "mm", "cm", or "m"
    #cutoffs only extract a desired region from the original files, in case region is irrelevant (e.g. gaseous)
    #relative_t flag ensures the first time step is t=0, regardless of its actual absolute time
    sanitized_path = tsv_folder_path.rstrip("/\\")
    files_in_folder = glob(f"{sanitized_path}/*")
    header = None
    for file in files_in_folder:
        if(pathlib.Path(file).suffix == ".hdf5"):
            continue
        else:
            split = file.rsplit("time", 1)
            if(len(split) == 2):
                header = split[0]
                break
    if(header is None):
        print("No files matching that name! Aborting")
        return None
    files = list(set(glob(header+"*")) - set(glob(header+"*.hdf5")))
    
    t_scaling = units_dict[file_t_units]/units_dict[target_t_units]
    x_scaling = units_dict[file_x_units]/units_dict[target_x_units]
    
    xs = []
    ys = []
    zs = []
    ts = []
    for file in files:
        tstr = file.rsplit("time", 1)[1]
        t = float(tstr)
        if not(t in ts):
            ts.append(t)
    ts.sort()
    tstrs = ts.copy()
    for file in files:
        tstr = file.rsplit("time", 1)[1]
        t = float(tstr)
        tstrs[ts.index(t)] = tstr
    
    f = open(files[0], "r")
    l = f.readline()
    while not(len(l.strip()) == 0):
        vals = l.strip("").split("\t")
        if not(len(vals) == 4):
            l = f.readline()
            continue
        xval = float(vals[0])
        yval = float(vals[1])
        zval = float(vals[2])
        l = f.readline()
        if((xval < cutoff_x[0]) or (xval > cutoff_x[1])):
            continue
        if((yval < cutoff_y[0]) or (yval > cutoff_y[1])):
            continue
        if((zval < cutoff_z[0]) or (zval > cutoff_z[1])):
            continue
        if not(xval in xs):
            xs.append(xval)
        if not(yval in ys):
            ys.append(yval)
        if not(zval in zs):
            zs.append(zval)
    xs.sort()
    ys.sort()
    zs.sort()
    f.close()
    
    array = np.zeros([len(ts), len(zs), len(ys), len(xs)])
    times = np.array(ts)
    if(relative_t):
        times -= np.min(times)
    times *= t_scaling
        
    gridsize_F = np.zeros([3])
    gridsize_F[0] = (xs[1]-xs[0])*x_scaling
    gridsize_F[1] = (ys[1]-ys[0])*x_scaling
    gridsize_F[2] = (zs[1]-zs[0])*x_scaling
    
    for i, tstr in enumerate(tstrs):
        fn = files[0].rsplit("time", 1)[0]+"time"+tstr
        f = open(fn, "r")
        l = f.readline()
        while not(len(l.strip()) == 0):
            vals = l.strip("").split("\t")
            l = f.readline()
            if not(len(vals) == 4):
                continue
            xval = float(vals[0])
            yval = float(vals[1])
            zval = float(vals[2])
            Tval = float(vals[3])
            if((xval < cutoff_x[0]) or (xval > cutoff_x[1])):
                continue
            if((yval < cutoff_y[0]) or (yval > cutoff_y[1])):
                continue
            if((zval < cutoff_z[0]) or (zval > cutoff_z[1])):
                continue
            array[i, zs.index(zval), ys.index(yval), xs.index(xval)] = Tval
        f.close()
    
    f = h5py.File(files[0].rsplit("time", 1)[0]+".hdf5", 'w')
    dset = f.create_dataset("data", array.shape, dtype='f')
    dset[...] = array
    dset2 = f.create_dataset("times", times.shape, dtype='f')
    dset2[...] = times
    dset3 = f.create_dataset("gridsize_F", gridsize_F.shape, dtype='f')
    dset3[...] = gridsize_F
    f.close()

def convert_function(function_text):
    """
    Converts a function with array unpacking in a single line to a function 
    with each variable unpacked separately for better GPU performance.
    
    Args:
        function_text (str): The text of the function to convert
        
    Returns:
        str: The converted function text
    """
    # Extract function name and parameter
    function_def_match = re.match(r'def\s+(\w+)\((\w+)\):', function_text)
    if not function_def_match:
        raise ValueError("Could not parse function definition")
    
    function_name = function_def_match.group(1)
    array_param = function_def_match.group(2)
    
    # Extract variable list from first line after definition
    lines = function_text.strip().split('\n')
    if len(lines) < 2:
        raise ValueError("Function is too short, missing body")
    
    var_list_match = re.match(r'\s*\[([\w\s,]+)\]\s*=\s*' + re.escape(array_param), lines[1])
    if not var_list_match:
        raise ValueError("Could not find variable assignment list")
    
    # Get the variable names
    var_names = [v.strip() for v in var_list_match.group(1).split(',')]
    
    # Create new function with unpacked variables
    new_lines = [f"def {function_name}({array_param}):"]
    
    # Add variable assignments
    for i, var in enumerate(var_names):
        new_lines.append(f"    {var} = {array_param}[{i}]")
    
    # Add the rest of the function body (skipping the first line with the list assignment)
    body_lines = lines[2:]
    new_lines.extend(body_lines)
    
    return '\n'.join(new_lines)

def create_cupy_ufunc_from_sympy(sympy_ufunc):
    """
    Creates a cupy ufunc for the GPU from the given sympy ufunc
    """
    if not hasattr(create_cupy_ufunc_from_sympy, "num_funcs"):
        create_cupy_ufunc_from_sympy.num_funcs = 0
    raw_source = sympy_ufunc.__doc__.split("Source code:")[1].strip("\n").split("Imported modules:")[0].strip("\n")
    raw_source = raw_source.replace("log(", "cp.log(")
    raw_source = convert_function(raw_source)
    sc = compile(raw_source, f"<pycgpu_test_function{create_cupy_ufunc_from_sympy.num_funcs}>", "exec")
    exec(sc, globals())
    linecache.cache[f"<pycgpu_test_function{create_cupy_ufunc_from_sympy.num_funcs}>"] = (len(raw_source), None, raw_source.splitlines(True), f"<pycgpu_test_function{create_cupy_ufunc_from_sympy.num_funcs}>")
    create_cupy_ufunc_from_sympy.num_funcs += 1
    cp_ufunc = jit.rawkernel(device=True)(_lambdifygenerated)
    return cp_ufunc

def create_sympy_ufunc_from_tdb(model):
    """
    Creates a sympy ufunc from the given phase/components of the tdb
    """
    expr = sp.parse_expr(str(model.GM))
    syms = expr.free_symbols
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    for sym in syms: 
        if(sym.name == "T"):
            syms.remove(sym)
            syms.append(sym) #explicitly ensure T is always the last symbol, in case there is a phase "Zeta" or something
    sympy_ufunc = sp.lambdify([syms], expr, "math")
    return sympy_ufunc
        
class TDBContainer():
    def __init__(self, tdb_path, phases=None, components=None):
        if(successfully_imported_pycalphad):
            import pycalphad as pyc
            from pycalphad.core.utils import instantiate_models
        else:
            raise Exception("Aborting, pycalphad must be installed for this class to be used")
        self._tdb_path = tdb_path
        self._tdb = pyc.Database(self._tdb_path)
        self._tdb_phases = phases
        if self._tdb_phases is None:
            self._tdb_phases = list(self._tdb.phases)
        self._tdb_components = components
        if self._tdb_components is None:
            self._tdb_components = list(self._tdb.elements)
        self._tdb_phases.sort()
        self._tdb_components.sort()
        self._tdb_cpu_ufuncs = []
        self._tdb_gpu_ufuncs = []
        cupy_enabled = False
        try:
            import cupy as cp
            cupy_enabled = True
        except:
            print("Cannot import numba, therefore cannot create TDB ufuncs built for GPUs")
        models = instantiate_models(self._tdb, self._tdb_components, self._tdb_phases)
        for phase in self._tdb_phases:
            sp_ufunc = create_sympy_ufunc_from_tdb(models[phase])
            self._tdb_cpu_ufuncs.append(sp_ufunc)
            if(cupy_enabled):
                cp_ufunc = create_cp_ufunc_from_sympy(sp_ufunc)
                self._tdb_gpu_ufuncs.append(cp_ufunc)
        