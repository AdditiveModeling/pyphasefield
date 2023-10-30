from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import meshio
import numpy as np
import sympy as sp
import symengine as se
from tinydb import where

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

def successfully_imported_pycalphad():
    """
    Checks if pycalphad is installed. 
    If not, warns the user that pycalphad-dependent features cannot be used
    Also tells the user how to install it (if the user has Anaconda)
    """
    try:
        import pycalphad as pyc
    except ImportError:
        print("The feature you are trying to use requires pycalphad")
        print("In Anaconda, use \'conda install -c pycalphad -c conda-forge pycalphad\' to install it")
        return False
    return True

def successfully_imported_numba():
    """
    Checks if numba/cuda is installed. 
    If not, warns the user that gpu-dependent features cannot be used
    Also tells the user how to install it (if the user has Anaconda)
    """
    try:
        import numba
        from numba import cuda
        from . import ppf_gpu_utils
    except ImportError as e:
        print("The feature you are trying to use requires numba (and cudatoolkit)")
        print("In Anaconda, use \'conda install cudatoolkit\' and \'conda install numba\' to install them")
        print(e)
        return False
    return True

def make_seed_masks(r, q_extra, ndims):
    global seed_mask_p, seed_mask_q, r_p, r_q
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
    global seed_mask_p, seed_mask_q, r_p, r_q
    phi = sim.fields[p]
    shape = phi.data.shape
    gdims = sim._global_dimensions
    if not((seed_radius == r_p) and ((seed_radius+q_extra) == r_q)):
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
    
    for i in range(len(coordinates)):
        center = coordinates[i]-sim._dim_offset[i]
        
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
                    

def CSVtoXDMF(csv_path, T_cutoffs=False, starting_T=None, ending_T=None, reflect_X = False):
    try:
        f = open(csv_path)
        s = f.readline() #header
        s = f.readline() #origins
        s = f.readline() #value of origins
        s = f.readline() #spacings
        s = f.readline() #value of spacings
        s = f.readline() #numpoints
        s = f.readline() #value of numpoints
        dims = [int(item) for item in filter(None, s.strip('\n').strip(' ').split(","))]
        print(dims)
        s = f.readline() #time, temperature header
        s = f.readline() #first value for time, temperature array!
        points = np.array([[[0, 0],[0, 0]], [[0, 0],[0, 0]]])
        cells = {}
        reached_first_time = False
        time_offset = 0
        with meshio.xdmf.TimeSeriesWriter("T.xdmf") as writer:
            writer.write_points_cells(points, cells)
            while(s):
                s = s.split(",", 1)
                time = float(s[0])*0.000001
                #reshape T to dims.reverse(), due to ordering of array (last term is num_cols)
                l = s[1].strip('\n').split(',')
                T = np.transpose(1000*np.resize(np.array([float(item) for item in filter(None, l)]), dims))
                print(time, np.min(T), np.max(T))
                if(T_cutoffs):
                    if not reached_first_time: #find first timestep where temperature is above starting_T
                        if(np.max(T) > starting_T):
                            reached_first_time = True
                            time_offset = time
                    else:
                        if(np.max(T) < ending_T):
                            break
                    if(reflect_X):
                        T = np.concatenate((np.flip(T, 1), T), axis=1)
                    if(reached_first_time):
                        writer.write_data(time-time_offset, point_data={"T": T})
                    s = f.readline()
                else:
                    if(reflect_X):
                        T = np.concatenate((np.flip(T, 1), T), axis=1)
                    writer.write_data(time, point_data={"T": T})
                    s = f.readline()

    finally:
        f.close()
        
def create_sympy_ufunc_from_tdb(tdb, phase, components, mode):
    """
    Creates a sympy ufunc from the given phase/components of the tdb
    """
    if(successfully_imported_pycalphad):
        import pycalphad as pyc
    else:
        raise Exception("Aborting, pycalphad must be installed for this class to be used")
    phase_id = phase
    phase = tdb.phases[phase_id]
    param_search = tdb.search
    g_param_query = (
        (where('phase_name') == phase.name) & \
        ((where('parameter_type') == 'G') | \
        (where('parameter_type') == 'L'))
    )
    model = pyc.Model(tdb, components, phase_id)
    symengine_expr = model.redlich_kister_sum(phase, param_search, g_param_query)
    symengine_ime = model.ideal_mixing_energy(tdb)
    
    for i in tdb.symbols:
        d = tdb.symbols[i]
        g = se.Symbol(i)
        symengine_expr = symengine_expr.subs(g, d)
        
    #do it again, just in case symbols are defined in terms of symbols
    #fix this later - detect how many symbols are in expression?
    for i in tdb.symbols:
        d = tdb.symbols[i]
        g = se.Symbol(i)
        symengine_expr = symengine_expr.subs(g, d)
        
    #have to parse the sympy expression from the symengine expression, sympy doesn't like the variables with "()," symbols
    
    sympy_expr = sp.parse_expr(str(symengine_expr))
    sympy_ime = sp.parse_expr(str(symengine_ime))
    
    sympysyms_list = []
    T = None
    symbol_name_list = []
    for i in list(components):
        symbol_name_list.append(phase_id+"0"+i)

    for j in sympy_expr.free_symbols:
        if j.name in symbol_name_list: 
            sympysyms_list.append(j)
        elif j.name == "T":
            T = j
        else:
            symengine_expr = symengine_expr.subs(j, 0)
    sympysyms_list = sorted(sympysyms_list, key=lambda t:t.name)
    sympysyms_list.append(T)
    
    sympy_ufunc = sp.lambdify([sympysyms_list], sympy_expr+sympy_ime, mode)
    return sympy_ufunc

def create_numba_ufunc_from_sympy(sp_ufunc):
    """
    Converts sympy ufunc to numba
    """
    try:
        import numba
    except:
        print("Cannot import numba, therefore cannot create TDB ufuncs built for GPUs")
    numba_ufunc = numba.jit(sp_ufunc, nopython=True)
    return numba_ufunc
        
class XDMFLoader():
    def __init__(self, t_file_path):
        self.t_file_path = t_file_path
        with meshio.xdmf.TimeSeriesReader(self.t_file_path) as reader:
            self.num_steps = reader.num_steps
            
    def data(self, step):
        with meshio.xdmf.TimeSeriesReader(self.t_file_path) as reader:
            points, cells = reader.read_points_cells()
            bound, point_data0, cell_data0 = reader.read_data(step)
            array = np.squeeze(point_data0['T'])
            return bound, array
        
class TDBContainer():
    def __init__(self, tdb_path, phases=None, components=None):
        if(successfully_imported_pycalphad):
            import pycalphad as pyc
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
        numba_enabled = False
        try:
            import numba
            numba_enabled = True
        except:
            print("Cannot import numba, therefore cannot create TDB ufuncs built for GPUs")
        for k in range(len(self._tdb_phases)):
            
            sp_ufunc_numpy = create_sympy_ufunc_from_tdb(self._tdb, self._tdb_phases[k], components, 'numpy')
            self._tdb_cpu_ufuncs.append(sp_ufunc_numpy)
            if(numba_enabled):
                sp_ufunc_math = create_sympy_ufunc_from_tdb(self._tdb, self._tdb_phases[k], components, 'math')
                nb_ufunc = create_numba_ufunc_from_sympy(sp_ufunc_math)
                self._tdb_gpu_ufuncs.append(nb_ufunc)
        