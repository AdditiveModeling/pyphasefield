from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import meshio
import numpy as np
import sympy as sp
from tinydb import where

colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
COLORMAP_OTHER = LinearSegmentedColormap.from_list('rgb', colors)
colors2 = [(1, 0, 0), (1, 1, 0), (0, 0, 1)]
COLORMAP_PHASE = LinearSegmentedColormap.from_list('rgb', colors2)
colors2 = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]
COLORMAP_PHASE_INV = LinearSegmentedColormap.from_list('rgb', colors2)

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

def make_seed(phi, q1, q4, x, y, angle, seed_radius):
    shape = phi.shape
    qrad = seed_radius+5
    x_size = shape[1]
    y_size = shape[0]
    for i in range((int)(y-seed_radius), (int)(y+seed_radius)):
        for j in range((int)(x-seed_radius), (int)(x+seed_radius)):
            if((i-y)*(i-y)+(j-x)*(j-x) < (seed_radius**2)):
                phi[i%y_size][j%x_size] = 1
    for i in range((int)(y-qrad), (int)(y+qrad)):
        for j in range((int)(x-qrad), (int)(x+qrad)):
            if((i-y)*(i-y)+(j-x)*(j-x) < (qrad**2)):
                #angle is halved because that is how quaternions do
                q1[i%y_size][j%x_size] = np.cos(0.5*angle)
                q4[i%y_size][j%x_size] = np.sin(0.5*angle)
    return phi, q1, q4

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
        param_search = self._tdb.search
        numba_enabled = False
        try:
            import numba
            numba_enabled = True
        except:
            print("Cannot import numba, therefore cannot create TDB ufuncs built for GPUs")
        for k in range(len(self._tdb_phases)):
            phase_id = self._tdb_phases[k]
            phase = self._tdb.phases[phase_id]
            g_param_query = (
                (where('phase_name') == phase.name) & \
                ((where('parameter_type') == 'G') | \
                (where('parameter_type') == 'L'))
            )
            model = pyc.Model(self._tdb, self._tdb_components, phase_id)
            sympyexpr = model.redlich_kister_sum(phase, param_search, g_param_query)
            ime = model.ideal_mixing_energy(self._tdb)
            
            for i in self._tdb.symbols:
                d = self._tdb.symbols[i]
                g = sp.Symbol(i)
                sympyexpr = sympyexpr.subs(g, d)

            sympysyms_list = []
            T = None
            symbol_name_list = []
            for i in list(self._tdb_components):
                symbol_name_list.append(phase_id+"0"+i)
                
            for j in sympyexpr.free_symbols:
                if j.name in symbol_name_list: 
                    #this may need additional work for phases with sublattices...
                    sympysyms_list.append(j)
                elif j.name == "T":
                    T = j
                else:
                    sympyexpr = sympyexpr.subs(j, 0)
            sympysyms_list = sorted(sympysyms_list, key=lambda t:t.name)
            sympysyms_list.append(T)
            
            self._tdb_cpu_ufuncs.append(sp.lambdify([sympysyms_list], sympyexpr+ime, 'numpy'))
            if(numba_enabled):
                self._tdb_gpu_ufuncs.append(numba.jit(sp.lambdify([sympysyms_list], sympyexpr+ime, 'math')))
        