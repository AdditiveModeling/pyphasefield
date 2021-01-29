from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import meshio
import numpy as np

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