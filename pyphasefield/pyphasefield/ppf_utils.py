from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

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