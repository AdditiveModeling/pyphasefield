try:
    import warnings
    #in case Numba 0.57.0, filter these warnings
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    from .field import Field
    from .simulation import Simulation
    from .ppf_utils import *


except Exception as error:
    #probably not all dependencies are installed (installed pyphasefield
    import sys
    not_running_as_main = True
    try:
        if(sys.argv[0] == "-m"):
            not_running_as_main = False
    except:
        pass
    if(not_running_as_main):
        print("Not all required packages for pyphasefield are installed yet!")
        print("Run this module ('python -m pyphasefield') to install all packages (optionally including cuda, mpi, pycalphad)")
    print("Full error message below:")
    print(error)
