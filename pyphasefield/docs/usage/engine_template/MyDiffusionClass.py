import numpy as np
try:
    #import from within Engines folder
    from ..field import Field
    from ..simulation import Simulation
    from ..ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
except:
    try:
        #import classes from pyphasefield library
        from pyphasefield.field import Field
        from pyphasefield.simulation import Simulation
        from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
    except:
        raise ImportError("Cannot import from pyphasefield library!")

class MyDiffusionClass(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #additional initialization code goes below
        #runs *before* tdb, thermal, fields, and boundary conditions are loaded/initialized

    def init_tdb_params(self):
        super().init_tdb_params()
        #additional tdb-related code goes below
        #runs *after* tdb file is loaded, tdb_phases and tdb_components are initialized
        #runs *before* thermal, fields, and boundary conditions are loaded/initialized

    def init_fields(self):
        #initialization of fields code goes here
        #runs *after* tdb and thermal data is loaded/initialized
        #runs *before* boundary conditions are initialized
        if not ("D" in self.user_data):
            self.user_data["D"] = 0.1
        dim = self.dimensions
        c = np.zeros(dim)
        length = dim[0]
        width = dim[1]
        c[length // 4:3 * length // 4, width // 4:3 * width // 4] = 1
        self.add_field(c, "c")

    def initialize_fields_and_imported_data(self):
        super().initialize_fields_and_imported_data()
        #initialization of fields/imported data goes below
        #runs *after* tdb, thermal, fields, and boundary conditions are loaded/initialized

    def just_before_simulating(self):
        super().just_before_simulating()
        #additional code to run just before beginning the simulation goes below
        #runs immediately before simulating, no manual changes permitted to changes implemented here

    def simulation_loop(self):
        #code to run each simulation step goes here
        c = self.fields[0].data
        D = self.user_data["D"]
        dx = self.dx
        dt = self.dt

        #define offset arrays, remember the sign of roll is opposite the direction of the cell of interest
        #also, x is dimension 1, y is dimension 0 (C style arrays...)
        c_p0 = np.roll(c, -1, 1) #x+1, y. 
        c_m0 = np.roll(c, 1, 1) #x-1, y. 
        c_0p = np.roll(c, -1, 0) #x, y+1. 
        c_0m = np.roll(c, 1, 0) #x, y-1. 

        #apply change from a single step
        c += D*dt*(c_p0 + c_m0 + c_0p + c_0m - 4*c)/(dx**2)