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

class Template(Simulation):
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
        #boundary conditions are created simulatenously through the use of the add_field method
        pass
                        
    def just_before_simulating(self):
        super().just_before_simulating()
        #additional code to run just before beginning the simulation goes below
        #runs immediately before simulating, no manual changes permitted to changes implemented here
        
    def simulation_loop(self):
        #code to run each simulation step goes here
        pass