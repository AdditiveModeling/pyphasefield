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
        #runs *before* boundary conditions are initialized
        pass
        
        
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
        pass