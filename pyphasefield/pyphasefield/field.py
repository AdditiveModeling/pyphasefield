import numpy as np

defaultsim = ""

class Field(np.ndarray):
    
    #Modified subclass of ndarray, mostly following the guide from:
    #https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    
    def __new__(cls, input_array, name="phi", field_type="scalar", simulation=None):
        from .simulation import Simulation
        if simulation is None:
            defaultsim = Simulation("")
            simulation = defaultsim
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._name = name
        obj._field_type = field_type
        obj._simulation = simulation
        
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self._name = getattr(obj, '_name', "phi")
        self._field_type = getattr(obj, '_field_type', "scalar")
        self._simulation = getattr(obj, '_simulation', defaultsim)
        
    def gradient_cell(self):
        number_of_dimensions = len(self.shape)
        inverse_cell_spacing = 1./(self._simulation.get_cell_spacing())
        inverse_2dx = 0.5*inverse_cell_spacing
        array = (np.roll(self, -1, 0) - np.roll(self, 1, 0))*inverse_2dx
        array = array[np.newaxis] #add dimension for different partial derivatives
        for i in range(1, number_of_dimensions):
            temp = (np.roll(self, -1, i) - np.roll(self, 1, i))*inverse_2dx
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array
        
    def gradient_face_left(self):
        number_of_dimensions = len(self.shape)
        inverse_cell_spacing = 1./(self._simulation.get_cell_spacing())
        array = (self - np.roll(self, 1, 0))*inverse_cell_spacing
        array = array[np.newaxis] #add dimension for different partial derivatives
        for i in range(1, number_of_dimensions):
            temp = (self - np.roll(self, 1, i))*inverse_cell_spacing
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array
    
    def gradient_face_right(self):
        number_of_dimensions = len(self.shape)
        inverse_cell_spacing = 1./(self._simulation.get_cell_spacing())
        array = (np.roll(self, -1, 0) - self)*inverse_cell_spacing
        array = array[np.newaxis] #add dimension for different partial derivatives
        for i in range(1, number_of_dimensions):
            temp = (np.roll(self, -1, i) - self)*inverse_cell_spacing
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array
        
    def laplacian(self):
        number_of_dimensions = len(self.shape)
        inverse_cell_spacing = 1./(self._simulation.get_cell_spacing())
        inverse_dx_squared = inverse_cell_spacing**2
        result = (np.roll(self, -1, 0) + np.roll(self, 1, 0) - 2*self)*inverse_dx_squared
        for i in range(1, number_of_dimensions):
            result += (np.roll(self, -1, i) + np.roll(self, 1, i) - 2*self)*inverse_dx_squared
        return result
    
    def get_nonboundary_cells(self, boundary_conditions_type):
        #for periodic, this is simply the following
        return self