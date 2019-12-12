import numpy as np

class Field(np.ndarray):
    
    #Modified subclass of ndarray, mostly following the guide from:
    #https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    
    def __new__(cls, input_array, name="phi", field_type="scalar", cell_spacing=0.0000046):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._name = name
        obj._field_type = field_type
        obj._cell_spacing = cell_spacing
        obj._dimensions_of_simulation_region = input_array.shape
        obj._number_of_dimensions = len(input_array.shape)
        
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self._name = getattr(obj, '_name', "phi")
        self._field_type = getattr(obj, '_field_type', "scalar")
        self._cell_spacing = getattr(obj, '_cell_spacing', 0.0000046)
        self._dimensions_of_simulation_region = getattr(obj, '_dimensions_of_simulation_region', [200,200])
        self._number_of_dimensions = getattr(obj, '_number_of_dimensions', 2)
        
    def gradient_cell(self):
        array = (np.roll(self, -1, 0) - np.roll(self, 1, 0))/(2*self._cell_spacing)
        array = array[np.newaxis] #add dimension for different partial derivatives
        for i in range(1, self._number_of_dimensions):
            temp = (np.roll(self, -1, i) - np.roll(self, 1, i))/(2*self._cell_spacing)
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array
        
    def gradient_face_left(self):
        array = (self - np.roll(self, 1, 0))/(self._cell_spacing)
        array = array[np.newaxis] #add dimension for different partial derivatives
        for i in range(1, self._number_of_dimensions):
            temp = (self - np.roll(self, 1, i))/(self._cell_spacing)
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array
    
    def gradient_face_right(self):
        array = (np.roll(self, -1, 0) - self)/(self._cell_spacing)
        array = array[np.newaxis] #add dimension for different partial derivatives
        for i in range(1, self._number_of_dimensions):
            temp = (np.roll(self, -1, i) - self)/(self._cell_spacing)
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array
        
    def laplacian(self):
        result = (np.roll(self, -1, 0) + np.roll(self, 1, 0) - 2*self)/(self._cell_spacing**2)
        for i in range(1, self._number_of_dimensions):
            result += (np.roll(self, -1, i) + np.roll(self, 1, i) - 2*self)/(self._cell_spacing**2)
        return result
    
    def get_nonboundary_cells(self, boundary_conditions_type):
        #for periodic, this is simply the following
        return self