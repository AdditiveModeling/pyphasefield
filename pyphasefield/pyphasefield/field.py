import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class Field():
    
    def __init__(self, data=None, simulation=None, name=None, colormap="GnBu"):
        #add boundary cells
        dim = list(data.shape)
        self._slice = []
        for i in range(len(dim)):
            dim[i] += 2
            self._slice.append(slice(1, -1)) #build dynamic slice tuple for access to center cells
        self._slice = tuple(self._slice)
        fullarray = None
        fullarray = np.zeros(dim)
        fullarray[self._slice] += data
        self.data = fullarray
        self.name = name
        self._simulation = simulation
        self.colormap = colormap
        

    def __str__(self):
        return self.data.__str__()

    def gradient_cell(self):
        number_of_dimensions = len(self.data.shape)
        inverse_cell_spacing = 1. / (self._simulation.get_cell_spacing())
        inverse_2dx = 0.5 * inverse_cell_spacing
        array = (np.roll(self.data, -1, 0) - np.roll(self.data, 1, 0)) * inverse_2dx
        array = array[np.newaxis]  # add dimension for different partial derivatives
        for i in range(1, number_of_dimensions):
            temp = (np.roll(self.data, -1, i) - np.roll(self.data, 1, i)) * inverse_2dx
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array

    def gradient_face_left(self):
        number_of_dimensions = len(self.data.shape)
        inverse_cell_spacing = 1. / (self._simulation.get_cell_spacing())
        array = (self.data - np.roll(self.data, 1, 0)) * inverse_cell_spacing
        array = array[np.newaxis]  # add dimension for different partial derivatives
        for i in range(1, number_of_dimensions):
            temp = (self.data - np.roll(self.data, 1, i)) * inverse_cell_spacing
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array

    def gradient_face_right(self):
        number_of_dimensions = len(self.data.shape)
        inverse_cell_spacing = 1. / (self._simulation.get_cell_spacing())
        array = (np.roll(self.data, -1, 0) - self.data) * inverse_cell_spacing
        array = array[np.newaxis]  # add dimension for different partial derivatives
        for i in range(1, number_of_dimensions):
            temp = (np.roll(self.data, -1, i) - self.data) * inverse_cell_spacing
            temp = temp[np.newaxis]
            array = np.concatenate((array, temp), axis=0)
        return array

    def laplacian(self):
        number_of_dimensions = len(self.data.shape)
        inverse_cell_spacing = 1. / (self._simulation.get_cell_spacing())
        inverse_dx_squared = inverse_cell_spacing ** 2
        result = (np.roll(self.data, -1, 0) + np.roll(self.data, 1, 0) - 2 * self.data) * inverse_dx_squared
        for i in range(1, number_of_dimensions):
            result += (np.roll(self.data, -1, i) + np.roll(self.data, 1, i) - 2 * self.data) * inverse_dx_squared
        return result

    def get_cells(self):
        return self.data[self._slice]
    
    def get_all_cells(self):
        return self.data