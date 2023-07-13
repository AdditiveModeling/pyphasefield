import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class Field():
    """
    The Field class, which stores data related to one field in the overall simulation
    
    Attributes
    ----------
    
    data : ndarray
        the *full* array of data in the field, including boundary condition cells
        use the method Field.get_cells() to just get the non-boundary data
    name : str
        Name of the field, for use in plotting images and the like
    colormap : Matplotlib formatted Colormap
        The colormap to be used for plotting images
    
    for reference, private attributes are included below
    
    _simulation : Simulation
        pointer back to the Simulation instance
    _slice : Slice object
        internally used method for getting the non-boundary cells of the Field
    _neighbors : list of int
        Used exclusively for parallel simulations
        A list containing 2*D values, D being the number of dimensions of the simulation
        The order for the list is [d0_left, d0_right, d1_left, d1_right, d2_left, d2_right]
        d0, d1, d2 correspond to the first, second, and third dimensions of the simulation (C-ordering! z then y then x)
        Left and right correspond to the boundary at index 0, and the boundary at index (len(dimension)-1)
        Values in the list of 0 or larger correspond to a boundary condition with the given MPI rank
        Value of -1 corresponds to Neumann boundary conditions
        Value of -2 corresponds to Dirichlet boundary conditions
    _slice_sbc_in : list of Slice objects
        Used exclusively for parallel simulations
        A list of 2*D slices, corresponding to the boundary condition cells specified in the _neighbors list
    _slice_sbc_out : list of Slice objects
        Used exclusively for parallel simulations
        Only used for MPI boundaries, these slices correspond to the data that will be sent to neighboring MPI subgrids
    _transfer_in : list of ndarray
        Used exclusively for parallel simulations
        A list of D ndarrays, which are contiguous 1D arrays used for the Send and Recv numpy array functions in mpi4py
        Each dimension can be a different size, each have as many cells as the two corresponding _slice_sbc_in Slice objects
        These are arrays reserved for receiving data from neighboring MPI subgrids
    _transfer_out : list of ndarray
        Used exclusively for parallel simulations
        A list of D ndarrays, which are contiguous 1D arrays used for the Send and Recv numpy array functions in mpi4py
        Each dimension can be a different size, each have as many cells as the two corresponding _slice_sbc_out Slice objects
        These are arrays reserved for sending data to neighboring MPI subgrids
    _shapes : list of tuple
        Corresponds to the shape of the _slice_sbc_in (and out) Slice object for each dimension
        Used for reshaping the 1D flattened transfer arrays back to the "correct" shape in the boundary condition cells of the Field
    
    """
    
    def __init__(self, data=None, simulation=None, name=None, full_grid=False, boundary_field=None, colormap="GnBu"):
        """
        Initialize a Field instance
        
        Parameters
        ----------
        
        data : ndarray
            the ndarray to be held by the field.
        simulation : Simulation
            a pointer back to the Simulation class which the Field is a part of
        name : str
            The name of the Field, for printing results
        full_grid : Bool
            If True, the data array is of the entire domain, and should be split up to only contain the local piece of the subgrid
        boundary_field : Field, optional
            The "main" fields of the simulation have subordinate boundary fields, containing data for non-periodic boundary conditions
        colormap : Matplotlib formatted Colormap
            A colormap for producing plots of the fields, useful in case different colors show data best for different fields
        """
        
        
        self._simulation = simulation
        
        sim = simulation
        #create neighbors list, with any whole number pointing to a core as a neighbor, -1 = neumann bcs, -2 = dirichlet bcs
        self._neighbors = []
        self._sbc_in = []
        self._sbc_out = []
        self._slice = []
        
        for i in range(len(sim.dimensions)):
            self._neighbors.append([])
            self._sbc_in.append([])
            self._sbc_in[i].append([])
            self._sbc_in[i].append([])
            self._sbc_out.append([])
            self._sbc_out[i].append([])
            self._sbc_out[i].append([])
        
        if(sim._parallel):
            #full_grid says the array is of the entire simulation, and should be cut up if the simulation is parallel
            if(full_grid):
                _d = sim.dimensions
                _o = sim._dim_offset
                _s = []
                for i in range(len(_d)):
                    _s.append(slice(_o[i], _o[i]+_d[i]))
                data = data[tuple(_s)]
            
            #fill neighbors array, set size of field including boundary cells, and make array slices for center and bcs
            
            dim = list(data.shape)
            offset_quantity = sim._MPI_size
            
            for i in range(len(dim)):
                
                offset_quantity //= sim._MPI_array_size[i]
                if(sim._MPI_array_rank[i] == 0):
                    rows, core = self._get_bc_index(i, 0)
                    self._neighbors[i].append(core)
                    i1 = rows
                else:
                    self._neighbors[i].append(sim._MPI_rank-offset_quantity)
                    i1 = sim._ghost_rows
                dim[i] += i1
                if(sim._MPI_array_rank[i] == sim._MPI_array_size[i]-1):
                    rows, core = self._get_bc_index(i, 1)
                    self._neighbors[i].append(core)
                    i2 = rows
                else:
                    self._neighbors[i].append(sim._MPI_rank+offset_quantity)
                    i2 = sim._ghost_rows
                dim[i] += i2
                self._slice.append(slice(i1, -i2))
                for j in range(len(dim)):
                    if(i == j):
                        self._sbc_in[j][0].append(slice(0, i1))
                        self._sbc_in[j][1].append(slice(-i2, None))
                        self._sbc_out[j][0].append(slice(i1, 2*i1))
                        self._sbc_out[j][1].append(slice(-2*i2, -i2))
                    else:
                        self._sbc_in[j][0].append(slice(None))
                        self._sbc_in[j][1].append(slice(None))
                        self._sbc_out[j][0].append(slice(None))
                        self._sbc_out[j][1].append(slice(None))
            self._slice = tuple(self._slice)
            for i in range(len(self._sbc_in)):
                for j in range(2):
                    self._sbc_in[i][j] = tuple(self._sbc_in[i][j])
                    self._sbc_out[i][j] = tuple(self._sbc_out[i][j])
                    
            #create flattened bc arrays, as well as a shapes list for the ndarray.shape of each bc
            self._transfer_in = []
            self._transfer_out = []
            self._shapes = []
            for i in range(len(dim)):
                shape = dim.copy()
                shape[i] = sim._ghost_rows
                self._transfer_in.append(np.zeros(shape).flatten())
                self._transfer_out.append(np.zeros(shape).flatten())
                self._shapes.append(shape)
            #assume uniformly ordered for now - revisit later if order affects speed!
            
        else: #serial
            dim = list(data.shape)
            for i in range(len(dim)):
                dim[i] += 2
                self._slice.append(slice(1, -1)) #build dynamic slice tuple for access to center cells
            self._slice = tuple(self._slice)
            sim = self._simulation
            bcs = sim._boundary_conditions_type
            for i in range(len(dim)):
                for j in range(len(dim)):
                    if(i == j):
                        self._sbc_in[j][0].append(slice(0, 1))
                        self._sbc_in[j][1].append(slice(-1, None))
                        self._sbc_out[j][0].append(slice(1, 2))
                        self._sbc_out[j][1].append(slice(-2, -1))
                    else:
                        self._sbc_in[j][0].append(slice(None))
                        self._sbc_in[j][1].append(slice(None))
                        self._sbc_out[j][0].append(slice(None))
                        self._sbc_out[j][1].append(slice(None))
            for i in range(len(self._sbc_in)):
                for j in range(2):
                    self._sbc_in[i][j] = tuple(self._sbc_in[i][j])
                    self._sbc_out[i][j] = tuple(self._sbc_out[i][j])
            for i in range(len(bcs)):
                for bc2 in bcs[i]:
                    if (bc2 == "DIRICHLET"):
                        self._neighbors[i].append(-2)
                    elif(bc2 == "NEUMANN"):
                        self._neighbors[i].append(-1)
                    else:
                        self._neighbors[i].append(0)
        
        fullarray = np.zeros(dim)
        fullarray[self._slice] += data
        self.data = fullarray
        self.name = name
        self.colormap = colormap
        
    def __setitem__(self, key, value):
        dims = self.data.shape
        set_slice = []
        if(type(key) is int):
            set_slice.append(self._get_local_index(key, 0))
        if(type(key) is slice):
            set_slice.append(self._get_local_slice(key, 0))
        else: #list or tuple?
            for i in range(len(key)):
                if(type(key[i]) is int):
                    set_slice.append(self._get_local_index(key[i], i))
                elif(type(key[i]) is slice):
                    set_slice.append(self._get_local_slice(key[i], i))
        for i in range(len(set_slice), len(dims)):
            set_slice.append(slice(None))
        set_slice = tuple(set_slice)
        self.data[self._slice][set_slice] = value
        
    def _get_local_slice(self, _slice, direction):
        offset = self._simulation._dim_offset[direction]
        start = _slice.start-offset
        if(start < 0):
            start = 0
        stop = _slice.stop-offset
        if(stop < 0):
            stop = 0
        return slice(start, stop)
        
    def _get_local_index(self, _int, direction):
        offset = self._simulation._dim_offset[direction]
        _int -= offset
        if(_int < 0):
            return slice(0, 0, 1) #nothing slice, ensures assignments outside the area do nothing
        return slice(_int, _int+1)
       
    def _get_bc_index(self, index, lor):
        sim = self._simulation
        bcs = sim._boundary_conditions_type
        bc_type = bcs[index][lor]
            
        if(bc_type == "DIRICHLET"):
            return 1, -2
        elif(bc_type == "NEUMANN"):
            return 1, -1
        else: #periodic, actual thinking required
            loc = sim._MPI_rank
            if(index == 0): 
                offset = sim._MPI_size - sim._MPI_size//sim._MPI_array_size[0]
            elif(index == 1):
                offset = sim._MPI_size//sim._MPI_array_size[0] - sim._MPI_size//sim._MPI_array_size[0]//sim._MPI_array_size[1]
            else:
                offset = sim._MPI_size//sim._MPI_array_size[0]//sim._MPI_array_size[1] - 1
            if(lor == 1):
                offset *= -1
            return sim._ghost_rows, loc+offset
        
    def _swap(self):
        """
        Method to exchange boundary cells between fields, to enable parallelism
        
        *Should* only be run if the simulation is parallel, but there's no explicit check for this
        """
        
        ghosts = self._simulation._ghost_rows
        for i in range(len(self.data.shape)):
            left = self._neighbors[i][0]
            right = self._neighbors[i][1]
            comm = self._simulation._MPI_COMM_WORLD
            rank = self._simulation._MPI_rank
            
            #first, send left border cells to left neighbor, and receive from right neighbor
            if(left > -1):
                self._transfer_out[i][:] = self.data[self._sbc_out[i][0]].flatten()
                reqs1 = comm.Isend(self._transfer_out[i], dest=left, tag=rank)
            if(right > -1):
                comm.Recv(self._transfer_in[i], source=right, tag=right)
                self.data[self._sbc_in[i][1]] = self._transfer_in[i].reshape(self._shapes[i])
            if(left > -1):
                reqs1.wait()
                
            #then, send right border cells to right neighbor, and receive from left neighbor
            if(right > -1):
                self._transfer_out[i][:] = self.data[self._sbc_out[i][1]].flatten()
                reqs2 = comm.Isend(self._transfer_out[i], dest=right, tag=rank)
            if(left > -1):
                comm.Recv(self._transfer_in[i], source=left, tag=left)
                self.data[self._sbc_in[i][0]] = self._transfer_in[i].reshape(self._shapes[i])
            if(right > -1):
                reqs2.wait()
        

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