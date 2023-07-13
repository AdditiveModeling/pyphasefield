import numpy as np
import warnings

def region2d(dims, cores):
    """
    Divide a 2D region into <cores> subregions. These regions have roughly minimum surface area
    
    Returns two lists, containing the dimensions of each subgrid based on x and y index, respectively
    Some dimensions will differ with index, to account for remainders of say dividing dims_x with cores_x
    E.g. if a dimension is 23 cells long, and has 4 subgrids along that dimension, they will have lengths 
        along that dimension equal to 6, 6, 6, and 5 respectively (the list = [6,6,6,5] in this case)
    
    Parameters
    ----------
    
    dims : ndarray or list
        The dimensions of the full domain, should be 2D. E.g. [100, 150] for a simulation 100 tall, 150 wide (C order)
    cores : int
        Number of cores to split the problem up between. Cores can be #gpus or #cpus for cpu-only computing
        
    Returns
    -------
    
    x_dims : list
        list of widths of each subgrid, based on x index
        E.g. for 150 wide domain, and 4 subgrids wide, this should return [38, 38, 37, 37]
    y_dims : list
        list of heights of each subgrid, based on y index
        E.g. for 100 tall domain, and 3 subgrids tall, this should return [34, 33, 33]
        
    Warnings
    --------
    
    If the number of cores is not highly divisible, this will warn the user that performance may suffer
    E.g. a [100, 150] domain with 24 cores can be split into [25, 25] subgrids, but 23 cores have sections of size [100, 6]
    With 5 ghost cells on each side, [25, 25] becomes [35, 35], about 2 times larger
    However, [100, 6] becomes [110, 16], nearly 3 times larger
    This scaling gets much worse in 3d, 1000 cores on a 1000^3 domain can either make [100, 100, 100] regions, or
        it can make [1000, 1000, 1] regions. 5 ghost cells would either make a negligible difference, or ~11x more data!
    Warns the user if the ratio in dimensions is larger than 4
    """
    
    x_dims = []
    y_dims = []
    
    #check if cores == 1, will break the below algorithm since it doesn't have any factors!
    #shouldn't need to be caught, a process with 1 core should default to cpu_serial or gpu_serial!
    if(cores == 1):
        x_dims.append(dims[0])
        y_dims.append(dims[1])
        return x_dims, y_dims
    
    #factorize cores inefficiently, this shouldn't take long *if the user doesn't use a prime number of cores*
    i = 2
    index = 0
    new_factor = True
    factors = []
    count_factors = []
    c = cores
    while i <= c:
        if(c%i == 0):
            if(new_factor):
                count_factors.append(0)
                factors.append(i)
                new_factor = False
            count_factors[len(count_factors)-1] += 1
            c /= i
        else:
            i += 1
            new_factor = True
            
    #brute force all possible combinations of factors
    #takes a few milliseconds for even large numbers of cores, so not worth optimizing
    indices = []
    arrays = []
    array = np.array([1, 1])
    for i in range(len(factors)):
        indices.append(0)
        arrays.append([])
        for j in range(count_factors[i]+1):
            a = array.copy()
            a[0] *= factors[i]**j
            a[1] *= factors[i]**(count_factors[i]-j)
            arrays[i].append(a)
    ratio = 2**32
    best = None
    while(indices[len(indices)-1] < (count_factors[len(indices)-1]+1)):
        a = array.copy()
        for i in range(len(indices)):
            a *= arrays[i][indices[i]]
        lr = (dims[0]/a[0])/(dims[1]/a[1])
        if(lr < 1):
            lr = 1./lr
        if(lr < ratio):
            ratio = lr
            best = a
        indices[0] += 1
        for i in range(len(indices)-1):
            if(indices[i] == (count_factors[i]+1)):
                indices[i] = 0
                indices[i+1] += 1
    if(ratio > 4):
        warnings.warn("Array is ["+str(dims[0]//best[0])+", "+str(dims[1]//best[1])+"], with a ratio of "+str(ratio)+"\n"+
                      "This size ratio is larger than 4, and could lead to poor performance!\n"+
                      "For number of cores, consider using a highly composite number")
    x = best[0]
    y = best[1]
    for i in range(x):
        x_dims.append(dims[0]//x)
    for i in range(dims[0]-(x*(dims[0]//x))):
        x_dims[i] += 1
    for i in range(y):
        y_dims.append(dims[1]//y)
    for i in range(dims[1]-(y*(dims[1]//y))):
        y_dims[i] += 1
    
    return x_dims, y_dims
    
def region3d(dims, cores):
    """
    Divide a 3D region into <cores> regions. These regions have roughly minimum surface area
    
    Returns three lists, containing the dimensions of each subgrid based on x, y, and z index, respectively 
    Some dimensions will differ with index, to account for remainders of say dividing dims_x with cores_x
    E.g. if a dimension is 23 cells long, and has 4 subgrids along that dimension, they will have lengths 
        along that dimension equal to 6, 6, 6, and 5 respectively (the list = [6,6,6,5] in this case)
    
    Parameters
    ----------
    
    dims : ndarray or list
        The dimensions of the full domain, should be 2D. E.g. [100, 150] for a simulation 100 tall, 150 wide (C order)
    cores : int
        Number of cores to split the problem up between. Cores can be #gpus or #cpus for cpu-only computing
        
    Returns
    -------
    
    x_dims : list
        list of widths of each subgrid, based on x index
        E.g. for 150 wide domain, and 4 subgrids wide, this should return [38, 38, 37, 37]
    y_dims : list
        list of heights of each subgrid, based on y index
        E.g. for 100 tall domain, and 3 subgrids tall, this should return [34, 33, 33]
        
    Warnings
    --------
    
    If the number of cores is not highly divisible, this will warn the user that performance may suffer
    E.g. a [100, 150] domain with 24 cores can be split into [25, 25] subgrids, but 23 cores have sections of size [100, 6]
    With 5 ghost cells on each side, [25, 25] becomes [35, 35], about 2 times larger
    However, [100, 6] becomes [110, 16], nearly 3 times larger
    This scaling gets much worse in 3d, 1000 cores on a 1000^3 domain can either make [100, 100, 100] regions, or
        it can make [1000, 1000, 1] regions. 5 ghost cells would either make a negligible difference, or ~11x more data!
    Warns the user if the largest ratio in dimensions is larger than 4
    """
    
    x_dims = []
    y_dims = []
    z_dims = []
    
    #check if cores == 1, will break the below algorithm since it doesn't have any factors!
    #shouldn't need to be caught, a process with 1 core should default to cpu_serial or gpu_serial!
    if(cores == 1):
        x_dims.append(dims[0])
        y_dims.append(dims[1])
        z_dims.append(dims[2])
        return x_dims, y_dims, z_dims
    
    #factorize cores inefficiently, this shouldn't take long *if the user doesn't use a prime number of cores*
    i = 2
    index = 0
    new_factor = True
    factors = []
    count_factors = []
    c = cores
    while i <= c:
        if(c%i == 0):
            if(new_factor):
                count_factors.append(0)
                factors.append(i)
                new_factor = False
            count_factors[len(count_factors)-1] += 1
            c /= i
        else:
            i += 1
            new_factor = True
            
    #brute force all possible combinations of factors
    #takes a few milliseconds for even large numbers of factors, so not worth optimizing
    indices = []
    arrays = []
    array = np.array([1, 1, 1])
    for i in range(len(factors)):
        indices.append(0)
        arrays.append([])
        for j in range(count_factors[i]+1):
            for k in range(count_factors[i]+1-j):
                a = array.copy()
                a[0] *= factors[i]**j
                a[1] *= factors[i]**k
                a[2] *= factors[i]**(count_factors[i]-j-k)
                arrays[i].append(a)
    ratio = 2**32
    best = None
    limit = (count_factors[len(indices)-1]**2+3*count_factors[len(indices)-1])//2+1
    while(indices[len(indices)-1] < limit):
        a = array.copy()
        for i in range(len(indices)):
            a *= arrays[i][indices[i]]
        lrs = a.copy()+0.0
        for i in range(len(lrs)):
            lrs[i] /= dims[i]
        lrs = np.sort(lrs)
        if((lrs[2]/lrs[0]) < ratio):
            ratio = lrs[2]/lrs[0]
            best = a
        indices[0] += 1
        for i in range(len(indices)-1):
            limit2 = (count_factors[i]**2+3*count_factors[i])//2+1
            if(indices[i] == limit2):
                indices[i] = 0
                indices[i+1] += 1
    if(ratio > 4):
        warnings.warn("Array is ["+str(dims[0]//best[0])+", "+str(dims[1]//best[1])+", "+str(dims[2]//best[2])+"], with a ratio of "+str(ratio)+"\n"+
                      "This size ratio is larger than 4, and could lead to poor performance!\n"+
                      "For number of cores, consider using a highly composite number")
    x = best[0]
    y = best[1]
    z = best[2]
    for i in range(x):
        x_dims.append(dims[0]//x)
    for i in range(dims[0]-(x*(dims[0]//x))):
        x_dims[i] += 1
    for i in range(y):
        y_dims.append(dims[1]//y)
    for i in range(dims[1]-(y*(dims[1]//y))):
        y_dims[i] += 1
    for i in range(z):
        z_dims.append(dims[2]//z)
    for i in range(dims[2]-(z*(dims[2]//z))):
        z_dims[i] += 1
    
    return x_dims, y_dims, z_dims