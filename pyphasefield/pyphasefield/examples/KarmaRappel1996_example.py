from numba import cuda

import numpy as np
import matplotlib.pyplot as plt
import time
import pyphasefield.Engines as engines
from scipy.optimize import curve_fit


'''
Example calculation of dendrite tip velocities in the anisotropic Karma-Rappel model
This script reproduces Figure 5.7 in the book 'Phase-Field Methods in Materials Science and Engineering' by Nikolas Provatas and Ken Elder.

Takes a few minutes to run with a GPU
'''



start = time.time()

# Define the hyperbolic tangent function
def tanh_function(x, a, b):
    return a * np.tanh((x - b)/np.sqrt(2))

# Set same parameters as in the book
length = 400
Nx = 1000
Ny = 1000

dx = length/Nx
dt = 0.014
t_fin = 800


lambdas = [3.19,1.8]

dpi = 1000
plt.figure(figsize=(4600/dpi,3200/dpi),dpi=dpi)


for lambda_ in lambdas:

    sim = engines.KarmaRappel1996(dimensions=[Nx, Ny])

    #initialize non-array parameters
    sim.set_framework("GPU_SERIAL")
    sim.set_dx(dx)
    sim.set_dt(dt)
    sim.set_boundary_conditions(['NEUMANN','NEUMANN','NEUMANN','NEUMANN'])

    a1 = 0.6267
    a2 = 0.8839

    data = {
        'w':1.,
        'lambda_val':lambda_,
        'tau':1.,
        'D':a2*lambda_,
        'e4':0.05
    }
    sim.set_user_data(data)

    r0 = 10

    n_steps = t_fin / dt

    length = sim.dimensions[0]
    width = sim.dimensions[1]


    sim.initialize_fields_and_imported_data()
    for xi in range(Nx+2):
        for yi in range(Ny+2):
            # Set hyperbolic initial condition:

            sim.fields[0].data[xi,yi] = np.tanh((r0**2 - ((xi-1)*dx)**2 - ((yi-1)*dx)**2 )/np.sqrt(2) )
            
            sim.fields[1].data[xi,yi] = -0.55

    # Number of data points to calculate
    n = 140

    steps_per_iter = int(n_steps/n)

    x_grid = np.arange(Nx+2)

    t = [0]
    boundary = [r0]


    for i in range(1,n+1):
        sim.simulate(steps_per_iter)
        
        sim._fields_gpu_device[0].copy_to_host(sim.fields[0].data)
        phidata = sim.fields[0].data[0,:]

        indices = np.where((phidata < 0.999) & (phidata > -0.999))
        phi_filtered = phidata[indices]
        x_filtered = x_grid[indices]
        
        popt, _ = curve_fit(tanh_function, x_filtered, phi_filtered, p0=(-0.9,np.take(x_filtered, x_filtered.size // 2)))
        a, b = popt
                
        tval = i*steps_per_iter*dt

        boundary.append(b*sim.dx)
        t.append(tval)
        
    

    V_tip = np.gradient(boundary)
    V_tip = V_tip / (t[1]-t[0])

    V_tip_dimensionless = V_tip*a1 / (a2*sim.user_data['lambda_val']**2)

    plt.plot(t, V_tip, label=r'$ V \:\: (\bar{\lambda} = %s )$' % lambda_)
    plt.plot(t, V_tip_dimensionless, linestyle='dashed', label=r'$ \bar{V} \:\: (\bar{\lambda} = %s )$' % lambda_)


end = time.time()

print('Elapsed time:',str(end-start))
    
        
positions_x = [0,100,200,300,400,500,600,700,800]
labels_x = [0,'',200,'',400,'',600,'',800]

positions_y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
labels_y = [0, '', 0.2, '', 0.4, '', 0.6, '', 0.8]

plt.xticks(positions_x, labels_x)

plt.yticks(positions_y, labels_y)

plt.tick_params(axis='y', which='both', left=True, right=True)    
plt.grid()    
plt.xlabel(r'$t$')
plt.ylabel('Tip velocities')
plt.ylim([-0.02, 0.8])
plt.xlim([-15,t_fin])
plt.legend()
plt.savefig('dendrite_tip_velocities_example.png',bbox_inches='tight')
