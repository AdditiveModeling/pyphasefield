from Karma2001 import Karma2001

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit


'''
Example calculation of dendrite tip velocity and concentration profile in the anisotropic Karma2001 binary alloy model
This script partly reproduces Figures 1 and 2 in 10.1103/PhysRevLett.87.115701 with d0=0.277

Takes a few minutes to run with a GPU
'''

start = time.time()

# Define the hyperbolic tangent function used to fit the interface profile
def tanh_function(x, a, b):
    return a * np.tanh((x - b)/np.sqrt(2))


length = 400
Nx = 1000
Ny = 1000

dx = length/Nx
dt = 0.008

sim = Karma2001(dimensions=[Nx, Ny])

# Initialize non-array parameters
sim.set_framework("GPU_SERIAL") #"CPU_SERIAL"
sim.set_dx(dx)
sim.set_dt(dt)
sim.set_save_path("Karma2001_example")
sim.set_autosave_flag(False)
sim.set_autosave_save_images_flag(False)
sim.set_autosave_rate(20000)
sim.set_boundary_conditions(['NEUMANN','NEUMANN','NEUMANN','NEUMANN'])


# Dimensionless units (W = tau = 1)
a1 = 0.8839
a2 = 0.6267
d0 = 0.277                  # Capillary length
lambda_val = 3.19           # Coupling parameter
D = 2                       # Diffusion coefficient
e4 = 0.02                   # Anisotropy strength
k = 0.15                    # Partition coefficient
a_t = 1/(2*np.sqrt(2))      # Antitrapping current

r0 = 22*d0                  # Initial seed radius

data = {
    'w':1.,
    'lambda_val':lambda_val,
    'tau':1.,
    'D':D,
    'e4':e4,
    'k':k,
    'a_t':a_t
}
sim.set_user_data(data)


# Simulation time and number of steps
t_fin = 400
n_steps = t_fin / dt


# Initialize simulation arrays, all parameter changes should be BEFORE this point!
sim.initialize_fields_and_imported_data()

# Initialize circular seed
supersaturation = 0.55
init_u = (1 - (1 - k) * supersaturation)

center_x = Nx // 2 + 1
center_y = Ny // 2 + 1

for xi in range(Nx+2):
    for yi in range(Ny+2):
        phi_val = np.tanh((r0-np.sqrt(((xi - center_x)*dx)**2 + ((yi - center_y)*dx)**2))/np.sqrt(2))
        sim.fields[0].data[xi, yi] = phi_val

        sim.fields[1].data[xi, yi] = 0.5 * init_u * (1 + k - (1 - k) * phi_val)
        sim.fields[2].data[xi, yi] = init_u
        
n = 140

steps_per_iter = int(n_steps/n)
x_grid = np.arange(Nx+2)

t = [0]
boundary = [r0]


# Main simulation loop
for i in range(1,n+1):
    # Uncomment to save images
    #sim.save_images()
    sim.simulate(steps_per_iter)

    sim._fields_gpu_device[0].copy_to_host(sim.fields[0].data)
    
    phidata = sim.fields[0].data
    

    phidata = phidata[Nx // 2 + 1][Ny // 2:] 

    # Fit boundary data to tanh function and extract boundary position
    indices = np.where((phidata < 0.999) & (phidata > -0.999))
    phi_filtered = phidata[indices]
    x_filtered = x_grid[indices]

    
    popt, _ = curve_fit(tanh_function, x_filtered, phi_filtered, p0=(-0.95,np.take(x_filtered, x_filtered.size // 2)))
    a, b = popt
            
    tval = i*steps_per_iter*dt

    boundary.append(b)
    t.append(tval)
    

end = time.time()

print('Elapsed time:',str(end-start))


# Plot the dendrite tip velocity
V_tip = np.gradient(boundary)
V_tip = sim.dx * V_tip/(t[1]-t[0]) * d0/D

t = np.array(t) * D/d0**2

dpi = 1000
plt.figure(figsize=(4600/dpi,3200/dpi),dpi=dpi)
plt.plot(t, V_tip)

plt.tick_params(axis='y', which='both', left=True, right=True)    
plt.grid()    
plt.xlabel(r'$tD/d_0^2$')
plt.ylabel(r'$Vd_0/D$')
plt.ylim([0, 0.08])
plt.xlim([0,10000])

plt.savefig('Karma2001_example_dendrite_tip_velocity.png',bbox_inches='tight')


# Plot the final concentration profile
sim._fields_gpu_device[1].copy_to_host(sim.fields[1].data)
c_data = sim.fields[1].data
c_profile = c_data[int(Nx/2)][int(Nx/2):]
x = np.linspace(0,length,len(c_profile))
plt.figure(figsize=(4600/dpi,3200/dpi),dpi=dpi)
plt.plot(x, c_profile)

plt.tick_params(axis='y', which='both', left=True, right=True)    
plt.grid()    
plt.xlabel(r'$x$')
plt.ylabel(r'$c/c_0^l$')

plt.savefig('Karma2001_example_concentration profile.png',bbox_inches='tight')
