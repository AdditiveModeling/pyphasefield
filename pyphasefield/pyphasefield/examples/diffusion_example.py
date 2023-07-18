import sys
sys.path.insert(0,"../..")
import pyphasefield.Engines as engines
import time

print("Running CPU diffusion engine!")

t0 = time.time()

############################################
# Beginning of typical pyphasefield script #
############################################

timesteps = 1000
length = 1000

sim = engines.Diffusion(dimensions=[length, length])

#initialize non-array parameters/flags
sim.set_framework("CPU_SERIAL") #"CPU_SERIAL", "GPU_SERIAL" (GPU_SERIAL requires numba)
sim.set_dx(1.)
sim.set_dt(0.1)
sim.set_save_path("data/diffusion_test_cpu")
sim.set_boundary_conditions("PERIODIC")

data = {
    "D":1.
}
sim.set_user_data(data)

#initialize simulation arrays, all parameter changes should be BEFORE this point!
sim.initialize_engine()

#save initial condition images
sim.save_images()

#run simulation
sim.simulate(timesteps)

#save final condition images
sim.save_images()

######################################
# End of typical pyphasefield script #
######################################

t1 = time.time()

print("Done! Took {:.3f} seconds to run ".format(t1-t0)+str(timesteps)+" timesteps for a "+str(length)+"x"+str(length)+" simulation")

try:
    from numba import cuda
    
    print("Running GPU diffusion engine!")
    
    t0 = time.time()

    timesteps = 1000
    length = 1000
    
    sim = engines.Diffusion(dimensions=[length, length])

    #initialize non-array parameters/flags
    sim.set_framework("GPU_SERIAL") #"CPU_SERIAL", "GPU_SERIAL" (GPU_SERIAL requires numba)
    sim.set_dx(1.)
    sim.set_dt(0.1)
    sim.set_save_path("data/diffusion_test_gpu")
    sim.set_boundary_conditions("PERIODIC")

    data = {
        "D":1.
    }
    sim.set_user_data(data)

    #initialize simulation arrays, all parameter changes should be BEFORE this point!
    sim.initialize_engine()

    #save initial condition images
    sim.save_images()

    #run simulation
    sim.simulate(timesteps)

    #save final condition images
    sim.save_images()
    
    t1 = time.time()
    
    print("Done! Took {:.3f} seconds to run ".format(t1-t0)+str(timesteps)+" timesteps for a "+str(length)+"x"+str(length)+" simulation")
except:
    print("Cannot run GPU diffusion engine, Numba/CUDA is not installed!")