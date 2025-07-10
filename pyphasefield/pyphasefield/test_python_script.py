import sys
sys.path.insert(0,"..")
import pyphasefield.Engines as engines
import matplotlib.pyplot as plt
import numpy as np
import time

def run_sim(x, tstart, ts, size):
    sim = engines.NCGPU_new(dimensions=[size, size])

    #initialize non-array parameters
    sim.set_framework("GPU_SERIAL") #"CPU_SERIAL", "GPU_SERIAL"
    sim.set_dx(x)
    sim.set_time_step_counter(0)
    sim.set_temperature_type("ISOTHERMAL") #None, "ISOTHERMAL", "LINEAR_GRADIENT", "XDMF_FILE"
    sim.set_temperature_initial_T(1574.)
    sim.set_temperature_dTdx(None)
    sim.set_temperature_dTdy(None)
    sim.set_temperature_dTdz(None)
    sim.set_temperature_dTdt(None)
    sim.set_temperature_path(None)
    sim.set_temperature_units("K")
    sim.set_tdb_path("tests/Ni-Cu-Al_Ideal.tdb")
    #sim.set_tdb_path("tests/Ni-Nb_Simplified.tdb")
    #sim.set_tdb_path("tests/mc_ni_v2.034.pycalphad.tdb")
    sim.set_tdb_phases(["FCC_A1", "LIQUID"])
    sim.set_tdb_components(["AL", "CU", "NI"])
    sim.set_save_path(f"data/{x}_{size}")
    sim.set_autosave_flag(False)
    sim.set_autosave_save_images_flag(False)
    sim.set_autosave_rate(100000)
    sim.set_boundary_conditions(["PERIODIC", "PERIODIC"])

    data = {
        "d_ratio":1.1,
        "noise_phi":1.,
        "noise_c":1.,
        "noise_q":1.,
        "sim_type":"seeds",
        "number_of_seeds":3,
        "initial_concentration_array":[0.0001, 0.3937],
        "melt_angle":0.,
        "seed_angle":np.pi/4
    }
    sim.set_user_data(data)

    #initialize simulation arrays, all parameter changes should be BEFORE this point!
    sim.initialize_engine()
    
    #make seeds manually
    for i in range(10000):
        x = int(np.random.rand()*100)
        #make_seed(sim, q=[1, 2], x=x)
        

    #change array data here, for custom simulations
    if(tstart > 0):
        sim.load_simulation(step=tstart)

    #run simulation
    sim.plot_simulation(fields=[0, 1, 2, 3, 4], interpolation="nearest", save_images=False)
    t0 = time.time()
    sim.simulate(1)
    t1 = time.time()
    sim.simulate(ts-1)
    t2 = time.time()
    print("Time to run first iteration: "+str(t1-t0)+" seconds")
    print("Average time to run later iterations: "+str((t2-t1)/(ts-1))+" seconds")
    print("Total runtime: "+str((t2-t0))+" seconds for "+str(ts)+" timesteps")
    sim.plot_simulation(fields=[0, 1, 2, 3, 4], interpolation="nearest", save_images=False)
    sim.save_simulation()
    #sim.finish_simulation()
    #print(sim.fields[1].data)
    return sim #debugging

sim = run_sim(0.0000092, 1250, 1250, 200)