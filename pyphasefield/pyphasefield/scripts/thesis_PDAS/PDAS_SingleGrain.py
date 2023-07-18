import sys
sys.path.insert(0,"../../../pyphasefield/pyphasefield")
import pyphasefield as ppf
import pyphasefield.Engines as engines
import matplotlib.pyplot as plt
import numpy as np

tdbc = ppf.TDBContainer("../mc_ni_2.034_simplified.tdb", ["FCC_A1", "LIQUID"], ["AL", "CR", "FE", "MO", "NB", "NI", "TI"])

def run_sg_simulation(G, K, nm):
    
    #G: Thermal gradient (K/cm)
    #K: Thermal cooling rate (K/s)
    
    size_x = 1000
    size_y = int(nm//8)

    sim = engines.NCGPU(dimensions=[size_y, size_x])

    #initialize non-array parameters
    sim.set_framework("GPU_SERIAL") #"CPU_SERIAL", "GPU_SERIAL"
    sim.set_dx(0.0000008)
    sim.set_time_step_counter(0)
    sim.set_temperature_type("LINEAR_GRADIENT") 
    sim.set_temperature_initial_T(1576.)
    sim.set_temperature_dTdx(G)
    sim.set_temperature_dTdy(0.)
    sim.set_temperature_dTdz(0.)
    sim.set_temperature_dTdt(K)
    sim.set_temperature_units("K")
    sim.set_tdb_container(tdbc)
    sim.set_save_path("data/moving_singlegrain2_"+str(G)+"_"+str(K)+"_"+str(nm))
    sim.set_autosave_flag(True)
    sim.set_autosave_rate(100)
    sim.set_boundary_conditions(["NEUMANN", "PERIODIC"])

    data = {
        "d_ratio":2.5,
        "noise_phi":0.,
        "noise_c":0.,
        "noise_q":0.,
        "melt_angle":0.,
        "sim_type":"seeds",
        "number_of_seeds":0,
        "initial_concentration_array":[0.011, 0.212, 0.192, 0.018, 0.031, 0.524]
    }
    sim.set_user_data(data)

    #initialize simulation arrays, all parameter changes should be BEFORE this point!
    sim.initialize_engine()

    #change array data here, for custom simulations
    num_seeds = 1
    d_size = 200/(sim.dx*10000000)
    for i in range(num_seeds):
        sim.fields[0].data, sim.fields[1].data, sim.fields[2].data = ppf.make_seed(sim.fields[0].data, sim.fields[1].data, sim.fields[2].data, int(0.5*d_size), int(nm//16), 0., int(0.25*d_size))

    sim.dt = 0.000000001
    time_steps_required = int(-sim.dimensions[1]*sim.dx*G/K/sim.dt)
    sim.set_autosave_rate(time_steps_required)
    
    sim.simulate(time_steps_required)
    sim.save_simulation()
    print("Complete: G="+str(G)+", K="+str(K)+", nm="+str(nm))

for i in range(46):
    run_sg_simulation(88800., -1980000., 200+40*i)
    run_sg_simulation(103000., -1870000., 200+40*i)
    run_sg_simulation(105000., -1810000., 200+40*i)
    run_sg_simulation(130000., -1830000., 200+40*i)
    run_sg_simulation(144000., -1670000., 200+40*i)
    run_sg_simulation(200000., -1560000., 200+40*i)