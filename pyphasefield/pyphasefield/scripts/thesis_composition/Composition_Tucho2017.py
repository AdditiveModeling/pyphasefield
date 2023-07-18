import pyphasefield as ppf
import pyphasefield.Engines as engines
import numpy as np

tdbc = ppf.TDBContainer("../mc_ni_2.034_simplified.tdb", ["FCC_A1", "LIQUID"], ["AL", "CR", "FE", "MO", "NB", "NI", "TI"])

def run_composition_simulation(G, K):
    
    #G: Thermal gradient (K/cm)
    #K: Thermal cooling rate (K/s)
    
    size_x = 1000
    size_y = 1000

    sim = engines.NCGPU(dimensions=[size_y, size_x])

    #initialize non-array parameters
    sim.set_framework("GPU_SERIAL") #"CPU_SERIAL", "GPU_SERIAL"
    sim.set_dx(0.0000008)
    sim.set_time_step_counter(0)
    sim.set_temperature_type("LINEAR_GRADIENT") 
    sim.set_temperature_initial_T(1584.)
    sim.set_temperature_dTdx(G)
    sim.set_temperature_dTdy(0.)
    sim.set_temperature_dTdz(0.)
    sim.set_temperature_dTdt(K)
    sim.set_temperature_units("K")
    sim.set_tdb_container(tdbc)
    sim.set_save_path("data/composition_"+str(G)+"_"+str(K))
    sim.set_autosave_flag(True)
    sim.set_autosave_rate(100)
    sim.set_boundary_conditions(["NEUMANN", "PERIODIC"])

    data = {
        "d_ratio":2.5,
        "noise_c":1.,
        "melt_angle":0.,
        "sim_type":"seeds",
        "number_of_seeds":0,
        "initial_concentration_array":[0.011, 0.212, 0.192, 0.018, 0.031, 0.524]
    }
    sim.set_user_data(data)

    #initialize simulation arrays, all parameter changes should be BEFORE this point!
    sim.initialize_engine()
    sim.user_data["H"] = 0.00000000001
    sim.user_data["M_qmax"] = 12000000000.

    #change array data here, for custom simulations
    num_seeds = int(sim.dimensions[0]*(sim.dx*10000000)/250)
    d_size = 250/(sim.dx*10000000)
    for i in range(num_seeds):
        sim.fields[0].data, sim.fields[1].data, sim.fields[2].data = ppf.make_seed(sim.fields[0].data, sim.fields[1].data, sim.fields[2].data, int(0.5*d_size), int((i+0.5)*d_size), 0., int(0.25*d_size))

    sim.dt = 0.000000001
    time_steps_required = int(-sim.dimensions[1]*sim.dx*G/K/sim.dt)
    sim.set_autosave_rate(time_steps_required//2)
    
    sim.simulate(4*time_steps_required)
    print("Complete: G="+str(G)+", K="+str(K))

run_composition_simulation(130000., -1830000.)