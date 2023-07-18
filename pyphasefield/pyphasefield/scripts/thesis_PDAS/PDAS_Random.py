import pyphasefield as ppf
import pyphasefield.Engines as engines

tdbc = ppf.TDBContainer("../mc_ni_2.034_simplified.tdb", ["FCC_A1", "LIQUID"], ["AL", "CR", "FE", "MO", "NB", "NI", "TI"])

def run_random_simulation(G, K, length, nm):
    
    #G: Thermal gradient (K/cm)
    #K: Thermal cooling rate (K/s)
    #length: Total length of simulation, in cells
    
    size_x = 1000
    size_y = 3000

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
    sim.set_save_path("data/random_"+str(G)+"_"+str(K))
    sim.set_autosave_flag(True)
    sim.set_autosave_rate(100)
    sim.set_boundary_conditions(["NEUMANN", "PERIODIC"])
    
    #assume rectangular grid density, num seeds = 64*1000*3000/nm**2 (factor of 64 is because dx = 8nm)

    numseeds = int(64*size_x*size_y/(nm**2))

    data = {
        "d_ratio":2.5,
        "noise_c":1.,
        "melt_angle":0.,
        "sim_type":"seeds",
        "number_of_seeds":numseeds,
        "initial_concentration_array":[0.011, 0.212, 0.192, 0.018, 0.031, 0.524]
    }
    sim.set_user_data(data)

    #initialize simulation arrays, all parameter changes should be BEFORE this point!
    sim.initialize_engine()

    #change array data here, for custom simulations
    
    sim.dt = 0.000000001
    time_steps_required = int(-sim.dimensions[1]*sim.dx*G/K/sim.dt)
    sim.set_autosave_rate(time_steps_required//2)
    
    sim.simulate(time_steps_required//2)
    
    default_values = [0., 1., 0., 0.011, 0.212, 0.192, 0.018, 0.031, 0.524]
    
    for i in range(int(2*(length//size_x)-1)):
        sim.simulate(time_steps_required//2)
        sim.save_simulation()
        for j in range(len(sim.fields)):
            sim.fields[j].data[1:(size_y+1), 1:(size_x//2+1)] = sim.fields[j].data[1:(size_y+1), (size_x//2+1):(size_x+1)]
            sim.fields[j].data[1:(size_y+1), (size_x//2+1):(size_x+1)] = default_values[j]
        sim.temperature.data += G*sim.dx*(size_x//2)
        sim.just_before_simulating()
    print("Complete: G="+str(G)+", K="+str(K)+", nm="+str(nm))

run_random_simulation(88800., -1980000., 3000, 250)
run_random_simulation(103000., -1870000., 3000, 250)
run_random_simulation(105000., -1810000., 3000, 250)
run_random_simulation(130000., -1830000., 3000, 250)
run_random_simulation(144000., -1670000., 3000, 250)
run_random_simulation(200000., -1560000., 3000, 250)
