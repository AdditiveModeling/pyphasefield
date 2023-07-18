import pyphasefield as ppf
import pyphasefield.Engines as engines

tdbc = ppf.TDBContainer("mc_ni_2.034_simplified.tdb", ["FCC_A1", "LIQUID"], ["AL", "CR", "FE", "MO", "NB", "NI", "TI"])

sim = engines.NCGPU(dimensions=[200, 200])
sim.set_dx(0.0000008)
sim.set_tdb_container(tdbc)
sim.set_temperature_type("ISOTHERMAL") 
sim.set_temperature_initial_T(1584.)
sim.set_save_path("data/NCGPU_test_Inconel718")
sim.set_boundary_conditions(["PERIODIC", "PERIODIC"])
user_data = {
    "d_ratio":2.5, #interface width to cell spacing ratio
    "sim_type":"seed",
    "initial_concentration_array":[0.011, 0.212, 0.192, 0.018, 0.031, 0.524]
    
}
sim.set_user_data(user_data)

sim.initialize_engine()
sim.dt = 0.000000001 #manually set timestep to 1ns

sim.simulate(20000)

#this will automatically create the file folder for the simulation, plot_simulation will not!
sim.save_simulation()

#save images as files. If this code is copy-pasted into a Jupyter notebook, reverse the booleans to plot images in the NB!
#also don't forget to fix the path to the TDB!
sim.plot_simulation(save_images=True, show_images=False)