import pyphasefield as ppf
import pyphasefield.Engines as engines

tdbc = ppf.TDBContainer("Ni-Cu_Ideal.tdb", ["FCC_A1", "LIQUID"], ["CU", "NI"])

sim = engines.NCGPU(dimensions=[400, 400])
sim.set_dx(0.0000046)
sim.set_tdb_container(tdbc)
sim.set_temperature_type("ISOTHERMAL") 
sim.set_temperature_initial_T(1574.)
sim.set_save_path("data/NCGPU_test")
sim.set_boundary_conditions(["PERIODIC", "PERIODIC"])
user_data = {
    "d_ratio":1.1, #interface width to cell spacing ratio
    "sim_type":"seed",
    "initial_concentration_array":[0.3937]
    
}
sim.set_user_data(user_data)

sim.initialize_engine()

sim.simulate(15000)

#this will automatically create the file folder for the simulation, plot_simulation will not!
sim.save_simulation()

#save images as files. If this code is copy-pasted into a Jupyter notebook, reverse the booleans to plot images in the NB!
#also don't forget to fix the path to the TDB!
sim.plot_simulation(save_images=True, show_images=False)