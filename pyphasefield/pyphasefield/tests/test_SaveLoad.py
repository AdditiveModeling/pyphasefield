import pyphasefield as ppf

def test_1_saving():
    sim = ppf.Simulation("save_folder")
    sim.init_sim_Diffusion([10])
    sim._time_steps_per_checkpoint = 5
    sim.simulate(10)
    
def test_2_loading():
    sim = ppf.Simulation("save_folder")
    sim.load_simulation(5)
    print(sim.fields[0].data)
    data1 = sim.fields[0].data
    sim.load_simulation(10)
    print(sim.fields[0].data)
    assert(not (sim.fields[0].data[0] == data1[0]))
    
#def test_3_cleanup()
    #import os
    #os.remove("./save_folder/data_0.npz")
    #os.rmdir("./save_folder")