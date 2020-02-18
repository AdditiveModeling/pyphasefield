import pyphasefield as ppf
from pathlib import Path

def teardown_module(module):
    path = Path.cwd()
    path.joinpath("save_folder/step_5.npz").unlink()
    path.joinpath("save_folder/step_10.npz").unlink()
    path.joinpath("save_folder").rmdir()

def test_1_saving():
    sim = ppf.Simulation(save_path="save_folder")
    sim.init_sim_Diffusion([10])
    sim._time_steps_per_checkpoint = 5
    sim.simulate(10)
    
def test_2_loading_step_number():
    sim = ppf.Simulation(save_path="save_folder")
    sim.load_simulation(step=5)
    print(sim.fields[0].data)
    data1 = sim.fields[0].data
    sim.load_simulation(step=10)
    print(sim.fields[0].data)
    #check to see that the saved data files are different for different time steps
    assert(not (sim.fields[0].data[0] == data1[0]))
    
def test_3_loading_relative_path():
    sim = ppf.Simulation(save_path="save_folder")
    sim.load_simulation(file_path="step_5.npz")
    print(sim.fields[0].data)
    data1 = sim.fields[0].data
    sim.load_simulation(file_path="step_10.npz")
    print(sim.fields[0].data)
    #check to see that the saved data files are different for different time steps
    assert(not (sim.fields[0].data[0] == data1[0]))
    
def test_4_loading_absolute_path():
    sim = ppf.Simulation()
    sim.load_simulation(file_path="save_folder/step_5.npz")
    print(sim.fields[0].data)
    data1 = sim.fields[0].data
    sim.load_simulation(file_path="save_folder/step_10.npz")
    print(sim.fields[0].data)
    #check to see that the saved data files are different for different time steps
    assert(not (sim.fields[0].data[0] == data1[0]))