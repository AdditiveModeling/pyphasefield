import pyphasefield as ppf
from pyphasefield.Engines.Diffusion import Diffusion
from pathlib import Path
import numpy as np

def teardown_module(module):
    path = Path.cwd()
    try:
        path.joinpath("save_folder/step_5.hdf5").unlink(missing_ok=True)
        path.joinpath("save_folder/step_10.hdf5").unlink(missing_ok=True)
        path.joinpath("save_folder").rmdir()
    except FileNotFoundError:
        pass

def test_1_saving():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, save_path="save_folder", user_data={"D": 0.1}, boundary_conditions=["PERIODIC"])
    sim._autosave_flag = True
    sim._autosave_rate = 5
    sim.initialize_engine()
    # Create a more interesting initial condition that will diffuse
    sim.fields[0].data[sim.fields[0]._slice] = 0.0
    sim.fields[0].data[sim.fields[0]._slice][0:3] = 1.0  # High concentration on left side
    sim.simulate(10)
    assert sim.time_step_counter == 10
    assert len(sim.fields) > 0
    
def test_2_loading_step_number():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, save_path="save_folder", user_data={"D": 0.1}, boundary_conditions=["PERIODIC"])
    sim.load_simulation(step=5)
    print(sim.fields[0].data)
    data1 = sim.fields[0].data.copy()
    sim.load_simulation(step=10)
    print(sim.fields[0].data)
    #check to see that the saved data files are different for different time steps
    assert not np.array_equal(sim.fields[0].data, data1), "Data should be different between step 5 and step 10"
    
def test_3_loading_relative_path():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, save_path="save_folder", user_data={"D": 0.1}, boundary_conditions=["PERIODIC"])
    sim.load_simulation(file_path="step_5.hdf5")
    print(sim.fields[0].data)
    data1 = sim.fields[0].data.copy()
    sim.load_simulation(file_path="step_10.hdf5")
    print(sim.fields[0].data)
    #check to see that the saved data files are different for different time steps
    assert not np.array_equal(sim.fields[0].data, data1), "Data should be different between step 5 and step 10"
    
def test_4_loading_absolute_path():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, save_path="save_folder", user_data={"D": 0.1}, boundary_conditions=["PERIODIC"])
    sim.load_simulation(file_path="save_folder/step_5.hdf5")
    print(sim.fields[0].data)
    data1 = sim.fields[0].data.copy()
    sim.load_simulation(file_path="save_folder/step_10.hdf5")
    print(sim.fields[0].data)
    #check to see that the saved data files are different for different time steps
    assert not np.array_equal(sim.fields[0].data, data1), "Data should be different between step 5 and step 10"