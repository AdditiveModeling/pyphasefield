import pyphasefield as ppf
import os

def test_ncomponent():
    "Simple Test of N-Component model, minimum size to avoid seed overstepping bounds"
    sim = ppf.Simulation("test")
    print("Current directory is: "+os.getcwd())
    sim.init_sim_NComponent([10, 10], tdb_path="./tests/Ni-Cu_Ideal.tdb")
    sim.simulate(2)
    
def test_ncomponent_very_small():
    "Tests what happens when the seed exceeds the size of the sim region"
    "Doesnt normally happen, but may for the multiple seeds case when a seed is near the edge, or for a 1d simulation"
    sim = ppf.Simulation("test")
    print("Current directory is: "+os.getcwd())
    sim.init_sim_NComponent([5, 5], tdb_path="./tests/Ni-Cu_Ideal.tdb")
    sim.simulate(2)
    
def test_ncomponent_1d():
    "Tests a 2d sim, but with one dimension of length 1, equivalent to a 1d simulation"
    sim = ppf.Simulation("test")
    print("Current directory is: "+os.getcwd())
    sim.init_sim_NComponent([20, 1], tdb_path="./tests/Ni-Cu_Ideal.tdb")
    sim.simulate(2)
    
def test_ncomponent_1d_actual():
    "Tests a 1d sim"
    sim = ppf.Simulation("test")
    print("Current directory is: "+os.getcwd())
    sim.init_sim_NComponent([20], tdb_path="./tests/Ni-Cu_Ideal.tdb")
    sim.simulate(2)