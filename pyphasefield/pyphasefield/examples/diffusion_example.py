import sys
sys.path.insert(0,"../..")
import pyphasefield as ppf

sim = ppf.Simulation("data/diffusion_test")
sim.init_sim_Diffusion([20])
print(sim.fields[0])
sim.simulate(100)
print(sim.fields[0])
