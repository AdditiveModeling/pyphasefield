import sys
sys.path.insert(0,"../..")
import pyphasefield as ppf

saveloc = input("What folder in data to save under?")
sim = ppf.Simulation(save_path="data/"+saveloc)
conc = float(input("Initial concentration? [0.40831 is recommended!]"))
sim.init_sim_NComponent(dim=[100, 100], sim_type="seed", initial_temperature=1574, initial_concentration_array=[conc])
initial_step = int(input("What step to load from? (-1 = new simulation)"))
if(initial_step == -1):
    sim.save_simulation()
else:
    sim.load_simulation(step=initial_step)
sim._time_steps_per_checkpoint = 200
sim._boundary_conditions_type = ["periodic", "periodic"]

totalsteps = int(input("How many steps to run?"))
progress_bar_steps=int(totalsteps/20)
for i in range(20):
    sim.simulate(progress_bar_steps)
    print(str((i+1)*progress_bar_steps)+" steps completed out of "+str(totalsteps))
sim.simulate(totalsteps-20*progress_bar_steps)
print("Completed!")