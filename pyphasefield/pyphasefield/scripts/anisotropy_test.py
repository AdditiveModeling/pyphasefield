import sys
sys.path.insert(0, "../..") #location of pyphasefield files
import pyphasefield as ppf
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

sim = None
rotations = []
distances = []
angles = []
anisotropies = []
interfacewidths = []
def run_Q_test(angle, widthcells, isometric):
    global sim, rotations, distances, angles, anisotropies, interfacewidths
    dx = 2.3e-06
    #d = 4.8936e-06*2
    d = dx*widthcells
    sim = ppf.Simulation("data/q_test_"+str(angle))
    dim = [200, 200]
    sim.init_sim_NCGPU(dim=dim, solver="explicit", temperature_type="gradient", initial_temperature=1560, 
                            temperature_gradient=0, cooling_rate = 0,
                            initial_concentration_array=[0.126], tdb_path="Ni-Nb_Simplified.tdb", cell_spacing=dx, 
                            d_ratio=(d/dx), nbc=["periodic", "periodic"], sim_type="seed", number_of_seeds=0)
    #ratio = sim.H/H
    #sim.H = H
    #sim.M_qmax *= ratio
    #sim.params[3] = sim.M_qmax
    #sim.params[4] = sim.H
    if isometric:
        sim.y_e = 0
        sim.params[5] = 0
    sim.fields[1].data = np.zeros(dim)
    sim.fields[1].data += np.cos(0.5*angle)
    sim.fields[2].data = np.zeros(dim)
    sim.fields[2].data += np.sin(0.5*angle)
    D = sim.M[0]*sim.ebar*sim.ebar*1685.
    if D < sim.M[1]*sim.ebar*sim.ebar*1685.:
        D = sim.M[1]*sim.ebar*sim.ebar*1685.
    if D < sim.D_L:
        D = sim.D_L
    sim.set_time_step_length(sim.get_cell_spacing()**2/20./D)
    sim.set_checkpoint_rate(100000)
    sim.send_fields_to_GPU()
    #sim.plot_simulation(save_images=False)
    sim.simulate(1500)
    sim.retrieve_fields_from_GPU()
    #sim.plot_simulation(save_images=False)
    try:
        w = np.where(sim.fields[0].data > 0.5)
        dist = np.sqrt(np.max((w[0]-100)**2+(w[1]-100)**2))

        a = np.argmax((w[0]-100)**2+(w[1]-100)**2)
        #print(w[0][a], w[1][a])
        an = np.arctan2(w[1][a]-100, w[0][a]-100)
        while an > np.pi/2:
            an = an - np.pi/2
        if an > np.pi/4:
            an = np.pi/2-an
        #print(an)
        l1 = np.argmax(sim.fields[0].data[100] > 0.5)
        ani1 = np.sqrt((l1-100.)**2)
        l2 = np.argmax(np.diagonal(sim.fields[0].data) > 0.5)
        ani2 = np.sqrt(2*(l2-100.)**2)
        if isometric:
            anisotropies.append(ani1/ani2)
            interfacewidths.append(widthcells)
        else:
            angles.append(angle)
            rotations.append(an)
            distances.append(dist)
    except:
        print("Failed on following params:")
        print(angle, widthcells, isometric)
        
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
        
print("Anisotropy Test for Grid Effects")
print("Runs two sets of tests, explicitly anisotropic, and simulations that are supposed to show isotropic growth")
print("")
print("First set (anisotropic):")
print("num_widths_aniso: number of interface widths to simulate, inclusive of end points. 0 to skip")
print("max_width_aniso: maximum interface width in anisotropic runs, must be >2")
print("num_runs_aniso: number of anisotropic runs *per interface width*, inclusive of end points")
print("max_angle_aniso: maximum angle used in anisotropic runs, in degrees")
print("")
print("E.g.: 3, 4, 4, 45 will run simulations at angles of 0, 15, 30, 45 degrees, for interface widths of 2, 3, 4")
print("Recommended values to start: 100, 90, 3, 4")
print("")
print("Second set (isotropic):")
print("num_widths_iso: number of isotropic runs, one per interface width, inclusive of end points. 0 to skip")
print("min_width_iso: minimum interface width in isotropic runs (recommended >= 1 for stability)")
print("max_width_iso: maximum interface width in isotropic runs")
print("")
print("E.g.: 9, 1, 5 will run simulations at interface widths of 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5")
print("Recommended values to start: 100, 1, 5")
print("")

print("Number of interface widths to consider? (Inclusive of end points, 0 to skip)")
num_widths_aniso = int(input("num_widths_aniso = "))
if(num_widths_aniso > 0):
    print("Maximum interface width in cells? (must be greater than 2)")
    max_width_aniso = float(input("max_width_aniso = "))
    print("Number of Anisotropic Simulations to run per interface width?")
    num_runs_aniso = int(input("num_runs_aniso = "))
    print("Maximum Angle for Anisotropic Simulations? (In degrees)")
    max_angle_aniso = float(input("max_angle_aniso = "))
    max_angle_aniso = np.pi*max_angle_aniso/180.
    
print("Number of Isotropic Simulations to run? (0 to skip)")
num_widths_iso = int(input("num_widths_iso = "))
if(num_widths_iso > 0):
    print("Minimum interface width in cells? (should be greater or equal to 1)")
    min_width_iso = float(input("min_width_iso = "))
    print("Maximum interface width in cells?")
    max_width_iso = float(input("max_width_iso = "))
print("")

if(num_widths_aniso == 0 and num_widths_iso == 0):
    print("Done!")
else:
    t0 = time.time()
    run_Q_test(0, 2, False)
    t1 = time.time()
    t = t1-t0
    print("Time for one simulation: "+str(t)+" seconds")
    runs = num_runs_aniso*num_widths_aniso + num_widths_iso
    t_total = t*runs
    print("Expected completion time for "+str(runs)+" total runs: "+str(t_total)+" seconds") 
    print("    ("+str(t_total/60.)+" minutes, "+str(t_total/3600.)+" hours, "+str(t_total/3600./24.)+" days")

if(num_widths_aniso > 0):
    cmap = get_cmap(int(num_widths_aniso*1.5))
    all_r = []
    all_d = []
    all_a = []
    for i in range(num_widths_aniso):
        rotations = []
        distances = []
        angles = []
        for j in range(num_runs_aniso):
            run_Q_test(j*max_angle_aniso/(num_runs_aniso-1), 2.+i*(max_width_aniso-2.)/(num_widths_aniso-1), False)
        all_r.append(rotations)
        all_d.append(distances)
        all_a.append(angles)
    
    plt.figure(figsize=(12,6))
    legend = []
    for i in range(num_widths_aniso):
        plt.plot(all_a[i], all_r[i], "o", c=cmap(i))
        legend.append("dx/d = "+str(2.+i*(max_width_aniso-2.)/(num_widths_aniso-1)))
    plt.plot(all_a[0], np.minimum(all_a[0], np.pi/2-np.array(all_a[0])), "k")
    legend.append("Ideal")
    plt.xlabel("Input angle")
    plt.ylabel("Growth angle")
    plt.legend(legend)
    plt.savefig("Aniso_Angles.png")
    
    plt.figure(figsize=(12,6))
    for i in range(num_widths_aniso):
        plt.plot(all_a[i], all_d[i]/np.max(all_d[i]), "o", c=cmap(i))
    plt.plot(all_a[0], np.ones(len(all_a[0])), "k")
    plt.xlabel("Input angle")
    plt.ylabel("Relative Tip Velocity")
    plt.legend(legend)
    plt.savefig("Aniso_Velocities.png")
    
if(num_widths_iso > 0):
    for i in range(num_widths_iso):
        run_Q_test(0, min_width_iso+i*(max_width_iso-min_width_iso)/(num_widths_iso-1), True)
    plt.figure(figsize=(12,6))
    plt.plot(interfacewidths, anisotropies, "ko")
    plt.plot(interfacewidths, np.ones(len(interfacewidths)), "k")
    plt.xlabel("Interface width (#cells)")
    plt.ylabel("Measured Isotropic Deviation")
    plt.legend(["Simulations", "Ideal"])
    plt.savefig("Iso_Deviations.png")