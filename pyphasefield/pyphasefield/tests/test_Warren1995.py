import pyphasefield as ppf

def test_warren1995():
    sim = ppf.Simulation("test")
    sim.init_sim_Warren1995([20,20])
    sim.simulate(2)