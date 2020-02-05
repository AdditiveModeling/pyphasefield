import pyphasefield as ppf

def test_diffusion():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10])
    sim.simulate(2)