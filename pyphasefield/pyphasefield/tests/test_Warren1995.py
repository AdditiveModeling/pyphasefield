import pyphasefield as ppf
from pyphasefield.Engines.Warren1995 import Warren1995

def test_warren1995():
    sim = Warren1995(dimensions=[20,20], dx=1.0, dt=0.01, 
                     boundary_conditions=["PERIODIC", "PERIODIC"],
                     temperature_type="ISOTHERMAL", initial_T=1600.0)
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0