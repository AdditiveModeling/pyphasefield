import pyphasefield as ppf
from pyphasefield.Engines.Diffusion import Diffusion

def test_diffusion_default1dexplicit():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "explicit"}, boundary_conditions=["PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_default2dexplicit():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "explicit"}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_default3dexplicit():
    sim = Diffusion(dimensions=[10, 10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "explicit"}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit1D():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit"}, boundary_conditions=["PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit1D_GMRES():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit", "gmres": True}, boundary_conditions=["PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit2D():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit"}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit2D_GMRES():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit", "gmres": True}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit2D_ADI():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit", "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit2D_ADI_GMRES():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit", "gmres": True, "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit3D():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit"}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit3D_GMRES():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit", "gmres": True}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit3D_ADI():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit", "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_Implicit3D_ADI_GMRES():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "implicit", "gmres": True, "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson1D():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson"}, boundary_conditions=["PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson1D_GMRES():
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson", "gmres": True}, boundary_conditions=["PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson2D():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson"}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson2D_GMRES():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson", "gmres": True}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson2D_ADI():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson", "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson2D_ADI_GMRES():
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson", "gmres": True, "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson3D():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson"}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson3D_GMRES():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson", "gmres": True}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson3D_ADI():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson", "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0
    
def test_diffusion_CrankNicolson3D_ADI_GMRES():
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01, user_data={"D": 0.1, "solver": "crank-nicolson", "gmres": True, "adi": True}, boundary_conditions=["PERIODIC", "PERIODIC", "PERIODIC"])
    sim.initialize_engine()
    sim.simulate(2)
    assert sim.time_step_counter == 2
    assert len(sim.fields) > 0