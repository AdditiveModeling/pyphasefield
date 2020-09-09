import pyphasefield as ppf

def test_diffusion_default1dexplicit():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10])
    sim.simulate(2)
    
def test_diffusion_default2dexplicit():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10])
    sim.simulate(2)
    
def test_diffusion_default3dexplicit():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10, 10])
    sim.simulate(2)
    
def test_diffusion_Implicit1D():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10], solver="implicit")
    sim.simulate(2)
    
def test_diffusion_Implicit1D_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10], solver="implicit", gmres=True)
    sim.simulate(2)
    
def test_diffusion_Implicit2D():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="implicit")
    sim.simulate(2)
    
def test_diffusion_Implicit2D_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="implicit", gmres=True)
    sim.simulate(2)
    
def test_diffusion_Implicit2D_ADI():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="implicit", adi=True)
    sim.simulate(2)
    
def test_diffusion_Implicit2D_ADI_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="implicit", gmres=True, adi=True)
    sim.simulate(2)
    
def test_diffusion_Implicit3D():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="implicit")
    sim.simulate(2)
    
def test_diffusion_Implicit3D_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="implicit", gmres=True)
    sim.simulate(2)
    
def test_diffusion_Implicit3D_ADI():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="implicit", adi=True)
    sim.simulate(2)
    
def test_diffusion_Implicit3D_ADI_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="implicit", gmres=True, adi=True)
    sim.simulate(2)
    
def test_diffusion_CrankNicolson1D():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10], solver="crank-nicolson")
    sim.simulate(2)
    
def test_diffusion_CrankNicolson1D_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10], solver="crank-nicolson", gmres=True)
    sim.simulate(2)
    
def test_diffusion_CrankNicolson2D():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="crank-nicolson")
    sim.simulate(2)
    
def test_diffusion_CrankNicolson2D_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="crank-nicolson", gmres=True)
    sim.simulate(2)
    
def test_diffusion_CrankNicolson2D_ADI():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="crank-nicolson", adi=True)
    sim.simulate(2)
    
def test_diffusion_CrankNicolson2D_ADI_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([10, 10], solver="crank-nicolson", gmres=True, adi=True)
    sim.simulate(2)
    
def test_diffusion_CrankNicolson3D():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="crank-nicolson")
    sim.simulate(2)
    
def test_diffusion_CrankNicolson3D_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="crank-nicolson", gmres=True)
    sim.simulate(2)
    
def test_diffusion_CrankNicolson3D_ADI():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="crank-nicolson", adi=True)
    sim.simulate(2)
    
def test_diffusion_CrankNicolson3D_ADI_GMRES():
    sim = ppf.Simulation("test")
    sim.init_sim_Diffusion([5, 5, 5], solver="crank-nicolson", gmres=True, adi=True)
    sim.simulate(2)