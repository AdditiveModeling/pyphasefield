import pytest
import pyphasefield as ppf
from pyphasefield.Engines.Diffusion import Diffusion


def test_boundary_conditions_validation_1d():
    """Test that 1D simulations accept correct BC formats"""
    # Valid: single string
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01, 
                   boundary_conditions="PERIODIC")
    assert sim._boundary_conditions_type == [["PERIODIC", "PERIODIC"]]
    
    # Valid: list with 1 entry
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01,
                   boundary_conditions=["NEUMANN"])
    assert sim._boundary_conditions_type == [["NEUMANN", "NEUMANN"]]
    
    # Valid: list with 2 entries (left and right)
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01,
                   boundary_conditions=["PERIODIC", "DIRICHLET"])
    assert sim._boundary_conditions_type == [["PERIODIC", "DIRICHLET"]]
    
    # Invalid: wrong number of entries
    with pytest.raises(ValueError, match="Number of boundary conditions"):
        sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01,
                       boundary_conditions=["PERIODIC", "PERIODIC", "NEUMANN"])


def test_boundary_conditions_validation_2d():
    """Test that 2D simulations accept correct BC formats"""
    # Valid: single string
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                   boundary_conditions="PERIODIC")
    assert sim._boundary_conditions_type == [["PERIODIC", "PERIODIC"], 
                                            ["PERIODIC", "PERIODIC"]]
    
    # Valid: list with 2 entries (one per dimension)
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                   boundary_conditions=["PERIODIC", "NEUMANN"])
    assert sim._boundary_conditions_type == [["PERIODIC", "PERIODIC"], 
                                            ["NEUMANN", "NEUMANN"]]
    
    # Valid: list with 4 entries (left/right for each dimension)
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                   boundary_conditions=["PERIODIC", "NEUMANN", "DIRICHLET", "PERIODIC"])
    assert sim._boundary_conditions_type == [["PERIODIC", "NEUMANN"], 
                                            ["DIRICHLET", "PERIODIC"]]
    
    # Valid: 2D list format
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                   boundary_conditions=[["PERIODIC", "NEUMANN"], ["DIRICHLET", "PERIODIC"]])
    assert sim._boundary_conditions_type == [["PERIODIC", "NEUMANN"], 
                                            ["DIRICHLET", "PERIODIC"]]
    
    # Invalid: wrong number of entries
    with pytest.raises(ValueError, match="Number of boundary conditions"):
        sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                       boundary_conditions=["PERIODIC"])
    
    with pytest.raises(ValueError, match="Number of boundary conditions"):
        sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                       boundary_conditions=["PERIODIC", "NEUMANN", "DIRICHLET"])


def test_boundary_conditions_validation_3d():
    """Test that 3D simulations accept correct BC formats"""
    # Valid: single string
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01,
                   boundary_conditions="PERIODIC")
    assert sim._boundary_conditions_type == [["PERIODIC", "PERIODIC"], 
                                            ["PERIODIC", "PERIODIC"],
                                            ["PERIODIC", "PERIODIC"]]
    
    # Valid: list with 3 entries (one per dimension)
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01,
                   boundary_conditions=["PERIODIC", "NEUMANN", "DIRICHLET"])
    assert sim._boundary_conditions_type == [["PERIODIC", "PERIODIC"], 
                                            ["NEUMANN", "NEUMANN"],
                                            ["DIRICHLET", "DIRICHLET"]]
    
    # Valid: list with 6 entries (left/right for each dimension)
    sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01,
                   boundary_conditions=["PERIODIC", "NEUMANN", "DIRICHLET", 
                                      "PERIODIC", "NEUMANN", "DIRICHLET"])
    assert sim._boundary_conditions_type == [["PERIODIC", "NEUMANN"], 
                                            ["DIRICHLET", "PERIODIC"],
                                            ["NEUMANN", "DIRICHLET"]]
    
    # Invalid: wrong number of entries
    with pytest.raises(ValueError, match="Number of boundary conditions"):
        sim = Diffusion(dimensions=[5, 5, 5], dx=1.0, dt=0.01,
                       boundary_conditions=["PERIODIC", "PERIODIC"])


def test_boundary_conditions_dirchlet_correction():
    """Test that DIRCHLET is automatically corrected to DIRICHLET"""
    # Single string
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01,
                   boundary_conditions="DIRCHLET")
    assert sim._boundary_conditions_type == [["DIRICHLET", "DIRICHLET"]]
    
    # In list
    sim = Diffusion(dimensions=[10], dx=1.0, dt=0.01,
                   boundary_conditions=["DIRCHLET"])
    assert sim._boundary_conditions_type == [["DIRICHLET", "DIRICHLET"]]
    
    # Mixed in list
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                   boundary_conditions=["PERIODIC", "DIRCHLET"])
    assert sim._boundary_conditions_type == [["PERIODIC", "PERIODIC"], 
                                            ["DIRICHLET", "DIRICHLET"]]
    
    # In 2D list
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                   boundary_conditions=[["PERIODIC", "DIRCHLET"], ["DIRCHLET", "PERIODIC"]])
    assert sim._boundary_conditions_type == [["PERIODIC", "DIRICHLET"], 
                                            ["DIRICHLET", "PERIODIC"]]


def test_boundary_conditions_runtime_change():
    """Test that boundary conditions can be changed after initialization"""
    sim = Diffusion(dimensions=[10, 10], dx=1.0, dt=0.01,
                   boundary_conditions="PERIODIC")
    
    # Change to different format
    sim.set_boundary_conditions(["NEUMANN", "DIRICHLET"])
    assert sim._boundary_conditions_type == [["NEUMANN", "NEUMANN"], 
                                            ["DIRICHLET", "DIRICHLET"]]
    
    # Invalid change should raise error
    with pytest.raises(ValueError, match="Number of boundary conditions"):
        sim.set_boundary_conditions(["PERIODIC"])