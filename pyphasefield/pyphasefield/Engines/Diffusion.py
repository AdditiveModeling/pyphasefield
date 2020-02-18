import numpy as np
from ..field import Field


def Diffusion(sim):
    dt = sim._time_step_in_seconds
    c = sim.fields[0]
    dc = dt * (sim.D * c.laplacian())
    sim.fields[0].data += dc

def init_Diffusion(sim, dim):
    sim.D = 0.1
    sim.set_dimensions(dim)
    sim.set_cell_spacing(1.)
    c = np.zeros(dim)
    length = dim[0]
    c[length // 4:3 * length // 4] = 1
    c_field = Field(data=c, name="c", simulation=sim)
    sim.add_field(c_field)
    sim.set_engine(Diffusion)