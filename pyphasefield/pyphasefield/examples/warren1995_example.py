"""
This file runs a simulation according to the Warren model using the physical parameters
used in the paper.
"""
from pyphasefield.Engines import warren_eng, warren_diamond
import pyphasefield.simulate as sim
import pyphasefield.io as io

# Initialize function with physical parameters
nickel = sim.Component(T_m=1728, L=2350000000, S=0.37, B=0.0033)
copper = sim.Component(T_m=1358, L=1728000000, S=0.29, B=0.0039)
data = {'comp1': nickel,
        'comp2': copper,
        'D_S': 1e-13,
        'D_L': 1e-9,
        'v_m': 0.00000742,
        'y_e': 0.04,
        'T': 1574.,
        'alpha': 0.3,
        'cell_spacing': 4.6e-8}

# Initialize fields and engine
fields = warren_diamond([200, 200], 15)
eng = warren_eng(**data)

# Run simulation
sim.run(fields, engine=eng, steps=500, output=50)

# Post processing
io.plot_field(fields['c'], "c", 50, data['cell_spacing'])
