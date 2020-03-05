"""
This file runs a simulation according to the Warren model using the physical parameters
used in the paper.
"""
from pyphasefield.Engines import warren_eng, warren_diamond
from pyphasefield.Engines import Component, run_warren
from pyphasefield.io import plot_field

# Initialize function with physical parameters
nickel = Component(T_m=1728, L=2350000000, S=0.37, B=0.0033)
copper = Component(T_m=1358, L=1728000000, S=0.29, B=0.0039)
data = {'comp1': nickel,
        'comp2': copper,
        'D_s': 1e-13,
        'D_l': 1e-9,
        'v_m': 0.00000742,
        'y_e': 0.04,
        'T': 1574.,
        'alpha': 0.3,
        'cell_spacing': 4.6e-8}

# Initialize fields and engine
fields = warren_diamond((200, 200), 15)
eng = warren_eng(**data)
# Run simulation
run_warren(fields.phi, fields.c, eng, 1000, output=50)
# Post processing
# plot_field(fields['phi'], "phi", 500, data['cell_spacing'])
