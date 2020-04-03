import numpy as np
from ..field import Field

def __p(phi):
    return phi*phi*phi*(10-15*phi+6*phi*phi)

def __g(phi):
    return (phi*phi*(1-phi)*(1-phi))

def __gprime(phi):
    return (4*phi*phi*phi - 6*phi*phi +2*phi)

_p = np.vectorize(__p)
_g = np.vectorize(__g) 
_gprime = np.vectorize(__gprime)

def gradx(phi, dx):
    phim = np.roll(phi, -1, 0)
    phip = np.roll(phi, 1, 0)
    return (phip-phim)/(2*dx)

def grady(phi, dx):
    phim = np.roll(phi, -1, 1)
    phip = np.roll(phi, 1, 1)
    return (phip-phim)/(2*dx)

def gradxx(phi, dx):
    phim = np.roll(phi, -1, 0)
    phip = np.roll(phi, 1, 0)
    return (phip+phim-2*phi)/(dx*dx)

def gradyy(phi, dx):
    phim = np.roll(phi, -1, 1)
    phip = np.roll(phi, 1, 1)
    return (phip+phim-2*phi)/(dx*dx)

def Warren1995(sim):
    dt = sim.get_time_step_length()
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    c = sim.fields[1].data
    
    g = _g(phi)
    p = _p(phi)
    gprime = _gprime(phi)
    H_A = sim.W_A*gprime + 30*sim.L_A*(1/sim.T-1/sim.T_mA)*g
    H_B = sim.W_B*gprime + 30*sim.L_B*(1/sim.T-1/sim.T_mB)*g
    phixx = gradxx(phi, dx)
    phiyy = gradyy(phi, dx)
    lphi = phixx+phiyy
    phix = gradx(phi, dx)
    phiy = grady(phi, dx)
    phixy = grady(phix, dx)
    
    #change in c
    D_C = sim.D_S+p*(sim.D_L-sim.D_S)
    temp = D_C*sim.v_m*c*(1-c)*(H_B-H_A)/sim.R
    deltac = D_C*(gradxx(c, dx)+gradyy(c, dx))+(gradx(D_C, dx)*gradx(c, dx)+grady(D_C, dx)*grady(c, dx))+temp*(lphi)+(gradx(temp, dx)*phix+grady(temp, dx)*phiy)
    #print(deltac)
    #print(temp)
    
    #change in phi
    theta = np.arctan2(phiy, phix)
    eta = 1+sim.y_e*np.cos(4*theta)
    etap = -4*sim.y_e*np.sin(4*theta)
    etapp = -16*(eta-1)
    c2 = np.cos(2*theta)
    s2 = np.sin(2*theta)
    M_phi = (1-c)*sim.M_A + c*sim.M_B
    ebar2 = sim.ebar**2
    deltaphi = M_phi*((ebar2*eta*eta*lphi-(1-c)*H_A-c*H_B)+ebar2*eta*etap*(s2*(phiyy-phixx)+2*c2*phixy)+0.5*ebar2*(etap*etap+eta*etapp)*(-2*s2*phixy+lphi+c2*(phiyy-phixx)))
    randArray = 2*np.random.random(phi.shape)-1
    deltaphi += M_phi*sim.alpha*randArray*(16*g)*((1-c)*H_A+c*H_B)
    np.set_printoptions(threshold=np.inf)
    
    
    #apply changes
    sim.fields[0].data += deltaphi*dt
    sim.fields[1].data += deltac*dt
    
def init_Warren1995(sim, dim, diamond_size=15):
    #original Warren1995 model uses centimeters, values have been converted to meters!
    sim.set_dimensions(dim)
    phi = np.zeros(dim)
    phi += 1.
    for i in range(diamond_size):
        phi[(int)(dim[0]/2-i):(int)(dim[0]/2+i), ((int)(dim[1]/2-(diamond_size-i))):(int)(dim[1]/2+(diamond_size-i))] = 0
    phi_field = Field(data=phi, name="phi", simulation=sim)
    sim.add_field(phi_field)
    c = np.zeros(dim)
    c += 0.40831
    c_field = Field(data=c, name="c", simulation=sim)
    sim.add_field(c_field)
    sim.T_mA = 1728. #melting point of nickel
    sim.T_mB = 1358. #melting point of copper
    sim.L_A = 2350000000. #latent heat of nickel, J/m^3
    sim.L_B = 1728000000. #latent heat of copper, J/m^3
    sim.s_A = 0.37 #surface energy of nickel, J/m^2
    sim.s_B = 0.29 #surface energy of copper, J/m^2
    sim.D_L = 1e-9 #diffusion in liquid, m^2/s
    sim.D_S = 1e-13 #diffusion in solid, m^2/s
    sim.B_A = 0.0033 #linear kinetic coefficient of nickel, m/K/s
    sim.B_B = 0.0039 #linear kinetic coefficient of copper, m/K/s
    sim.v_m = 0.00000742 #molar volume, m^3/mol
    sim.R = 8.314 #gas constant, J/mol*K
    sim.y_e = 0.04 #anisotropy
    sim.T = 1574.
    sim.alpha = 0.3
    sim.set_cell_spacing(4.6e-8)
    sim.set_time_step_length(sim.get_cell_spacing()**2/5./sim.D_L)
    sim.d = sim.get_cell_spacing()/0.94 #interfacial thickness
    sim.ebar = np.sqrt(6*np.sqrt(2)*sim.s_A*sim.d/sim.T_mA) #baseline energy
    sim.W_A = 3*sim.s_A/(np.sqrt(2)*sim.T_mA*sim.d)
    sim.W_B = 3*sim.s_B/(np.sqrt(2)*sim.T_mB*sim.d)
    sim.M_A = sim.T_mA*sim.T_mA*sim.B_A/(6*np.sqrt(2)*sim.L_A*sim.d)
    sim.M_B = sim.T_mB*sim.T_mB*sim.B_B/(6*np.sqrt(2)*sim.L_B*sim.d)
    sim.ebar = np.sqrt(6*np.sqrt(2)*sim.s_A*sim.d/sim.T_mA)
    sim.set_engine(Warren1995)