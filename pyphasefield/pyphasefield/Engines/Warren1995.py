import numpy as np
import math
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE_INV
        
try:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except:
    import pyphasefield.jit_placeholder as cuda

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

def engine_Warren1995(sim):
    dt = sim.get_time_step_length()
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    c = sim.fields[1].data
    T = sim.temperature.data
    
    g = _g(phi)
    p = _p(phi)
    gprime = _gprime(phi)
    
    R = 8.314
    
    W_A = sim.user_data["W_A"]
    W_B = sim.user_data["W_B"]
    M_A = sim.user_data["M_A"]
    M_B = sim.user_data["M_B"]
    L_A = sim.user_data["L_A"]
    L_B = sim.user_data["L_B"]
    T_MA = sim.user_data["T_MA"]
    T_MB = sim.user_data["T_MB"]
    D_S = sim.user_data["D_S"]
    D_L = sim.user_data["D_L"]
    v_m = sim.user_data["v_m"]
    ebar2 = sim.user_data["ebar"]**2
    alpha = sim.user_data["alpha"]
    y_e = sim.user_data["y_e"]
    
    H_A = W_A*gprime + 30*L_A*(1/T-1/T_MA)*g
    H_B = W_B*gprime + 30*L_B*(1/T-1/T_MB)*g
    phixx = gradxx(phi, dx)
    phiyy = gradyy(phi, dx)
    lphi = phixx+phiyy
    phix = gradx(phi, dx)
    phiy = grady(phi, dx)
    phixy = grady(phix, dx)
    
    #change in c
    D_C = D_S+p*(D_L-D_S)
    temp = D_C*v_m*c*(1-c)*(H_B-H_A)/R
    deltac = D_C*(gradxx(c, dx)+gradyy(c, dx))+(gradx(D_C, dx)*gradx(c, dx)+grady(D_C, dx)*grady(c, dx))+temp*(lphi)+(gradx(temp, dx)*phix+grady(temp, dx)*phiy)
    #print(deltac)
    #print(temp)
    
    #change in phi
    theta = np.arctan2(phiy, phix)
    eta = 1+y_e*np.cos(4*theta)
    etap = -4*y_e*np.sin(4*theta)
    etapp = -16*(eta-1)
    c2 = np.cos(2*theta)
    s2 = np.sin(2*theta)
    M_phi = (1-c)*M_A + c*M_B
    deltaphi = M_phi*((ebar2*eta*eta*lphi-(1-c)*H_A-c*H_B)+ebar2*eta*etap*(s2*(phiyy-phixx)+2*c2*phixy)+0.5*ebar2*(etap*etap+eta*etapp)*(-2*s2*phixy+lphi+c2*(phiyy-phixx)))
    randArray = 2*np.random.random(phi.shape)-1
    deltaphi += M_phi*alpha*randArray*(16*g)*((1-c)*H_A+c*H_B)
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
    phi_field = Field(data=phi, name="phi", simulation=sim, colormap=COLORMAP_PHASE_INV)
    sim.add_field(phi_field)
    c = np.zeros(dim)
    c += 0.40831
    c_field = Field(data=c, name="c", simulation=sim, colormap=COLORMAP_OTHER)
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
    
@cuda.jit
def Warren1995_kernel(fields, T, transfer, fields_out, rng_states, params, c_params):
    
    startx, starty = cuda.grid(2)    
    stridex, stridey = cuda.gridsize(2) 
    thread_id = startx + starty*stridex
    
    R = 8.314
    dx = params[0]
    dt = params[1]
    d = params[2]
    y_e = params[4]
    D_L = params[5]
    D_S = params[6]
    ebar = params[7]
    a = params[8]
    
    L_A = c_params[0][0]
    L_B = c_params[0][1]
    T_mA = c_params[1][0]
    T_mB = c_params[1][1]
    W_A = c_params[4][0]
    W_B = c_params[4][1]
    M_A = c_params[5][0]
    M_B = c_params[5][1]
    
    D = transfer[0]
    temp = transfer[1]
    
    phi = fields[0]
    c = fields[1]
    phi_out = fields_out[0]
    c_out = fields_out[1]
    
    e2 = ebar*ebar

    # assuming x and y inputs are same length
    for i in range(starty+1, phi.shape[0]-1, stridey):
        for j in range(startx+1, phi.shape[1]-1, stridex):
            g = (phi[i][j]**2)*(1-phi[i][j])**2
            gprime = 4*phi[i][j]**3 - 6*phi[i][j]**2 + 2*phi[i][j]
            H_A = W_A*gprime + 30.*g*L_A*(1./T[i][j]-1./T_mA)
            H_B = W_B*gprime + 30.*g*L_B*(1./T[i][j]-1./T_mB)
            
            idx = 1./dx
            
            dphidx = 0.5*(phi[i][j+1]-phi[i][j-1])*idx
            dphidy = 0.5*(phi[i+1][j]-phi[i-1][j])*idx
            d2phidx2 = (phi[i][j+1]+phi[i][j-1]-2*phi[i][j])*idx*idx
            d2phidy2 = (phi[i+1][j]+phi[i-1][j]-2*phi[i][j])*idx*idx
            lphi = d2phidx2 + d2phidy2
            d2phidxy = 0.25*(phi[i+1][j+1]-phi[i+1][j-1]-phi[i-1][j+1]+phi[i-1][j-1])*idx*idx
            theta = math.atan2(dphidy, dphidx)
            eta = 1+y_e*math.cos(4*theta)
            etaprime = -4*y_e*math.sin(4*theta)
            etadoubleprime = -16*(eta-1)
            
            M_phi = (1-c[i][j])*M_A + c[i][j]*M_B
            
            dphidt = e2*eta*eta*lphi - (1.-c[i][j])*H_A - c[i][j]*H_B 
            dphidt += e2*eta*etaprime*(math.sin(2.*theta)*(d2phidy2-d2phidx2) + 2.*math.cos(2.*theta)*d2phidxy)
            dphidt -= 0.5*e2*(etaprime*etaprime + eta*etadoubleprime)*(2.*math.sin(2.*theta)*d2phidxy - lphi - math.cos(2.*theta)*(d2phidy2-d2phidx2))
            random = xoroshiro128p_uniform_float32(rng_states, thread_id)
            dphidt += a*16*g*((1-c[i][j])*H_A + c[i][j]*H_B)*(2.*random-1.)
            dphidt *= M_phi
            
            #dcdt = D[i][j]*(c[i][j+1]+c[i][j-1]+c[i+1][j]+c[i-1][j]-4*c[i][j])
            #dcdt += 0.25*(D[i][j+1]-D[i][j-1])*(c[i][j+1]-c[i][j-1])
            #dcdt += 0.25*(D[i+1][j]-D[i-1][j])*(c[i+1][j]-c[i-1][j])
            #dcdt *= (idx*idx)
            #dcdt += temp[i][j]*lphi
            #dcdt += 0.5*(temp[i][j+1]-temp[i][j-1])*dphidx*idx
            #dcdt += 0.5*(temp[i+1][j]-temp[i-1][j])*dphidy*idx
            
            dcdt = 0.5*idx*idx*((D[i+1][j] + D[i][j])*(c[i+1][j] - c[i][j]) - (D[i][j] + D[i-1][j])*(c[i][j] - c[i-1][j]))
            dcdt += 0.5*idx*idx*((D[i][j+1] + D[i][j])*(c[i][j+1] - c[i][j]) - (D[i][j] + D[i][j-1])*(c[i][j] - c[i][j-1]))
            dcdt += 0.5*idx*idx*((temp[i+1][j] + temp[i][j])*(phi[i+1][j] - phi[i][j]) - (temp[i][j] + temp[i-1][j])*(phi[i][j] - phi[i-1][j]))
            dcdt += 0.5*idx*idx*((temp[i][j+1] + temp[i][j])*(phi[i][j+1] - phi[i][j]) - (temp[i][j] + temp[i][j-1])*(phi[i][j] - phi[i][j-1]))
            
            phi_out[i][j] = phi[i][j] + dphidt*dt
            c_out[i][j] = c[i][j] + dcdt*dt
            
@cuda.jit
def Warren1995_helper_kernel(fields, T, transfer, rng_states, params, c_params):
    startx, starty = cuda.grid(2)     
    stridex, stridey = cuda.gridsize(2) 
    
    phi = fields[0]
    c = fields[1]
    
    R = 8.314
    v_m = params[3]
    D_L = params[5]
    D_S = params[6]
    
    L_A = c_params[0][0]
    L_B = c_params[0][1]
    T_mA = c_params[1][0]
    T_mB = c_params[1][1]
    W_A = c_params[4][0]
    W_B = c_params[4][1]
    
    
    
    D = transfer[0]
    temp = transfer[1]
    
    
    
    for i in range(startx, phi.shape[0], stridex):
        for j in range(starty, phi.shape[1], stridey):
            g = (phi[i][j]**2)*(1.-phi[i][j])**2
            gprime = 4.*phi[i][j]**3 - 6.*phi[i][j]**2 + 2.*phi[i][j]
            p = (phi[i][j]**3)*(10.-15.*phi[i][j]+6.*phi[i][j]**2)
            H_A = W_A*gprime + 30.*g*L_A*(1./T[i][j]-1./T_mA)
            H_B = W_B*gprime + 30.*g*L_B*(1./T[i][j]-1./T_mB)
            D[i][j] = D_S+p*(D_L-D_S)
            temp[i][j] = D[i][j]*v_m*c[i][j]*(1-c[i][j])*(H_B-H_A)/R
    
class Warren1995(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #additional initialization code goes below
        #runs *before* tdb, thermal, fields, and boundary conditions are loaded/initialized
        
        
    def init_fields(self):
        #initialization of fields code goes here
        #runs *after* tdb and thermal data is loaded/initialized
        #runs *before* boundary conditions are initialized
        if not ("T_MA" in self.user_data):
            self.user_data["T_MA"] = 1728. #melting point of nickel
        if not ("T_MB" in self.user_data):
            self.user_data["T_MB"] = 1358. #melting point of copper
        if not ("L_A" in self.user_data):
            self.user_data["L_A"] = 2350000000. #latent heat of nickel, J/m^3
        if not ("L_B" in self.user_data):
            self.user_data["L_B"] = 1728000000. #latent heat of copper, J/m^3
        if not ("S_A" in self.user_data):
            self.user_data["S_A"] = 0.37 #surface energy of nickel, J/m^2
        if not ("S_B" in self.user_data):
            self.user_data["S_B"] = 0.29 #surface energy of copper, J/m^2
        if not ("D_L" in self.user_data):
            self.user_data["D_L"] = 1e-9 #diffusion in liquid, m^2/s
        if not ("D_S" in self.user_data):
            self.user_data["D_S"] = 1e-13 #diffusion in solid, m^2/s
        if not ("B_A" in self.user_data):
            self.user_data["B_A"] = 0.0033 #linear kinetic coefficient of nickel, m/K/s
        if not ("B_B" in self.user_data):
            self.user_data["B_B"] = 0.0039 #linear kinetic coefficient of copper, m/K/s
        if not ("v_m" in self.user_data):
            self.user_data["v_m"] = 0.00000742 #molar volume, m^3/mol
        if not ("y_e" in self.user_data):
            self.user_data["y_e"] = 0.04 #anisotropy
        if not ("alpha" in self.user_data):
            self.user_data["alpha"] = 0.3
        if not ("diamond_size" in self.user_data):
            self.user_data["diamond_size"] = 15
        if not ("initial_concentration" in self.user_data):
            self.user_data["initial_concentration"] = 0.40831
        self._num_transfer_arrays = 2
        dim = self.dimensions
        phi = np.zeros(dim)
        phi += 1.
        diamond_size = self.user_data["diamond_size"]
        for i in range(diamond_size):
            phi[(int)(dim[0]/2-i):(int)(dim[0]/2+i), ((int)(dim[1]/2-(diamond_size-i))):(int)(dim[1]/2+(diamond_size-i))] = 0
        self.add_field(phi, "phi", colormap=COLORMAP_PHASE_INV)
        c = np.zeros(dim)
        c += self.user_data["initial_concentration"]
        self.add_field(c, "c", colormap=COLORMAP_OTHER)
                        
    def just_before_simulating(self):
        super().just_before_simulating()
        #additional code to run just before beginning the simulation goes below
        #runs immediately before simulating, no manual changes permitted to changes implemented here
        self.set_time_step_length(self.get_cell_spacing()**2/5./self.user_data["D_L"])
        self.user_data["d"] = self.get_cell_spacing()/0.94 #interfacial thickness
        self.user_data["ebar"] = np.sqrt(6*np.sqrt(2)*self.user_data["S_A"]*self.user_data["d"]/self.user_data["T_MA"]) #baseline energy
        self.user_data["W_A"] = 3*self.user_data["S_A"]/(np.sqrt(2)*self.user_data["T_MA"]*self.user_data["d"])
        self.user_data["W_B"] = 3*self.user_data["S_B"]/(np.sqrt(2)*self.user_data["T_MB"]*self.user_data["d"])
        self.user_data["M_A"] = (self.user_data["T_MA"]**2)*self.user_data["B_A"]/(6*np.sqrt(2)*self.user_data["L_A"]*self.user_data["d"])
        self.user_data["M_B"] = (self.user_data["T_MB"]**2)*self.user_data["B_B"]/(6*np.sqrt(2)*self.user_data["L_B"]*self.user_data["d"])
        if(self._uses_gpu):
            params = []
            c_params = []
            params.append(self.dx)
            params.append(self.dt)
            params.append(self.user_data["d"])
            params.append(self.user_data["v_m"])
            params.append(self.user_data["y_e"])
            params.append(self.user_data["D_L"])
            params.append(self.user_data["D_S"])
            params.append(self.user_data["ebar"])
            params.append(self.user_data["alpha"])
            c_params.append([self.user_data["L_A"], self.user_data["L_B"]])
            c_params.append([self.user_data["T_MA"], self.user_data["T_MB"]])
            c_params.append([self.user_data["S_A"], self.user_data["S_B"]])
            c_params.append([self.user_data["B_A"], self.user_data["B_B"]])
            c_params.append([self.user_data["W_A"], self.user_data["W_B"]])
            c_params.append([self.user_data["M_A"], self.user_data["M_B"]])
            self.user_data["params"] = cuda.to_device(np.array(params))
            self.user_data["c_params"] = cuda.to_device(np.array(c_params))
            self.user_data["rng_states"] = create_xoroshiro128p_states(256*256, seed=1)
        
    def simulation_loop(self):
        #code to run each simulation step goes here
        if(self._uses_gpu):
            cuda.synchronize()
            Warren1995_helper_kernel[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            Warren1995_kernel[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            self._fields_gpu_device, self._fields_out_gpu_device = self._fields_out_gpu_device, self._fields_gpu_device
        else:
            engine_Warren1995(self)