import numpy as np
import matplotlib.pyplot as plt
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
        
try:
    from numba import cuda
    import numba
except:
    import pyphasefield.jit_placeholder as cuda
    import pyphasefield.jit_placeholder as numba


def n_fun(der1,der2):
    threshold=1e-8
    mag = (der1*der1+der2*der2)**(0.5)
    if mag < threshold:
        return 0
    else:
        return der1/mag
    
def Q(phi,k):
    # Assumes h = phi
    return (1-phi) / (1+k-(1-k)*phi)

def MAG2(derx, dery):
    return (derx*derx+dery*dery)*(derx*derx+dery*dery)

def A_fun(derx, dery, mag2, a_s, e_prime):
    threshold=1e-8
    if mag2 < threshold:
        return a_s, 0
    else:
        A = a_s * (1 + e_prime*(derx*derx*derx*derx + dery*dery*dery*dery)/mag2)
        derA = -4*a_s*e_prime * derx * dery * (derx*derx - dery*dery)/mag2
        return A,derA
        
def KarmaCPU(self):
    dt = self.dt
    dx = self.dx
    phi = self.fields[0].data
    c = self.fields[1].data
    eu = self.fields[2].data
    phi_new = np.zeros_like(phi)
    c_new = np.zeros_like(c)
    eu_new = np.zeros_like(eu)

    w = self.user_data['w']
    lambda_val = self.user_data['lambda_val']
    tau = self.user_data['tau']
    D = self.user_data['D']
    e4 = self.user_data['e4']
    k = self.user_data['k']
    a_t = self.user_data['a_t']

    inv_dx = 1./dx        
    a_s = 1-3*e4
    e_prime = 4*e4/a_s
    
    for i in range(1, phi.shape[0]-1):
        for j in range(1, phi.shape[1]-1):
            
            # Derivatives
            DERX_ipj = inv_dx * (phi[i+1][j] - phi[i][j])
            DERX_imj = inv_dx * (phi[i][j] - phi[i-1][j])
            DERY_ijp = inv_dx * (phi[i][j+1] - phi[i][j])
            DERY_ijm = inv_dx * (phi[i][j] - phi[i][j-1])
            
            DERX_ijp = 0.25*inv_dx * (phi[i+1][j+1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j+1])
            DERX_ijm = 0.25*inv_dx * (phi[i+1][j-1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j-1])
            DERY_ipj = 0.25*inv_dx * (phi[i+1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i+1][j-1])
            DERY_imj = 0.25*inv_dx * (phi[i-1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i-1][j-1])
            
            DERX_ij = 0.5 * (DERX_ipj + DERX_imj)
            DERY_ij = 0.5 * (DERY_ijp + DERY_ijm)
            
            # MAG2 terms
            MAG2_ipj = MAG2(DERX_ipj,DERY_ipj)
            MAG2_imj = MAG2(DERX_imj,DERY_imj)
            MAG2_ijp = MAG2(DERX_ijp,DERY_ijp) 
            MAG2_ijm = MAG2(DERX_ijm,DERY_ijm) 
            MAG2_ij = MAG2(DERX_ij,DERY_ij)
            
            # A and DERA terms
            A_ipj, DERA_ipj = A_fun(DERX_ipj,DERY_ipj,MAG2_ipj,a_s,e_prime)
            A_imj, DERA_imj = A_fun(DERX_imj,DERY_imj,MAG2_imj,a_s,e_prime)
            A_ijp, DERA_ijp = A_fun(DERX_ijp,DERY_ijp,MAG2_ijp,a_s,e_prime)
            A_ijm, DERA_ijm = A_fun(DERX_ijm,DERY_ijm,MAG2_ijm,a_s,e_prime)
            A_ij, DERA_ij = A_fun(DERX_ij,DERY_ij,MAG2_ij,a_s,e_prime)
            
            # Finite volume fluxes
            JR = A_ipj*(A_ipj*DERX_ipj - DERA_ipj*DERY_ipj)
            JL = A_imj*(A_imj*DERX_imj - DERA_imj*DERY_imj)
            JT = A_ijp*(A_ijp*DERY_ijp + DERA_ijp*DERX_ijp)
            JB = A_ijm*(A_ijm*DERY_ijm + DERA_ijm*DERX_ijm)
            
            # Interpolation functions
            f_prime = phi[i][j]*phi[i][j]*phi[i][j] - phi[i][j]
            #g_prime = 15/16 * (phi[i][j]**4 - 2*phi[i][j]**2 + 1)
            g_prime = (1-phi[i][j]*phi[i][j]) * (1-phi[i][j]*phi[i][j]) 
            
            dphi_dt = 1/(A_ij*A_ij) * (inv_dx*(JR-JL) + inv_dx*(JT-JB) - f_prime - lambda_val/(1-k)*(eu[i][j]-1)*g_prime)
            
            phi_new[i][j] = phi[i][j] + dt*dphi_dt
  
  
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
        
            # Derivatives
            DERX_ipj = inv_dx * (phi[i+1][j] - phi[i][j])
            DERX_imj = inv_dx * (phi[i][j] - phi[i-1][j])
            DERY_ijp = inv_dx * (phi[i][j+1] - phi[i][j])
            DERY_ijm = inv_dx * (phi[i][j] - phi[i][j-1])
            
            DERX_ijp = 0.25*inv_dx * (phi[i+1][j+1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j+1])
            DERX_ijm = 0.25*inv_dx * (phi[i+1][j-1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j-1])
            DERY_ipj = 0.25*inv_dx * (phi[i+1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i+1][j-1])
            DERY_imj = 0.25*inv_dx * (phi[i-1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i-1][j-1])
            
            # Interpolated values for finite volume cell edges
            phi_ipj = 0.5 * (phi[i+1][j] + phi[i][j])
            phi_imj = 0.5 * (phi[i][j] + phi[i-1][j])
            phi_ijp = 0.5 * (phi[i][j+1] + phi[i][j])
            phi_ijm = 0.5 * (phi[i][j] + phi[i][j-1])
            
            c_ipj = 0.5 * (c[i+1][j] + c[i][j])
            c_imj = 0.5 * (c[i][j] + c[i-1][j])
            c_ijp = 0.5 * (c[i][j+1] + c[i][j])
            c_ijm = 0.5 * (c[i][j] + c[i][j-1])
            
            eu_ipj = 0.5 * (eu[i+1][j] + eu[i][j])
            eu_imj = 0.5 * (eu[i][j] + eu[i-1][j])
            eu_ijp = 0.5 * (eu[i][j+1] + eu[i][j])
            eu_ijm = 0.5 * (eu[i][j] + eu[i][j-1])
            
            Q_ipj = Q(phi_ipj,k)
            Q_imj = Q(phi_imj,k)
            Q_ijp = Q(phi_ijp,k)
            Q_ijm = Q(phi_ijm,k)
            
            du_ipj = inv_dx/eu_ipj * (eu[i+1][j] - eu[i][j])
            du_imj = inv_dx/eu_imj * (-eu[i-1][j] + eu[i][j])
            du_ijp = inv_dx/eu_ijp * (eu[i][j+1] - eu[i][j])
            du_ijm = inv_dx/eu_ijm * (-eu[i][j-1] + eu[i][j])
            
            dphi_dt_ij = (phi_new[i][j] - phi[i][j])/dt
            
            dphi_dt_ipj = 0.5 * ((phi_new[i+1][j] - phi[i+1][j])/dt + dphi_dt_ij)
            dphi_dt_imj = 0.5 * ((phi_new[i-1][j] - phi[i-1][j])/dt + dphi_dt_ij)
            dphi_dt_ijp = 0.5 * ((phi_new[i][j+1] - phi[i][j+1])/dt + dphi_dt_ij)
            dphi_dt_ijm = 0.5 * ((phi_new[i][j-1] - phi[i][j-1])/dt + dphi_dt_ij)
            
            # Interface normals
            nr = n_fun(DERX_ipj,DERY_ipj)
            nl = n_fun(DERX_imj,DERY_imj)
            nt = n_fun(DERY_ijp,DERX_ijp)
            nb = n_fun(DERY_ijm,DERX_ijm)
            
            # Finite volume fluxes
            JR = -D * Q_ipj*c_ipj*du_ipj - a_t*(1-k)*(eu_ipj*dphi_dt_ipj)*nr
            JL = -D * Q_imj*c_imj*du_imj - a_t*(1-k)*(eu_imj*dphi_dt_imj)*nl
            JT = -D * Q_ijp*c_ijp*du_ijp - a_t*(1-k)*(eu_ijp*dphi_dt_ijp)*nt
            JB = -D * Q_ijm*c_ijm*du_ijm - a_t*(1-k)*(eu_ijm*dphi_dt_ijm)*nb
            
            dc_dt = -inv_dx*(JR-JL + JT-JB)
            c_new[i][j] = c[i][j] + dt*dc_dt
            
            eu_new[i][j] = 2*c_new[i][j] / (1+k-(1-k)*phi_new[i][j])
            
    phi.data = phi_new
    c.data = c_new
    eu.data = eu_new
    
    
    
@cuda.jit(Device=True)
def n_fun_GPU(der1,der2):
    threshold=1e-8
    mag = (der1*der1+der2*der2)**(0.5)
    if mag < threshold:
        return 0
    else:
        return der1/mag
        
@cuda.jit(Device=True)    
def Q_GPU(phi,k):
    # Assumes h = phi   --   (1-phi) / (1+k-(1-k)*h) 
    return (1-phi) / (1+k-(1-k)*phi)    
    
@cuda.jit(Device=True)
def MAG2_GPU(derx, dery):
    return (derx*derx+dery*dery)*(derx*derx+dery*dery)

@cuda.jit(Device=True)
def A_fun_GPU(derx, dery, mag2, a_s, e_prime):
    threshold=1e-8
    if mag2 < threshold:
        return a_s, 0
    else:
        A = a_s * (1 + e_prime*(derx*derx*derx*derx + dery*dery*dery*dery)/mag2)
        derA = -4*a_s*e_prime * derx * dery * (derx*derx - dery*dery)/mag2
        return A,derA


@cuda.jit
def kernel_phi_2DKarmaGPU(fields, fields_out, w, lambda_val, tau, D, dx, dt, e4, k, a_t):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    phi = fields[0]
    c = fields[1]
    eu = fields[2]
    phi_out = fields_out[0]
    c_out = fields_out[1]
    eu_out = fields_out[2]
    
    inv_dx = 1./dx        
    a_s = 1-3*e4
    e_prime = 4*e4/a_s
    
    for i in range(starty+1, phi.shape[0]-1, stridey):
        for j in range(startx+1, phi.shape[1]-1, stridex):
            
            # Finite volume derivatives
            DERX_ipj = inv_dx * (phi[i+1][j] - phi[i][j])
            DERX_imj = inv_dx * (phi[i][j] - phi[i-1][j])
            DERY_ijp = inv_dx * (phi[i][j+1] - phi[i][j])
            DERY_ijm = inv_dx * (phi[i][j] - phi[i][j-1])
            
            DERX_ijp = 0.25*inv_dx * (phi[i+1][j+1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j+1])
            DERX_ijm = 0.25*inv_dx * (phi[i+1][j-1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j-1])
            DERY_ipj = 0.25*inv_dx * (phi[i+1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i+1][j-1])
            DERY_imj = 0.25*inv_dx * (phi[i-1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i-1][j-1])
            
            DERX_ij = 0.5 * (DERX_ipj + DERX_imj)
            DERY_ij = 0.5 * (DERY_ijp + DERY_ijm)
            
            # MAG2 terms
            MAG2_ipj = MAG2_GPU(DERX_ipj,DERY_ipj)
            MAG2_imj = MAG2_GPU(DERX_imj,DERY_imj)
            MAG2_ijp = MAG2_GPU(DERX_ijp,DERY_ijp) 
            MAG2_ijm = MAG2_GPU(DERX_ijm,DERY_ijm) 
            MAG2_ij = MAG2_GPU(DERX_ij,DERY_ij)
            
            # A and DERA terms
            A_ipj, DERA_ipj = A_fun_GPU(DERX_ipj,DERY_ipj,MAG2_ipj,a_s,e_prime)
            A_imj, DERA_imj = A_fun_GPU(DERX_imj,DERY_imj,MAG2_imj,a_s,e_prime)
            A_ijp, DERA_ijp = A_fun_GPU(DERX_ijp,DERY_ijp,MAG2_ijp,a_s,e_prime)
            A_ijm, DERA_ijm = A_fun_GPU(DERX_ijm,DERY_ijm,MAG2_ijm,a_s,e_prime)
            A_ij, DERA_ij = A_fun_GPU(DERX_ij,DERY_ij,MAG2_ij,a_s,e_prime)
            
            # Finite volume fluxes    
            JR = A_ipj*(A_ipj*DERX_ipj - DERA_ipj*DERY_ipj)
            JL = A_imj*(A_imj*DERX_imj - DERA_imj*DERY_imj)
            JT = A_ijp*(A_ijp*DERY_ijp + DERA_ijp*DERX_ijp)
            JB = A_ijm*(A_ijm*DERY_ijm + DERA_ijm*DERX_ijm)
            
            # Interpolation functions
            f_prime = phi[i][j]*phi[i][j]*phi[i][j] - phi[i][j]
            #g_tilde_prime = 0.9375 * (phi[i][j]*phi[i][j]*phi[i][j]*phi[i][j] - 2*phi[i][j]*phi[i][j] + 1)
            #g_tilde_prime = (phi[i][j]*phi[i][j]*phi[i][j]*phi[i][j] - 2*phi[i][j]*phi[i][j] + 1)
            g_prime = (1-phi[i][j]*phi[i][j]) * (1-phi[i][j]*phi[i][j]) 
            dphi_dt = 1/(A_ij*A_ij) * (inv_dx*(JR-JL) + inv_dx*(JT-JB) - f_prime - lambda_val/(1-k)*(eu[i][j]-1)*g_prime)
            
            phi_out[i][j] = phi[i][j] + dt*dphi_dt


@cuda.jit
def kernel_c_2DKarmaGPU(fields, fields_out, w, lambda_val, tau, D, dx, dt, e4, k, a_t):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    phi = fields[0]
    c = fields[1]
    eu = fields[2]
    phi_out = fields_out[0]
    c_out = fields_out[1]
    eu_out = fields_out[2]
    
    inv_dx = 1./dx        
    a_s = 1-3*e4
    e_prime = 4*e4/a_s    
    for i in range(starty+1, c.shape[0]-1, stridey):
        for j in range(startx+1, c.shape[1]-1, stridex):
            
            DERX_ipj = inv_dx * (phi[i+1][j] - phi[i][j])
            DERX_imj = inv_dx * (phi[i][j] - phi[i-1][j])
            DERY_ijp = inv_dx * (phi[i][j+1] - phi[i][j])
            DERY_ijm = inv_dx * (phi[i][j] - phi[i][j-1])
            
            DERX_ijp = 0.25*inv_dx * (phi[i+1][j+1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j+1])
            DERX_ijm = 0.25*inv_dx * (phi[i+1][j-1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j-1])
            DERY_ipj = 0.25*inv_dx * (phi[i+1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i+1][j-1])
            DERY_imj = 0.25*inv_dx * (phi[i-1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i-1][j-1])
            
            # Interpolated values for finite volume cell edges
            phi_ipj = 0.5 * (phi[i+1][j] + phi[i][j])
            phi_imj = 0.5 * (phi[i][j] + phi[i-1][j])
            phi_ijp = 0.5 * (phi[i][j+1] + phi[i][j])
            phi_ijm = 0.5 * (phi[i][j] + phi[i][j-1])
            
            c_ipj = 0.5 * (c[i+1][j] + c[i][j])
            c_imj = 0.5 * (c[i][j] + c[i-1][j])
            c_ijp = 0.5 * (c[i][j+1] + c[i][j])
            c_ijm = 0.5 * (c[i][j] + c[i][j-1])
            
            eu_ipj = 0.5 * (eu[i+1][j] + eu[i][j])
            eu_imj = 0.5 * (eu[i][j] + eu[i-1][j])
            eu_ijp = 0.5 * (eu[i][j+1] + eu[i][j])
            eu_ijm = 0.5 * (eu[i][j] + eu[i][j-1])
            
            Q_ipj = Q_GPU(phi_ipj,k)
            Q_imj = Q_GPU(phi_imj,k)
            Q_ijp = Q_GPU(phi_ijp,k)
            Q_ijm = Q_GPU(phi_ijm,k)
            
            du_ipj = inv_dx/eu_ipj * (eu[i+1][j] - eu[i][j])
            du_imj = inv_dx/eu_imj * (-eu[i-1][j] + eu[i][j])
            du_ijp = inv_dx/eu_ijp * (eu[i][j+1] - eu[i][j])
            du_ijm = inv_dx/eu_ijm * (-eu[i][j-1] + eu[i][j])
            
            dphi_dt_ij = (phi_out[i][j] - phi[i][j])/dt
            
            dphi_dt_ipj = 0.5 * ((phi_out[i+1][j] - phi[i+1][j])/dt + dphi_dt_ij)
            dphi_dt_imj = 0.5 * ((phi_out[i-1][j] - phi[i-1][j])/dt + dphi_dt_ij)
            dphi_dt_ijp = 0.5 * ((phi_out[i][j+1] - phi[i][j+1])/dt + dphi_dt_ij)
            dphi_dt_ijm = 0.5 * ((phi_out[i][j-1] - phi[i][j-1])/dt + dphi_dt_ij)
            
            # Interface normals
            nr = n_fun_GPU(DERX_ipj,DERY_ipj)
            nl = n_fun_GPU(DERX_imj,DERY_imj)
            nt = n_fun_GPU(DERY_ijp,DERX_ijp)
            nb = n_fun_GPU(DERY_ijm,DERX_ijm)
            
            JR = -D * Q_ipj*c_ipj*du_ipj - a_t*(1-k)*(eu_ipj*dphi_dt_ipj)*nr
            JL = -D * Q_imj*c_imj*du_imj - a_t*(1-k)*(eu_imj*dphi_dt_imj)*nl
            JT = -D * Q_ijp*c_ijp*du_ijp - a_t*(1-k)*(eu_ijp*dphi_dt_ijp)*nt
            JB = -D * Q_ijm*c_ijm*du_ijm - a_t*(1-k)*(eu_ijm*dphi_dt_ijm)*nb
            
            dc_dt = -inv_dx*((JR-JL) + (JT-JB))
            c_out[i][j] = c[i][j] + dt*dc_dt
            
            eu_out[i][j] = 2*c_out[i][j] / (1+k-(1-k)*phi_out[i][j])
        

def engine_2DKarmaGPU(sim):
    cuda.synchronize()
    kernel_phi_2DKarmaGPU[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._fields_out_gpu_device, 
                                                                  sim.user_data['w'], sim.user_data['lambda_val'], sim.user_data['tau'], 
                                                                  sim.user_data['D'], sim.dx, sim.dt, sim.user_data['e4'], sim.user_data['k'], sim.user_data['a_t'])
    
    cuda.synchronize()
    
    kernel_c_2DKarmaGPU[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._fields_out_gpu_device, 
                                                                  sim.user_data['w'], sim.user_data['lambda_val'], sim.user_data['tau'], 
                                                                  sim.user_data['D'], sim.dx, sim.dt, sim.user_data['e4'], sim.user_data['k'], sim.user_data['a_t'])
    
    cuda.synchronize()



class Karma2001(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        phi = -1*np.ones(dim)
        c = np.ones(dim)
        eu = np.ones(dim)

        self.add_field(phi, 'phi')
        self.add_field(c, 'c')
        self.add_field(eu, 'eu')

    def initialize_fields_and_imported_data(self):
        super().initialize_fields_and_imported_data()

    def just_before_simulating(self):
        super().just_before_simulating()

    def simulation_loop(self):
        if self._uses_gpu:
            engine_2DKarmaGPU(self)
        else:
            KarmaCPU(self)
        
