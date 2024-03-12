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
        
def KarmaRappelCPU(self):
    dt = self.dt
    dx = self.dx
    phi = self.fields[0].data
    u = self.fields[1].data
    phi_new = np.zeros_like(phi)
    u_new = np.zeros_like(u)

    w = self.user_data['w']
    lambda_val = self.user_data['lambda_val']
    tau = self.user_data['tau']
    D = self.user_data['D']
    e4 = self.user_data['e4']

    inv_dx = 1./dx        
    a_s = 1-3*e4
    e_prime = 4*e4/a_s
    
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            
            # Finite volume derivatives
            DERX_ipj = inv_dx * (phi[i+1][j] - phi[i][j])
            DERX_imj = inv_dx * (phi[i][j] - phi[i-1][j])
            DERY_ijp = inv_dx * (phi[i][j+1] - phi[i][j])
            DERY_ijm = inv_dx * (phi[i][j] - phi[i][j-1])
            
            DERX_ijp = 0.25*inv_dx * (phi[i+1][j+1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j+1])
            DERX_ijm = 0.25*inv_dx * (phi[i+1][j-1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j-1])
            DERY_ipj = 0.25*inv_dx * (phi[i+1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i+1][j-1])
            DERY_imj = 0.25*inv_dx * (phi[i-1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i-1][j-1])
            
            DERX_ij = 0.5*inv_dx * (DERX_ipj + DERX_imj)
            DERY_ij = 0.5*inv_dx * (DERY_ijp + DERY_ijm)
            
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
            
                
            JR = A_ipj*(A_ipj*DERX_ipj - DERA_ipj*DERY_ipj)
            JL = A_imj*(A_imj*DERX_imj - DERA_imj*DERY_imj)
            JT = A_ijp*(A_ijp*DERY_ijp + DERA_ijp*DERX_ijp)
            JB = A_ijm*(A_ijm*DERY_ijm + DERA_ijm*DERX_ijm)
            
            # Interpolation functions
            g_prime = phi[i][j]*phi[i][j]*phi[i][j] - phi[i][j]
            P_prime = (1-phi[i][j]*phi[i][j])*(1-phi[i][j]*phi[i][j])
            h_prime = 0.5
            
            dphi_dt = 1/(A_ij*A_ij) * (inv_dx*(JR-JL) + inv_dx*(JT-JB) - g_prime - lambda_val*u[i][j]*P_prime)
            du_dt = D * inv_dx*inv_dx * (u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1] - 4*u[i][j]) + h_prime * dphi_dt
            
            phi_new[i][j] = phi[i][j] + dt*dphi_dt
            u_new[i][j] = u[i][j] + dt*du_dt    
                
    phi.data = phi_new
    u.data = u_new
    
    
    
    
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
def kernel_2DKarmaRappelGPU(fields, fields_out, w, lambda_val, tau, D, dx, dt, e4):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    phi = fields[0]
    u = fields[1]
    phi_out = fields_out[0]
    u_out = fields_out[1]
    
    inv_dx = 1./dx        
    a_s = 1-3*e4
    e_prime = 4*e4/a_s
    
    for i in range(starty+1, u.shape[0]-1, stridey):
        for j in range(startx+1, u.shape[1]-1, stridex):
            
            # Finite volume derivatives
            DERX_ipj = inv_dx * (phi[i+1][j] - phi[i][j])
            DERX_imj = inv_dx * (phi[i][j] - phi[i-1][j])
            DERY_ijp = inv_dx * (phi[i][j+1] - phi[i][j])
            DERY_ijm = inv_dx * (phi[i][j] - phi[i][j-1])
            
            DERX_ijp = 0.25*inv_dx * (phi[i+1][j+1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j+1])
            DERX_ijm = 0.25*inv_dx * (phi[i+1][j-1] + phi[i+1][j] - phi[i-1][j] - phi[i-1][j-1])
            DERY_ipj = 0.25*inv_dx * (phi[i+1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i+1][j-1])
            DERY_imj = 0.25*inv_dx * (phi[i-1][j+1] + phi[i][j+1] - phi[i][j-1] - phi[i-1][j-1])
            
            DERX_ij = 0.5*inv_dx * (DERX_ipj + DERX_imj)
            DERY_ij = 0.5*inv_dx * (DERY_ijp + DERY_ijm)
            
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
            
                
            JR = A_ipj*(A_ipj*DERX_ipj - DERA_ipj*DERY_ipj)
            JL = A_imj*(A_imj*DERX_imj - DERA_imj*DERY_imj)
            JT = A_ijp*(A_ijp*DERY_ijp + DERA_ijp*DERX_ijp)
            JB = A_ijm*(A_ijm*DERY_ijm + DERA_ijm*DERX_ijm)
            
            # Interpolation functions
            g_prime = phi[i][j]*phi[i][j]*phi[i][j] - phi[i][j]
            P_prime = (1-phi[i][j]*phi[i][j])*(1-phi[i][j]*phi[i][j])
            h_prime = 0.5
            
            dphi_dt = 1/(A_ij*A_ij) * (inv_dx*(JR-JL) + inv_dx*(JT-JB) - g_prime - lambda_val*u[i][j]*P_prime)
            du_dt = D * inv_dx*inv_dx * (u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1] - 4*u[i][j]) + h_prime * dphi_dt
            
            phi_out[i][j] = phi[i][j] + dt*dphi_dt
            u_out[i][j] = u[i][j] + dt*du_dt
    

def engine_2DKarmaRappelGPU(sim):
    cuda.synchronize()
    kernel_2DKarmaRappelGPU[sim._gpu_blocks_per_grid_2D, sim._gpu_threads_per_block_2D](sim._fields_gpu_device, sim._fields_out_gpu_device, 
                                                                  sim.user_data['w'], sim.user_data['lambda_val'], sim.user_data['tau'], 
                                                                  sim.user_data['D'], sim.dx, sim.dt, sim.user_data['e4'])
    
    cuda.synchronize()
    




class KarmaRappel1996(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        phi = -1*np.ones(dim)
        u = np.ones(dim)

        self.add_field(phi, 'phi')
        self.add_field(u, 'u')


    def initialize_fields_and_imported_data(self):
        super().initialize_fields_and_imported_data()

    def just_before_simulating(self):
        super().just_before_simulating()
        

    def simulation_loop(self):
        if self._uses_gpu:
            engine_2DKarmaRappelGPU(self)
        else:
            KarmaRappelCPU(self)
        
