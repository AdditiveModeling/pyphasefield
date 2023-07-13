import numpy as np
import sympy as sp
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import math
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE

try:
    from numba import cuda
    import numba
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except:
    import pyphasefield.jit_placeholder as numba
    import pyphasefield.jit_placeholder as cuda

ufunc_g_l = None
ufunc_g_s = None

@cuda.jit(device=True)
def divagradb(a, axp, axm, ayp, aym, b, bxp, bxm, byp, bym, idx): 
    return 0.5*idx*idx*((axp+a)*(bxp-b) - (a+axm)*(b-bxm) + (ayp+a)*(byp-b) - (a+aym)*(b-bym))
    #return (idx*idx*a*(bxp+bxm+byp+bym-4*b) + 0.25*idx*idx*((axp - axm)*(bxp - bxm) + (ayp - aym)*(byp - bym)))

@cuda.jit(device=True)
def f_ori_term(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, mgq_xp, mgq_xm, mgq_yp, mgq_ym, q, q_xp, q_xm, q_yp, q_ym, idx):
    return 0.5*idx*idx*((D_q+D_q_xp)*(q_xp-q)/mgq_xp - (D_q+D_q_xm)*(q-q_xm)/mgq_xm + (D_q+D_q_yp)*(q_yp-q)/mgq_yp - (D_q+D_q_ym)*(q-q_ym)/mgq_ym)

@cuda.jit
def NComponent_kernel(fields, T, transfer, fields_out, rng_states, params, c_params):
    
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    threadId = startx + starty*stridex
    
    dx = params[0]
    d = params[1]
    v_m = params[2]
    M_qmax = params[3]
    H = params[4]
    y_e = params[5]
    beta = params[6]
    dt = params[9]
    ebar = params[10]
    eqbar = params[11]
    noise_amp_phi = params[12]
    noise_amp_q = params[14]
    L = c_params[0]
    T_M = c_params[1]
    S = c_params[2]
    B = c_params[3]
    W = c_params[4]
    M = c_params[5]
    
    phi = fields[0]
    q1 = fields[1]
    q4 = fields[2]
    #c is fields 3 to k, where k equals the number of components +1. 
    #For N components this is N-1 fields, the last one being implicitly defined
    
    phi_out = fields_out[0]
    q1_out = fields_out[1]
    q4_out = fields_out[2]
    
    G_L = transfer[0]
    G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
        
    ebar2 = ebar*ebar
    eqbar2 = eqbar*eqbar
    for j in range(startx+1, phi.shape[1]-1, stridex):
        for i in range(starty+1, phi.shape[0]-1, stridey):
        
            #interpolating functions
            g = (phi[i][j]**2)*(1-phi[i][j])**2
            h = (phi[i][j]**3)*(6.*phi[i][j]**2 - 15.*phi[i][j] + 10.)
            hprime = 30.*g
            gprime = 4.*phi[i][j]**3 - 6.*phi[i][j]**2 + 2*phi[i][j]
            
            #gradients
            idx = 1./dx
            dphidx = 0.5*(phi[i][j+1]-phi[i][j-1])*idx
            dphidx2 = dphidx**2
            dphidx3 = dphidx2*dphidx
            dphidy = 0.5*(phi[i+1][j]-phi[i-1][j])*idx
            dphidy2 = dphidy**2
            dphidy3 = dphidy2*dphidy
            dTdx = 0.5*idx*(T[i][j+1]-T[i][j-1])
            dTdy = 0.5*idx*(T[i+1][j]-T[i-1][j])
            d2phidx2 = (phi[i][j+1]+phi[i][j-1]-2.*phi[i][j])*idx*idx
            d2phidy2 = (phi[i+1][j]+phi[i-1][j]-2.*phi[i][j])*idx*idx
            lphi = d2phidx2 + d2phidy2
            d2phidxy = 0.25*(phi[i+1][j+1]-phi[i+1][j-1]-phi[i-1][j+1]+phi[i-1][j-1])*idx*idx
            mag_grad_phi2 = dphidx**2 + dphidy**2
            if(mag_grad_phi2 < 1e-6):
                mag_grad_phi2 = 1e-6
            mag_grad_phi4 = mag_grad_phi2**2
            mag_grad_phi8 = mag_grad_phi4**2
            
            dq1dx = 0.5*idx*(q1[i][j+1]-q1[i][j-1])
            dq1dy = 0.5*idx*(q1[i+1][j]-q1[i-1][j])
            dq4dx = 0.5*idx*(q4[i][j+1]-q4[i][j-1])
            dq4dy = 0.5*idx*(q4[i+1][j]-q4[i-1][j])
            mgq_xp = idx*math.sqrt((q1[i][j+1]-q1[i][j])**2 + (q4[i][j+1]-q4[i][j])**2)
            mgq_xm = idx*math.sqrt((q1[i][j-1]-q1[i][j])**2 + (q4[i][j-1]-q4[i][j])**2)
            mgq_yp = idx*math.sqrt((q1[i+1][j]-q1[i][j])**2 + (q4[i+1][j]-q4[i][j])**2)
            mgq_ym = idx*math.sqrt((q1[i-1][j]-q1[i][j])**2 + (q4[i-1][j]-q4[i][j])**2)
            mag_grad_q = 0.5*(mgq_xp+mgq_xm+mgq_yp+mgq_ym)
            if(mgq_xp < beta):
                mgq_xp = beta
            if(mgq_xm < beta):
                mgq_xm = beta
            if(mgq_yp < beta):
                mgq_yp = beta
            if(mgq_ym < beta):
                mgq_ym = beta
            
            #psi terms
            q2q2 = (q1[i][j]**2 - q4[i][j]**2)
            qq2 = 2.*q1[i][j]*q4[i][j]
            psix = q2q2*dphidx - qq2*dphidy
            psiy = qq2*dphidx + q2q2*dphidy
            psix2 = psix**2
            psix3 = psix2*psix
            psix4 = psix2**2
            psiy2 = psiy**2
            psiy3 = psiy2*psiy
            psiy4 = psiy2**2
            eta = 1. - 3.*y_e + 4.*y_e*(psix**4 + psiy**4)/mag_grad_phi4
            dq2q2dx = (2.*q1[i][j]*dq1dx - 2.*q4[i][j]*dq4dx)
            dqq2dx = (2.*q4[i][j]*dq1dx + 2.*q1[i][j]*dq4dx)
            dq2q2dy = (2*q1[i][j]*dq1dy - 2.*q4[i][j]*dq4dy)
            dqq2dy = (2.*q4[i][j]*dq1dy + 2.*q1[i][j]*dq4dy)
            dpsixdx = dq2q2dx*dphidx + q2q2*d2phidx2 - dqq2dx*dphidy - qq2*d2phidxy
            dpsixdy = dq2q2dy*dphidx + q2q2*d2phidxy - dqq2dy*dphidy - qq2*d2phidy2
            dpsiydx = dq2q2dx*dphidy + q2q2*d2phidxy + dqq2dx*dphidx + qq2*d2phidx2
            dpsiydy = dq2q2dy*dphidy + q2q2*d2phidy2 + dqq2dy*dphidx + qq2*d2phidxy
            
            d_term_dx = dTdx*((2.*psix3*q2q2 + 2.*psiy3*qq2)/mag_grad_phi2 - dphidx*(psix4+psiy4)/mag_grad_phi4)
            d_term_dx += T[i][j]*(6.*psix2*dpsixdx*q2q2 + 2.*psix3*dq2q2dx + 6.*psiy2*dpsiydx*qq2 + 2.*psiy3*dqq2dx)/mag_grad_phi2
            d_term_dx -= T[i][j]*(2.*psix3*q2q2 + 2.*psiy3*qq2)*(2.*dphidx*d2phidx2 + 2.*dphidy*d2phidxy)/mag_grad_phi4
            d_term_dx -= T[i][j]*(d2phidx2*(psix4+psiy4) + dphidx*(4.*psix3*dpsixdx + 4.*psiy3*dpsiydx))/mag_grad_phi4
            d_term_dx += T[i][j]*(dphidx*(psix4+psiy4)*(4.*dphidx3*d2phidx2 + 4.*dphidx*d2phidx2*dphidy2 + 4.*dphidy*d2phidxy*dphidx2 + 4.*dphidy3*d2phidxy))/mag_grad_phi8
            d_term_dx *= (4.*y_e*ebar2)
            
            d_term_dy = dTdy*((-2.*psix3*qq2 + 2.*psiy3*q2q2)/mag_grad_phi2 - dphidy*(psix4+psiy4)/mag_grad_phi4)
            d_term_dy += T[i][j]*(-6.*psix2*dpsixdy*qq2 - 2.*psix3*dqq2dy + 6.*psiy2*dpsiydy*q2q2 + 2.*psiy3*dq2q2dy)/mag_grad_phi2
            d_term_dy -= T[i][j]*(-2.*psix3*qq2 + 2.*psiy3*q2q2)*(2.*dphidx*d2phidxy + 2.*dphidy*d2phidy2)/mag_grad_phi4
            d_term_dy -= T[i][j]*(d2phidy2*(psix4+psiy4) + dphidy*(4.*psix3*dpsixdy + 4.*psiy3*dpsiydy))/mag_grad_phi4
            d_term_dy += T[i][j]*(dphidy*(psix4+psiy4)*(4.*dphidx3*d2phidxy + 4.*dphidx*d2phidxy*dphidy2 + 4.*dphidy*d2phidy2*dphidx2 + 4.*dphidy3*d2phidy2))/mag_grad_phi8
            d_term_dy *= (4.*y_e*ebar2)
            
            #c_N stuff
            cW = 0.
            c_N = 1.
            M_phi = 0.
            #REACHED HERE
            for l in range(3, len(fields)):
                c_i = fields[l][i][j]
                cW += c_i*W[l-3]
                c_N -= c_i
                M_phi += c_i*M[l-3]
            cW += c_N*W[len(fields)-3]
            M_phi += c_N*M[len(fields)-3]
            
            #mobilities
            M_q = M_qmax + (1e-6-M_qmax)*h
            M_phi *= eta
            
            #dphidt
            dphidt = ebar2*(1.-3.*y_e)*(T[i][j]*lphi + dTdx*dphidx + dTdy*dphidy)
            dphidt += d_term_dx + d_term_dy
            dphidt -= hprime*(G_S[i][j] - G_L[i][j])/v_m
            dphidt -= gprime*T[i][j]*cW
            dphidt -= 4.*H*T[i][j]*phi[i][j]*mag_grad_q
            dphidt *= M_phi
            
            #noise in phi
            noise_phi = math.sqrt(2.*8.314*T[i][j]*M_phi/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            dphidt += noise_phi*noise_amp_phi
            
            #dcidt
            for l in range(3, len(fields)):
                c_i_out = fields_out[l]
                M_c = transfer[l-1]
                dFdci = transfer[l-1+len(fields)-3]
                c_i_out[i][j] = (divagradb(M_c[i][j], M_c[i][j+1], M_c[i][j-1], M_c[i+1][j], M_c[i-1][j], 
                                           dFdci[i][j], dFdci[i][j+1], dFdci[i][j-1], dFdci[i+1][j], dFdci[i-1][j], idx))
                for m in range(3, len(fields)):
                    c_j = fields[m]
                    dFdcj = transfer[m-1+len(fields)-3]
                    c_i_out[i][j] -= divagradb(M_c[i][j]*c_j[i][j], M_c[i][j+1]*c_j[i][j+1], M_c[i][j-1]*c_j[i][j-1], 
                                               M_c[i+1][j]*c_j[i+1][j], M_c[i-1][j]*c_j[i-1][j], 
                                               dFdcj[i][j], dFdcj[i][j+1], dFdcj[i][j-1], dFdcj[i+1][j], dFdcj[i-1][j], idx)
            
            #dqdt
            D_q = 2.*H*T[i][j]*(phi[i][j]**2)
            D_q_xp = 2.*H*T[i][j+1]*(phi[i][j+1]**2)
            D_q_xm = 2.*H*T[i][j-1]*(phi[i][j-1]**2)
            D_q_yp = 2.*H*T[i+1][j]*(phi[i+1][j]**2)
            D_q_ym = 2.*H*T[i-1][j]*(phi[i-1][j]**2)
                
            f_ori_1 = f_ori_term(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, mgq_xp, mgq_xm, mgq_yp, mgq_ym,
                                 q1[i][j], q1[i][j+1], q1[i][j-1], q1[i+1][j], q1[i-1][j], idx)
            f_ori_4 = f_ori_term(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, mgq_xp, mgq_xm, mgq_yp, mgq_ym,
                                 q4[i][j], q4[i][j+1], q4[i][j-1], q4[i+1][j], q4[i-1][j], idx)
            
            #dfintdq1 = 16.*ebar2*T[i][j]*y_e/mag_grad_phi2 * (psix3*(q1[i][j]*dphidx - q4[i][j]*dphidy) + psiy3*(q4[i][j]*dphidx + q1[i][j]*dphidy))
            #dfintdq4 = 16.*ebar2*T[i][j]*y_e/mag_grad_phi2 * (psix3*(-q4[i][j]*dphidx - q1[i][j]*dphidy) + psiy3*(q1[i][j]*dphidx - q4[i][j]*dphidy))
            dfintdq1 = 0. #use these blocks to zero out twisting in quaternion fields to lower interfacial energy
            dfintdq4 = 0.
            
            lq1 = (q1[i][j+1]+q1[i][j-1]+q1[i+1][j]+q1[i-1][j]-4*q1[i][j])*idx*idx
            lq4 = (q4[i][j+1]+q4[i][j-1]+q4[i+1][j]+q4[i-1][j]-4*q4[i][j])*idx*idx
            
            q_noise_coeff = 0.0000000001
            noise_q1 = noise_amp_q*math.sqrt(q_noise_coeff*8.314*T[i][j]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            noise_q4 = noise_amp_q*math.sqrt(q_noise_coeff*8.314*T[i][j]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            #noise_q1 = 0.
            #noise_q4 = 0.
            
            dq1dt = M_q*((1-q1[i][j]**2)*(f_ori_1+lq1*eqbar2-dfintdq1+noise_q1) - q1[i][j]*q4[i][j]*(f_ori_4+lq4*eqbar2-dfintdq4+noise_q4))
            dq4dt = M_q*((1-q4[i][j]**2)*(f_ori_4+lq4*eqbar2-dfintdq4+noise_q4) - q1[i][j]*q4[i][j]*(f_ori_1+lq1*eqbar2-dfintdq1+noise_q1))
            dphi = dt*dphidt
            #dphi = max(-0.01, dt*dphidt)
            #dphi = min(0.01, dphi)
            phi_out[i][j] = phi[i][j]+dphi
            if(phi_out[i][j] < 0.0001):
                phi_out[i][j] = 0.0001
            if(phi_out[i][j] > 0.9999):
                phi_out[i][j] = 0.9999
            q1_out[i][j] = q1[i][j] + dt*dq1dt
            q4_out[i][j] = q4[i][j] + dt*dq4dt
            renorm = math.sqrt((q1_out[i][j]**2+q4_out[i][j]**2))
            q1_out[i][j] = q1_out[i][j]/renorm
            q4_out[i][j] = q4_out[i][j]/renorm
            for l in range(3, len(fields)):
                c_i = fields[l]
                c_i_out = fields_out[l]
                c_i_out[i][j] *= dt
                #c_i_out[i][j] = max(-0.1, c_i_out[i][j])
                #c_i_out[i][j] = min(0.1, c_i_out[i][j])
                c_i_out[i][j] += c_i[i][j]
                #c_i_out[i][j] = max(0, c_i_out[i][j])
                
@numba.jit
def get_thermodynamics(ufunc, array):
    return ufunc(array)

@cuda.jit
def NComponent_noise_kernel(fields, T, transfer, rng_states, ufunc_array, params, c_params):
    startx, starty = cuda.grid(2)     
    stridex, stridey = cuda.gridsize(2) 
    threadId = startx + starty*stridex
    
    v_m = params[2]
    D_L = params[7]
    D_S = params[8]
    noise_amp_c = params[13]
    W = c_params[4]
    
    phi = fields[0]
    q1 = fields[1]
    q4 = fields[2]
    #c is fields 3 to k, where k equals the number of components +1. 
    #For N components this is N-1 fields, the last one being implicitly defined
    
    phi_out = fields[0]
    q1_out = fields[1]
    q4_out = fields[2]
    
    G_L = transfer[0]
    G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
    
    for i in range(starty+1, phi.shape[0]-1, stridey):
        for j in range(startx+1, phi.shape[1]-1, stridex):
            for l in range(3, len(fields)):
                dFdc = transfer[l-1+len(fields)-3]
                noise_c = noise_amp_c*math.sqrt(2.*8.314*T[i][j]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
                dFdc[i][j] += noise_c
                if(i == 1):
                    dFdc[phi.shape[0]-1][j] += noise_c
                if(j == 1):
                    dFdc[i][phi.shape[1]-1] += noise_c
                if(i == phi.shape[0]-2):
                    dFdc[0][j] += noise_c
                if(j == phi.shape[1]-2):
                    dFdc[i][0] += noise_c
            
@cuda.jit
def NComponent_helper_kernel(fields, T, transfer, rng_states, ufunc_array, params, c_params):
    #initializes certain arrays that are used in div-grad terms, to avoid recomputing terms 4 or 6 times
    startx, starty = cuda.grid(2)     
    stridex, stridey = cuda.gridsize(2) 
    threadId = startx + starty*stridex
    
    v_m = params[2]
    D_L = params[7]
    D_S = params[8]
    noise_amp_c = params[13]
    thermo_finite_diff_incr = params[15]
    tfdi_inv = 1./thermo_finite_diff_incr
    W = c_params[4]
    
    phi = fields[0]
    q1 = fields[1]
    q4 = fields[2]
    #c is fields 3 to k, where k equals the number of components +1. 
    #For N components this is N-1 fields, the last one being implicitly defined
    
    phi_out = fields[0]
    q1_out = fields[1]
    q4_out = fields[2]
    
    G_L = transfer[0]
    G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
    
    for i in range(starty, phi.shape[0], stridey):
        for j in range(startx, phi.shape[1], stridex):
            c_N = 1.
            for l in range(3, len(fields)):
                ufunc_array[i][j][l-3] = fields[l][i][j]
                c_N -= fields[l][i][j]
            comps = len(fields)-2
            ufunc_array[i][j][len(fields)-3] = c_N
            ufunc_array[i][j][len(fields)-2] = T[i][j]
            #dGdc = numba.cuda.local.array((2,1), numba.float64)
            #NEEDS FIXING vvvvvvvvvv
            G_L[i][j] = get_thermodynamics(ufunc_g_l, ufunc_array[i][j])
            G_S[i][j] = get_thermodynamics(ufunc_g_s, ufunc_array[i][j])
            
            g = (phi[i][j]**2)*(1-phi[i][j])**2
            h = (phi[i][j]**3)*(6.*phi[i][j]**2 - 15.*phi[i][j] + 10.)
            
            ufunc_array[i][j][len(fields)-3] -= thermo_finite_diff_incr
            for l in range(3, len(fields)):
                ufunc_array[i][j][l-3] += thermo_finite_diff_incr
                dGLdc = tfdi_inv*(get_thermodynamics(ufunc_g_l, ufunc_array[i][j])-G_L[i][j])
                dGSdc = tfdi_inv*(get_thermodynamics(ufunc_g_s, ufunc_array[i][j])-G_S[i][j])
                M_c = transfer[l-1]
                dFdc = transfer[l-1+len(fields)-3]
                M_c[i][j] = v_m*fields[l][i][j]*(D_L + h*(D_S - D_L))/(8.314*T[i][j])
                dFdc[i][j] = (dGLdc + h*(dGSdc-dGLdc))/v_m + (W[l-3]-W[len(fields)-3])*g*T[i][j]
                ufunc_array[i][j][l-3] -= thermo_finite_diff_incr
            ufunc_array[i][j][len(fields)-3] += thermo_finite_diff_incr
                    
                
def make_seed(phi, q1, q4, x, y, angle, seed_radius):
    shape = phi.shape
    qrad = seed_radius+5
    x_size = shape[1]
    y_size = shape[0]
    for i in range((int)(y-seed_radius), (int)(y+seed_radius)):
        for j in range((int)(x-seed_radius), (int)(x+seed_radius)):
            if((i-y)*(i-y)+(j-x)*(j-x) < (seed_radius**2)):
                phi[i%y_size][j%x_size] = 1
    for i in range((int)(y-qrad), (int)(y+qrad)):
        for j in range((int)(x-qrad), (int)(x+qrad)):
            if((i-y)*(i-y)+(j-x)*(j-x) < (qrad**2)):
                #angle is halved because that is how quaternions do
                q1[i%y_size][j%x_size] = np.cos(0.5*angle)
                q4[i%y_size][j%x_size] = np.sin(0.5*angle)
    return phi, q1, q4

def npvalue(var, string, tdb):
    """
    Returns a numpy float from the sympy expression gotten from pycalphad
    Reason: some numpy functions (i.e. sqrt) are incompatible with sympy floats!
    """
    return sp.lambdify(var, tdb.symbols[string], 'numpy')(1000)
    
class NCGPU(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL" #must be this framework for this engine
        self.user_data["d_ratio"] = 4. #default value
        self.user_data["noise_phi"] = 1. #default value
        self.user_data["noise_c"] = 1. #default value
        self.user_data["noise_q"] = 1. #default value
        
    def init_tdb_params(self):
        super().init_tdb_params()
        global ufunc_g_l, ufunc_g_s
        ufunc_g_l = self._tdb_ufuncs[1]
        ufunc_g_s = self._tdb_ufuncs[0] #FCC_A1 comes before LIQUID
        tdb = self._tdb
        comps = self._tdb_components
        try:
            self.user_data["d"] = self.get_cell_spacing()*self.user_data["d_ratio"]
            self.user_data["R"] = 8.314
            self.user_data["L"] = [] #latent heats, J/cm^3
            self.user_data["T_M"] = [] #melting temperatures, K
            self.user_data["S"] = [] #surface energies, J/cm^2
            self.user_data["B"] = [] #linear kinetic coefficients, cm/(K*s)
            self.user_data["W"] = [] #Well size
            self.user_data["M"] = [] #Order mobility coefficient
            T = tdb.symbols[comps[0]+"_L"].free_symbols.pop()
            for i in range(len(comps)):
                self.user_data["L"].append(npvalue(T, comps[i]+"_L", tdb))
                self.user_data["T_M"].append(npvalue(T, comps[i]+"_TM", tdb))
                self.user_data["S"].append(npvalue(T, comps[i]+"_S", tdb))
                self.user_data["B"].append(npvalue(T, comps[i]+"_B", tdb))
                self.user_data["W"].append(3*self.user_data["S"][i]/(np.sqrt(2)*self.user_data["T_M"][i]*self.user_data["d"])) #TODO: figure out specific form of this term in particular
                self.user_data["M"].append((self.user_data["T_M"][i]**2)*self.user_data["B"][i]/(6*np.sqrt(2)*self.user_data["L"][i]*self.user_data["d"])/1574.)
            self.user_data["D_S"] = npvalue(T, "D_S", tdb)
            self.user_data["D_L"] = npvalue(T, "D_L", tdb)
            self.user_data["v_m"] = npvalue(T, "V_M", tdb)
            self.user_data["M_qmax"] = npvalue(T, "M_Q", tdb)
            self.user_data["H"] = npvalue(T, "H", tdb)
            self.user_data["y_e"] = npvalue(T, "Y_E", tdb)
            self.user_data["ebar"] = np.sqrt(6*np.sqrt(2)*self.user_data["S"][1]*self.user_data["d"]/self.user_data["T_M"][1])
            self.user_data["eqbar"] = 0.1*self.user_data["ebar"]
            self.set_time_step_length(self.get_cell_spacing()**2/5./self.user_data["D_L"]/8)
            self.user_data["beta"] = 1.5
        except Exception as e:
            print("Could not load every parameter required from the TDB file!")
            print(e)
            raise Exception()
            
    def init_fields(self):
        self._num_transfer_arrays = 2*len(self._tdb_components)
        self._tdb_ufunc_input_size = len(self._tdb_components)+1
        self.user_data["rng_states"] = create_xoroshiro128p_states(256*256, seed=1)
        #init_xoroshiro128p_states(256*256, seed=3446621627)
        dim = self.dimensions
        q1 = np.zeros(dim)
        q4 = np.zeros(dim)
        try:
            melt_angle = self.user_data["melt_angle"]
        except:
            print("self.user_data[\"melt_angle\"] not defined, defaulting to 0")
            melt_angle = 0.
        #angle is halved because that is how quaternions do
        q1 += np.cos(0.5*melt_angle)
        q4 += np.sin(0.5*melt_angle)
        
        try:
            sim_type = self.user_data["sim_type"]
            if(sim_type == "seed"):
                #initialize phi, q1, q4
                
                phi = np.zeros(dim)
                try:
                    seed_angle = self.user_data["seed_angle"]
                except:
                    print("self.user_data[\"seed_angle\"] not defined, defaulting to pi/4")
                    seed_angle = 1*np.pi/4
                phi, q1, q4 = make_seed(phi, q1, q4, dim[1]/2, dim[0]/2, seed_angle, 5)
                self.add_field(phi, "phi", colormap=COLORMAP_PHASE)
                self.add_field(q1, "q1")
                self.add_field(q4, "q4")
                #initialize concentration array(s)
                try:
                    initial_concentration_array = self.user_data["initial_concentration_array"]
                    assert((len(initial_concentration_array)+1) == len(self._tdb_components))
                    for i in range(len(initial_concentration_array)):
                        c_n = np.zeros(dim)
                        c_n += initial_concentration_array[i]
                        self.add_field(c_n, "c_"+self._tdb_components[i], colormap=COLORMAP_OTHER)
                except: #initial_concentration array isnt defined?
                    for i in range(len(self._tdb_components)-1):
                        c_n = np.zeros(dim)
                        c_n += 1./len(self._tdb_components)
                        self.add_field(c_n, "c_"+self._tdb_components[i], colormap=COLORMAP_OTHER)
            elif(sim_type == "seeds"):
                #initialize phi, q1, q4
                phi = np.zeros(dim)
                try:
                    number_of_seeds = self.user_data["number_of_seeds"]
                except:
                    print("self.user_data[\"number_of_seeds\"] not defined, defaulting to about 1 seed per 10000 cells")
                    number_of_seeds = np.prod(dim)//10000
                for j in range(number_of_seeds):
                    seed_angle = (np.random.rand()-0.5)*np.pi/2 + melt_angle
                    x_pos = int(np.random.rand()*dim[1])
                    y_pos = int(np.random.rand()*dim[0])
                    phi, q1, q4 = make_seed(phi, q1, q4, x_pos, y_pos, seed_angle, 5)

                self.add_field(phi, "phi", colormap=COLORMAP_PHASE)
                self.add_field(q1, "q1")
                self.add_field(q4, "q4")

                #initialize concentration array(s)
                try:
                    initial_concentration_array = self.user_data["initial_concentration_array"]
                    assert((len(initial_concentration_array)+1) == len(self._tdb_components))
                    for i in range(len(initial_concentration_array)):
                        c_n = np.zeros(dim)
                        c_n += initial_concentration_array[i]
                        self.add_field(c_n, "c_"+self._tdb_components[i], colormap=COLORMAP_OTHER)
                except: #initial_concentration array isnt defined?
                    for i in range(len(self._tdb_components)-1):
                        c_n = np.zeros(dim)
                        c_n += 1./len(self._tdb_components)
                        self.add_field(c_n, "c_"+self._tdb_components[i], colormap=COLORMAP_OTHER)
        
        except:
            phi = np.zeros(dim)
            self.add_field(phi, "phi", colormap=COLORMAP_PHASE)
            self.add_field(q1, "q1")
            self.add_field(q4, "q4")
            #initialize concentration array(s)
            try:
                initial_concentration_array = self.user_data["initial_concentration_array"]
                assert((len(initial_concentration_array)+1) == len(self._tdb_components))
                for i in range(len(initial_concentration_array)):
                    c_n = np.zeros(dim)
                    c_n += initial_concentration_array[i]
                    self.add_field(c_n, "c_"+self._tdb_components[i], colormap=COLORMAP_OTHER)
            except: #initial_concentration array isnt defined?
                for i in range(len(self._tdb_components)-1):
                    c_n = np.zeros(dim)
                    c_n += 1./len(self._tdb_components)
                    self.add_field(c_n, "c_"+self._tdb_components[i], colormap=COLORMAP_OTHER)
                        
    def just_before_simulating(self):
        super().just_before_simulating()
        if not "d_ratio" in self.user_data:
            self.user_data["d_ratio"] = 4.
        if not "noise_phi" in self.user_data:
            self.user_data["noise_phi"] = 1.
        if not "noise_c" in self.user_data:
            self.user_data["noise_c"] = 1.
        if not "noise_q" in self.user_data:
            self.user_data["noise_q"] = 1.
        #initialize finite diff approximation for thermodynamic gradient terms to 1e-7, approx. half the digits of precision of doubles
        if not "thermo_finite_diff_incr" in self.user_data:
            self.user_data["thermo_finite_diff_incr"] = 0.0000001;
        params = []
        c_params = []
        params.append(self.dx)
        params.append(self.user_data["d"])
        params.append(self.user_data["v_m"])
        params.append(self.user_data["M_qmax"])
        params.append(self.user_data["H"])
        params.append(self.user_data["y_e"])
        params.append(self.user_data["beta"])
        params.append(self.user_data["D_L"])
        params.append(self.user_data["D_S"])
        params.append(self.dt)
        params.append(self.user_data["ebar"])
        params.append(self.user_data["eqbar"])
        params.append(self.user_data["noise_phi"])
        params.append(self.user_data["noise_c"])
        params.append(self.user_data["noise_q"])
        params.append(self.user_data["thermo_finite_diff_incr"])
        c_params.append(self.user_data["L"])
        c_params.append(self.user_data["T_M"])
        c_params.append(self.user_data["S"])
        c_params.append(self.user_data["B"])
        c_params.append(self.user_data["W"])
        c_params.append(self.user_data["M"])
        self.user_data["params"] = np.array(params)
        self.user_data["c_params"] = np.array(c_params)
        
    def simulation_loop(self):
        cuda.synchronize()
        if(len(self.dimensions) == 1):
            NComponent_helper_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            NComponent_noise_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            NComponent_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
        elif(len(self.dimensions) == 2):
            NComponent_helper_kernel[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            NComponent_noise_kernel[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            NComponent_kernel[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
        elif(len(self.dimensions) == 3):
            NComponent_helper_kernel[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            NComponent_noise_kernel[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            NComponent_kernel[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
        cuda.synchronize()