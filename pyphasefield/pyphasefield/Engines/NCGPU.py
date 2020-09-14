import numpy as np
import sympy as sp
from scipy.sparse.linalg import gmres
from ..field import Field
from ..ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
from numba import cuda
import numba
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

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
    threadId = cuda.grid(1)
    
    dx = params[0]
    d = params[1]
    v_m = params[2]
    M_qmax = params[3]
    H = params[4]
    y_e = params[5]
    beta = params[6]
    dt = params[9]
    L = c_params[0]
    T_M = c_params[1]
    S = c_params[2]
    B = c_params[3]
    W = c_params[4]
    M = c_params[5]
    
    #phi = fields[0]
    #q1 = fields[1]
    #q4 = fields[2]
    
    #G_L = transfer[0]
    #G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
        
    ebar2 = 6.*math.sqrt(2.)*S[1]*d/T_M[1]
    eqbar2 = 0.25*ebar2
    
    for i in range(starty, fields[0].shape[0], stridey):
        for j in range(startx, fields[0].shape[1], stridex):
            #interpolating functions
            g = (fields[0][i][j]**2)*(1-fields[0][i][j])**2
            h = (fields[0][i][j]**3)*(6.*fields[0][i][j]**2 - 15.*fields[0][i][j] + 10.)
            hprime = 30.*g
            gprime = 4.*fields[0][i][j]**3 - 6.*fields[0][i][j]**2 + 2*fields[0][i][j]
            
            #gradients
            idx = 1./dx
            dphidx = 0.5*(fields[0][i][j+1]-fields[0][i][j-1])*idx
            dphidx2 = dphidx**2
            dphidx3 = dphidx2*dphidx
            dphidy = 0.5*(fields[0][i+1][j]-fields[0][i-1][j])*idx
            dphidy2 = dphidy**2
            dphidy3 = dphidy2*dphidy
            dTdx = 0.5*idx*(T[i][j+1]-T[i][j-1])
            dTdy = 0.5*idx*(T[i+1][j]-T[i-1][j])
            d2phidx2 = (fields[0][i][j+1]+fields[0][i][j-1]-2*fields[0][i][j])*idx*idx
            d2phidy2 = (fields[0][i+1][j]+fields[0][i-1][j]-2*fields[0][i][j])*idx*idx
            lphi = d2phidx2 + d2phidy2
            d2phidxy = 0.25*(fields[0][i+1][j+1]-fields[0][i+1][j-1]-fields[0][i-1][j+1]+fields[0][i-1][j-1])*idx*idx
            mag_grad_phi2 = dphidx**2 + dphidy**2
            if(mag_grad_phi2 < 1e-6):
                mag_grad_phi2 = 1e-6
            mag_grad_phi4 = mag_grad_phi2**2
            mag_grad_phi8 = mag_grad_phi4**2
            
            dq1dx = 0.5*idx*(fields[1][i][j+1]-fields[1][i][j-1])
            dq1dy = 0.5*idx*(fields[1][i+1][j]-fields[1][i-1][j])
            dq4dx = 0.5*idx*(fields[2][i][j+1]-fields[2][i][j-1])
            dq4dy = 0.5*idx*(fields[2][i+1][j]-fields[2][i-1][j])
            mgq_xp = idx*math.sqrt((fields[1][i][j+1]-fields[1][i][j])**2 + (fields[2][i][j+1]-fields[2][i][j])**2)
            mgq_xm = idx*math.sqrt((fields[1][i][j-1]-fields[1][i][j])**2 + (fields[2][i][j-1]-fields[2][i][j])**2)
            mgq_yp = idx*math.sqrt((fields[1][i+1][j]-fields[1][i][j])**2 + (fields[2][i+1][j]-fields[2][i][j])**2)
            mgq_ym = idx*math.sqrt((fields[1][i-1][j]-fields[1][i][j])**2 + (fields[2][i-1][j]-fields[2][i][j])**2)
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
            q2q2 = (fields[1][i][j]**2 - fields[2][i][j]**2)
            qq2 = 2*fields[1][i][j]*fields[2][i][j]
            psix = q2q2*dphidx - qq2*dphidy
            psiy = qq2*dphidx + q2q2*dphidy
            psix2 = psix**2
            psix3 = psix2*psix
            psix4 = psix2**2
            psiy2 = psiy**2
            psiy3 = psiy2*psiy
            psiy4 = psiy2**2
            #eta = 1. - 3.*y_e + 4.*y_e*(psix**4 + psiy**4)/mag_grad_phi4
            dq2q2dx = (2*fields[1][i][j]*dq1dx - 2*fields[2][i][j]*dq4dx)
            dqq2dx = (2*fields[2][i][j]*dq1dx + 2*fields[1][i][j]*dq4dx)
            dq2q2dy = (2*fields[1][i][j]*dq1dy - 2*fields[2][i][j]*dq4dy)
            dqq2dy = (2*fields[2][i][j]*dq1dy + 2*fields[1][i][j]*dq4dy)
            dpsixdx = dq2q2dx*dphidx + q2q2*d2phidx2 - dqq2dx*dphidy - qq2*d2phidxy
            dpsixdy = dq2q2dy*dphidx + q2q2*d2phidxy - dqq2dy*dphidy - qq2*d2phidy2
            dpsiydx = dq2q2dx*dphidy + q2q2*d2phidxy + dqq2dx*dphidx + qq2*d2phidx2
            dpsiydy = dq2q2dy*dphidy + q2q2*d2phidy2 + dqq2dy*dphidx + qq2*d2phidxy
            
            d_term_dx = dTdx*((2*psix3*q2q2 + 2*psiy3*qq2)/mag_grad_phi2 - dphidx*(psix4+psiy4)/mag_grad_phi4)
            d_term_dx += T[i][j]*(6*psix2*dpsixdx*q2q2 + 2*psix3*dq2q2dx + 6*psiy2*dpsiydx*qq2 + 2*psiy3*dqq2dx)/mag_grad_phi2
            d_term_dx -= T[i][j]*(2*psix3*q2q2 + 2*psiy3*qq2)*(2*dphidx*d2phidx2 + 2*dphidy*d2phidxy)/mag_grad_phi4
            d_term_dx -= T[i][j]*(d2phidx2*(psix4+psiy4) + dphidx*(4*psix3*dpsixdx + 4*psiy3*dpsiydx))/mag_grad_phi4
            d_term_dx += T[i][j]*(dphidx*(psix4+psiy4)*(4*dphidx3*d2phidx2 + 4*dphidx*d2phidx2*dphidy2 + 4*dphidy*d2phidxy*dphidx2 + 4*dphidy3*d2phidxy))/mag_grad_phi8
            d_term_dx *= (4*y_e*ebar2)
            
            d_term_dy = dTdy*((-2*psix3*qq2 + 2*psiy3*q2q2)/mag_grad_phi2 - dphidy*(psix4+psiy4)/mag_grad_phi4)
            d_term_dy += T[i][j]*(-6*psix2*dpsixdy*qq2 - 2*psix3*dqq2dy + 6*psiy2*dpsiydy*q2q2 + 2*psiy3*dq2q2dy)/mag_grad_phi2
            d_term_dy -= T[i][j]*(-2*psix3*qq2 + 2*psiy3*q2q2)*(2*dphidx*d2phidxy + 2*dphidy*d2phidy2)/mag_grad_phi4
            d_term_dy -= T[i][j]*(d2phidy2*(psix4+psiy4) + dphidy*(4*psix3*dpsixdy + 4*psiy3*dpsiydy))/mag_grad_phi4
            d_term_dy += T[i][j]*(dphidy*(psix4+psiy4)*(4*dphidx3*d2phidxy + 4*dphidx*d2phidxy*dphidy2 + 4*dphidy*d2phidy2*dphidx2 + 4*dphidy3*d2phidy2))/mag_grad_phi8
            d_term_dy *= (4*y_e*ebar2)
            
            #c_N stuff
            cW = 0.
            c_N = 1.
            M_phi = 0.
            for l in range(3, len(fields)):
                cW += fields[l][i][j]*W[l-3]
                c_N -= fields[l][i][j]
                M_phi += fields[l][i][j]*M[l-3]
            cW += c_N*W[len(fields)-3]
            M_phi += c_N*M[len(fields)-3]
            
            #mobilities
            M_q = M_qmax + (1e-6-M_qmax)*h
            #M_phi *= eta
            
            #dphidt
            dphidt = ebar2*(1-3*y_e)*(T[i][j]*lphi + dTdx*dphidx + dTdy*dphidy)
            dphidt += d_term_dx + d_term_dy
            dphidt -= hprime*(transfer[1][i][j] - transfer[0][i][j])/v_m
            dphidt -= gprime*T[i][j]*cW
            dphidt -= 4*H*T[i][j]*fields[0][i][j]*mag_grad_q
            dphidt *= M_phi
            
            #noise in phi
            noise_phi = math.sqrt(2.*8.314*T[i][j]*M_phi/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            dphidt += noise_phi
            
            #dcidt
            for l in range(3, len(fields)):
                fields_out[l][i][j] = (divagradb(transfer[l-1][i][j], transfer[l-1][i][j+1], transfer[l-1][i][j-1], transfer[l-1][i+1][j], 
                                                 transfer[l-1][i-1][j], transfer[l-1+len(fields)-3][i][j], 
                                                 transfer[l-1+len(fields)-3][i][j+1], transfer[l-1+len(fields)-3][i][j-1], 
                                                 transfer[l-1+len(fields)-3][i+1][j], transfer[l-1+len(fields)-3][i-1][j], idx))
                for m in range(3, len(fields)):
                    fields_out[l][i][j] -= divagradb(transfer[l-1][i][j]*fields[m][i][j], transfer[l-1][i][j+1]*fields[m][i][j+1], 
                                                transfer[l-1][i][j-1]*fields[m][i][j-1], transfer[l-1][i+1][j]*fields[m][i+1][j], 
                                                transfer[l-1][i-1][j]*fields[m][i-1][j], transfer[m-1+len(fields)-3][i][j], 
                                                transfer[m-1+len(fields)-3][i][j+1], transfer[m-1+len(fields)-3][i][j-1], 
                                                transfer[m-1+len(fields)-3][i+1][j], transfer[m-1+len(fields)-3][i-1][j], idx)
            
            #dqdt
            D_q = 2.*H*T[i][j]*(fields[0][i][j]**2)
            D_q_xp = 2.*H*T[i][j+1]*(fields[0][i][j+1]**2)
            D_q_xm = 2.*H*T[i][j-1]*(fields[0][i][j-1]**2)
            D_q_yp = 2.*H*T[i+1][j]*(fields[0][i+1][j]**2)
            D_q_ym = 2.*H*T[i-1][j]*(fields[0][i-1][j]**2)
                
            f_ori_1 = f_ori_term(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, mgq_xp, mgq_xm, mgq_yp, mgq_ym,
                                 fields[1][i][j], fields[1][i][j+1], fields[1][i][j-1], fields[1][i+1][j], fields[1][i-1][j], idx)
            f_ori_4 = f_ori_term(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, mgq_xp, mgq_xm, mgq_yp, mgq_ym,
                                 fields[2][i][j], fields[2][i][j+1], fields[2][i][j-1], fields[2][i+1][j], fields[2][i-1][j], idx)
            
            dfintdq1 = 16.*ebar2*T[i][j]*y_e/mag_grad_phi2 * (psix3*(fields[1][i][j]*dphidx - fields[2][i][j]*dphidy) + psiy3*(fields[2][i][j]*dphidx + fields[1][i][j]*dphidy))
            dfintdq4 = 16.*ebar2*T[i][j]*y_e/mag_grad_phi2 * (psix3*(-fields[2][i][j]*dphidx - fields[1][i][j]*dphidy) + psiy3*(fields[1][i][j]*dphidx - fields[2][i][j]*dphidy))
            #dfintdq1 = 0. #use these blocks to zero out twisting in quaternion fields to lower interfacial energy
            #dfintdq4 = 0.
            
            lq1 = (fields[1][i][j+1]+fields[1][i][j-1]+fields[1][i+1][j]+fields[1][i-1][j]-4*fields[1][i][j])*idx*idx
            lq4 = (fields[2][i][j+1]+fields[2][i][j-1]+fields[2][i+1][j]+fields[2][i-1][j]-4*fields[2][i][j])*idx*idx
            
            #noise_q1 = math.sqrt(8.314*T[i][j]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            #noise_q4 = math.sqrt(8.314*T[i][j]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            noise_q1 = 0
            noise_q4 = 0
            
            dq1dt = M_q*((1-fields[1][i][j]**2)*(f_ori_1+lq1*eqbar2-dfintdq1+noise_q1) - fields[1][i][j]*fields[2][i][j]*(f_ori_4+lq4*eqbar2-dfintdq4+noise_q4))
            dq4dt = M_q*((1-fields[2][i][j]**2)*(f_ori_4+lq4*eqbar2-dfintdq4+noise_q4) - fields[1][i][j]*fields[2][i][j]*(f_ori_1+lq1*eqbar2-dfintdq1+noise_q1))
            fields_out[0][i][j] = fields[0][i][j] + dt*dphidt
            if(fields_out[0][i][j] < 0.000001):
                fields_out[0][i][j] = 0.000001
            if(fields_out[0][i][j] > 0.999999):
                fields_out[0][i][j] = 0.999999
            fields_out[1][i][j] = fields[1][i][j] + dt*dq1dt
            fields_out[2][i][j] = fields[2][i][j] + dt*dq4dt
            renorm = math.sqrt((fields_out[1][i][j]**2+fields_out[2][i][j]**2))
            fields_out[1][i][j] = fields_out[1][i][j]/renorm
            fields_out[2][i][j] = fields_out[2][i][j]/renorm
            for l in range(3, len(fields)):
                fields_out[l][i][j] *= dt
                fields_out[l][i][j] += fields[l][i][j]

@numba.jit
def get_thermodynamics(ufunc, array):
    if(len(array) == 3):
        G = ufunc(array[0], array[1], array[2])
        dGdc = 10000000.*(ufunc(array[0]+0.0000001, array[1]-0.0000001, array[2])-G)
    return G, dGdc
            
@cuda.jit
def NComponent_helper_kernel(fields, T, transfer, rng_states, ufunc_array, params, c_params):
    #initializes certain arrays that are used in div-grad terms, to avoid recomputing terms 4 or 6 times
    startx, starty = cuda.grid(2)     
    stridex, stridey = cuda.gridsize(2) 
    threadId = cuda.grid(1)
    
    v_m = params[2]
    D_L = params[7]
    D_S = params[8]
    W = c_params[4]
    
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
    
    for i in range(startx, fields[0].shape[0], stridex):
        for j in range(starty, fields[0].shape[1], stridey):
            c_N = 1.
            for l in range(3, len(fields)):
                ufunc_array[i][j][l-3] = fields[l][i][j]
                c_N -= fields[l][i][j]
            ufunc_array[i][j][len(fields)-3] = c_N
            ufunc_array[i][j][len(fields)-2] = T[i][j]
            #dGdc = numba.cuda.local.array((2,1), numba.float64)
            transfer[0][i][j], dGLdc = get_thermodynamics(ufunc_g_l, ufunc_array[i][j])
            transfer[1][i][j], dGSdc = get_thermodynamics(ufunc_g_s, ufunc_array[i][j])
            
            g = (fields[0][i][j]**2)*(1-fields[0][i][j])**2
            h = (fields[0][i][j]**3)*(6.*fields[0][i][j]**2 - 15.*fields[0][i][j] + 10.)
            
            #get Langevin noise, put in c_noise
            noise_c = math.sqrt(2.*8.314*T[i][j]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            #noise_c = 0.
            
            for l in range(3, len(fields)):
                transfer[l-1][i][j] = v_m*fields[l][i][j]*(D_L + h*(D_S - D_L))/(8.314*T[i][j])
                transfer[l-1+len(fields)-3][i][j] = (dGLdc + h*(dGSdc-dGLdc))/v_m + (W[l-3]-W[len(fields)-3])*g*T[i][j]+noise_c
                
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
                q1[i%y_size][j%x_size] = np.cos(angle)
                q4[i%y_size][j%x_size] = np.sin(angle)
    return phi, q1, q4

def npvalue(var, string, tdb):
    """
    Returns a numpy float from the sympy expression gotten from pycalphad
    Reason: some numpy functions (i.e. sqrt) are incompatible with sympy floats!
    """
    return sp.lambdify(var, tdb.symbols[string], 'numpy')(1000)

def init_tdb_parameters(sim):
    """
    Modifies the global vars which are parameters for the engine. Called from the function utils.preinitialize
    Returns True if variables are loaded successfully, False if certain variables dont exist in the TDB
    If false, preinitialize will print an error saying the TDB doesn't have enough info to run the sim
    """
    global ufunc_g_s, ufunc_g_l
    import pycalphad as pyc
    from tinydb import where
    tdb = sim._tdb
    comps = sim._components
    try:
        sim.R = 8.314
        sim.L = [] #latent heats, J/cm^3
        sim.T_M = [] #melting temperatures, K
        sim.S = [] #surface energies, J/cm^2
        sim.B = [] #linear kinetic coefficients, cm/(K*s)
        sim.W = [] #Well size
        sim.M = [] #Order mobility coefficient
        T = tdb.symbols[comps[0]+"_L"].free_symbols.pop()
        for i in range(len(comps)):
            sim.L.append(npvalue(T, comps[i]+"_L", tdb))
            sim.T_M.append(npvalue(T, comps[i]+"_TM", tdb))
            sim.S.append(npvalue(T, comps[i]+"_S", tdb))
            sim.B.append(npvalue(T, comps[i]+"_B", tdb))
            sim.W.append(3*sim.S[i]/(np.sqrt(2)*sim.T_M[i]*sim.d)) #TODO: figure out specific form of this term in particular
            sim.M.append(sim.T_M[i]*sim.T_M[i]*sim.B[i]/(6*np.sqrt(2)*sim.L[i]*sim.d)/1574.)
        sim.D_S = npvalue(T, "D_S", tdb)
        sim.D_L = npvalue(T, "D_L", tdb)
        sim.v_m = npvalue(T, "V_M", tdb)
        sim.M_qmax = npvalue(T, "M_Q", tdb)
        sim.H = npvalue(T, "H", tdb)
        sim.y_e = npvalue(T, "Y_E", tdb)
        sim.ebar = np.sqrt(6*np.sqrt(2)*sim.S[1]*sim.d/sim.T_M[1])
        sim.eqbar = 0.5*sim.ebar
        sim.set_time_step_length(sim.get_cell_spacing()**2/5./sim.D_L/8)
        sim.beta = 1.5
        
        #initialize thermodynamic ufuncs
        
        param_search = sim._tdb.search
        phase_solid = sim._tdb.phases["FCC_A1"]
        g_param_query_solid = (
            (where('phase_name') == phase_solid.name) & \
            ((where('parameter_type') == 'G') | \
            (where('parameter_type') == 'L'))
        )

        model_solid = pyc.Model(sim._tdb, list(sim._tdb.elements), "FCC_A1")
        sympyexpr_solid = model_solid.redlich_kister_sum(phase_solid, param_search, g_param_query_solid)
        ime_solid = model_solid.ideal_mixing_energy(sim._tdb)

        sympysyms_list_solid = []
        T = None
        for i in list(sim._tdb.elements):
            for j in sympyexpr_solid.free_symbols:
                if j.name == "FCC_A10"+i:
                    sympysyms_list_solid.append(j)
                if j.name == "T":
                    T = j
        sympysyms_list_solid = sorted(sympysyms_list_solid, key=lambda t:t.name)
        sympysyms_list_solid.append(T)

        phase_liquid = sim._tdb.phases["LIQUID"]
        g_param_query_liquid = (
            (where('phase_name') == phase_liquid.name) & \
            ((where('parameter_type') == 'G') | \
            (where('parameter_type') == 'L'))
        )

        model_liquid = pyc.Model(sim._tdb, list(sim._tdb.elements), "LIQUID")
        sympyexpr_liquid = model_liquid.redlich_kister_sum(phase_liquid, param_search, g_param_query_liquid)
        ime_liquid = model_liquid.ideal_mixing_energy(sim._tdb) #these are effectively the same but use different sympy vars

        sympysyms_list_liquid = []
        T = None
        for i in list(sim._tdb.elements):
            for j in sympyexpr_liquid.free_symbols:
                if j.name == "LIQUID0"+i:
                    sympysyms_list_liquid.append(j)
                if j.name == "T":
                    T = j
        sympysyms_list_liquid = sorted(sympysyms_list_liquid, key=lambda t:t.name)
        sympysyms_list_liquid.append(T)
        
        for i in sim._tdb.symbols:
            d = sim._tdb.symbols[i]
            g = sp.Symbol(i)
            sympyexpr_solid = sympyexpr_solid.subs(g, d)
            sympyexpr_liquid = sympyexpr_liquid.subs(g, d)

        #print(sympyexpr_solid+ime_solid)
                    
        ufunc_g_s = numba.jit(sp.lambdify(tuple(sympysyms_list_solid), sympyexpr_solid+ime_solid, 'math'))
        ufunc_g_l = numba.jit(sp.lambdify(tuple(sympysyms_list_liquid), sympyexpr_liquid+ime_liquid, 'math'))
            
        return True
    except Exception as e:
        print("Could not load every parameter required from the TDB file!")
        print(e)
        return False
    
def engine_NCGPU_Explicit(sim):
    
    cuda.synchronize()
    NComponent_helper_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device, sim.temperature_gpu_device,
                                                                          sim.transfer_gpu_device, sim.rng_states, 
                                                                          sim.ufunc_array, sim.params, sim.c_params)
    cuda.synchronize()
    NComponent_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device, sim.temperature_gpu_device, 
                                                                   sim.transfer_gpu_device, sim.fields_out_gpu_device,
                                                                   sim.rng_states, sim.params, sim.c_params)
    cuda.synchronize()
    sim.fields_gpu_device, sim.fields_out_gpu_device = sim.fields_out_gpu_device, sim.fields_gpu_device
    
    
                
def init_NCGPU(sim, dim=[200,200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb", temperature_type="isothermal", 
                           initial_temperature=1574, temperature_gradient=0, cooling_rate=0, temperature_file_path="T.xdmf", 
                           initial_concentration_array=[0.40831], cell_spacing=0.0000046, d_ratio=1/0.94, solver="explicit", 
                           nbc=["periodic", "periodic"], cuda_blocks=(16,16), cuda_threads_per_block=(256,1)):
    sim.uses_gpu = True
    sim.cuda_blocks = cuda_blocks
    sim.cuda_threads_per_block = cuda_threads_per_block
    sim.set_boundary_conditions(nbc)
    if(len(dim) == 1):
        dim.append(1)
    for i in range(len(dim)):
        dim[i] += 2
    sim.set_dimensions(dim)
    sim.load_tdb(tdb_path)
    sim.set_cell_spacing(cell_spacing)
    sim.d = sim.get_cell_spacing()*d_ratio
    if(solver == "explicit"):
        sim.set_engine(engine_NCGPU_Explicit)
    init_tdb_parameters(sim)
    if(temperature_type=="isothermal"):
        sim.set_temperature_isothermal(initial_temperature)
    elif(temperature_type=="gradient"):
        sim.set_temperature_gradient(initial_temperature, temperature_gradient, cooling_rate)
    elif(temperature_type=="file"):
        sim.set_temperature_file(temperature_file_path)
    else:
        print("Temperature type of "+temperature_type+" is not recognized, defaulting to isothermal with a temperature of "+str(initial_temperature))
        sim.set_temperature_isothermal(initial_temperature)
    if(sim_type=="seed"):
        #initialize phi, q1, q4
        phi = np.zeros(dim)
        q1 = np.zeros(dim)
        q4 = np.zeros(dim)
        initial_angle = 0*np.pi/8
        q1 += np.cos(initial_angle)
        q4 += np.sin(initial_angle)
        seed_angle = 1*np.pi/8
        phi, q1, q4 = make_seed(phi, q1, q4, dim[1]/2, dim[0]/2, seed_angle, 5)
        phi_field = Field(data=phi, name="phi", simulation=sim, colormap=COLORMAP_PHASE)
        q1_field = Field(data=q1, name="q1", simulation=sim)
        q4_field = Field(data=q4, name="q4", simulation=sim)
        sim.add_field(phi_field)
        sim.add_field(q1_field)
        sim.add_field(q4_field)
        
        #initialize concentration array(s)
        if(initial_concentration_array == None):
            for i in range(len(sim._components)-1):
                c_n = np.zeros(dim)
                c_n += 1./len(sim._components)
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim, colormap=COLORMAP_OTHER)
                sim.add_field(c_n_field)
        else:
            assert((len(initial_concentration_array)+1) == len(sim._components))
            for i in range(len(initial_concentration_array)):
                c_n = np.zeros(dim)
                c_n += initial_concentration_array[i]
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim, colormap=COLORMAP_OTHER)
                sim.add_field(c_n_field)
    elif(sim_type=="seeds"):
        #initialize phi, q1, q4
        phi = np.zeros(dim)
        q1 = np.zeros(dim)
        q4 = np.zeros(dim)
        initial_angle = 0*np.pi/8
        q1 += np.cos(initial_angle)
        q4 += np.sin(initial_angle)
        
        for j in range(number_of_seeds):
            seed_angle = (np.random.rand()-0.5)*np.pi/4
            x_pos = int(np.random.rand()*dim[1])
            y_pos = int(np.random.rand()*dim[0])
            phi, q1, q4 = make_seed(phi, q1, q4, x_pos, y_pos, seed_angle, 5)
        
        phi_field = Field(data=phi, name="phi", simulation=sim, colormap=COLORMAP_PHASE)
        q1_field = Field(data=q1, name="q1", simulation=sim)
        q4_field = Field(data=q4, name="q4", simulation=sim) 
        sim.add_field(phi_field)
        sim.add_field(q1_field)
        sim.add_field(q4_field)
        
        #initialize concentration array(s)
        if(initial_concentration_array == None):
            for i in range(len(sim._components)-1):
                c_n = np.zeros(dim)
                c_n += 1./len(sim._components)
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim, colormap=COLORMAP_OTHER)
                sim.add_field(c_n_field)
        else:
            assert((len(initial_concentration_array)+1) == len(sim._components))
            for i in range(len(initial_concentration_array)):
                c_n = np.zeros(dim)
                c_n += initial_concentration_array[i]
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim, colormap=COLORMAP_OTHER)
                sim.add_field(c_n_field)
    params = []
    c_params = []
    params.append(sim.get_cell_spacing())
    params.append(sim.d)
    params.append(sim.v_m)
    params.append(sim.M_qmax)
    params.append(sim.H)
    params.append(sim.y_e)
    params.append(sim.beta)
    params.append(sim.D_L)
    params.append(sim.D_S)
    params.append(sim.get_time_step_length())
    c_params.append(sim.L)
    c_params.append(sim.T_M)
    c_params.append(sim.S)
    c_params.append(sim.B)
    c_params.append(sim.W)
    c_params.append(sim.M)
    sim.params = np.array(params)
    sim.c_params = np.array(c_params)
    sim.rng_states = create_xoroshiro128p_states(256*256, seed=3)
    out_dim = dim.copy()
    out_dim.insert(0, len(sim._components)+2)
    sim.fields_out_gpu_device = cuda.device_array(out_dim)
    transfer_dim = dim.copy()
    transfer_dim.insert(0, 2*len(sim._components))
    sim.transfer_gpu_device = cuda.device_array(transfer_dim)
    ufunc_array_dim = dim.copy()
    ufunc_array_dim.append(len(sim._components)+1)
    sim.ufunc_array = cuda.device_array(ufunc_array_dim)