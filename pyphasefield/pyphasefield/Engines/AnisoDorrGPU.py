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

@cuda.jit(device=True)
def _h(phi):
    return phi*phi*phi*(10-15*phi+6*phi*phi)

@cuda.jit(device=True)
def _hprime(phi):
    return (30*phi*phi*(1-phi)*(1-phi))

@cuda.jit(device=True)
def _g(phi):
    return (phi*phi*(1-phi)*(1-phi))

@cuda.jit(device=True)
def _gprime(phi):
    return (4*phi*phi*phi - 6*phi*phi +2*phi)

@cuda.jit(device=True)
def grad2(a, axp, axm, ayp, aym, idx):
    return (axp+axm+ayp+aym-4*a)*(idx*idx)

@cuda.jit
def AnisoDorr_kernel(fields, T, transfer, fields_out, rng_states, params, c_params):
    
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    threadId = cuda.grid(1)
    
    #dx = params[0]
    #d = params[1]
    #v_m = params[2]
    #M_qmax = params[3]
    #H = params[4]
    #y_e = params[5]
    #beta = params[6]
    #dt = params[9]
    #L = c_params[0]
    #T_M = c_params[1]
    #S = c_params[2]
    #B = c_params[3]
    #W = c_params[4]
    #M = c_params[5]
    
    #temperature
    T = 1574.

    #bcc = 0L = e, fcc = 1S = d
    #material parameters, J, cm, K, s (except for R and Q terms, which use joules)
    M_qmax = 80000000. #maximum mobility of orientation, 1/(s*J)
    H = 1e-11 #interfacial energy term for quaternions, J/(K*cm)

    #material parameters, from Warren1995
    T_mA = 1728. #melting point of nickel
    T_mB = 1358. #melting point of copper
    L_A = 2350. #latent heat of nickel, J/cm^3
    L_B = 1728. #latent heat of copper, J/cm^3
    s_A = 0.000037 #surface energy of nickel, J/cm^2
    s_B = 0.000029 #surface energy of copper, J/cm^2
    D_L = 1e-5 #diffusion in liquid, cm^2/s
    D_S = 1e-9 #diffusion in solid, cm^2/s
    B_A = 0.33 #linear kinetic coefficient of nickel, cm/K/s
    B_B = 0.39 #linear kinetic coefficient of copper, cm/K/s
    v_m = 7.42 #molar volume, cm^3/mol
    R = 8.314 #gas constant, J/mol*K
    y_e = 0.12 #anisotropy

    #discretization params
    dx = 4.6e-6 #spacial division, cm
    idx = 1./dx
    dt = dx*dx/5./D_L/8.
    d = dx/0.94 #interfacial thickness

    #discretization dependent params, since d is constant, we compute them here for now
    ebar = math.sqrt(6.*math.sqrt(2.)*s_A*d/T_mA) #baseline energy
    eqbar = 0.5*ebar
    W_A = 3.*s_A/(math.sqrt(2.)*T_mA*d)
    W_B = 3.*s_B/(math.sqrt(2.)*T_mB*d)
    M_A = T_mA*T_mA*B_A/(6.*math.sqrt(2.)*L_A*d)
    M_B = T_mB*T_mB*B_B/(6.*math.sqrt(2.)*L_B*d)
    
    phi = fields[0]
    q1 = fields[1]
    q4 = fields[2]
    c = fields[3]
    
    pf_comp_xmm = transfer[0]
    pf_comp_ymm = transfer[1]
    t1_temp = transfer[2]
    t4_temp = transfer[3]
    temp = transfer[4]
    D_C = transfer[5]
    
    phi_out = fields_out[0]
    q1_out = fields_out[1]
    q4_out = fields_out[2]
    c_out = fields_out[3]
    
    #G_L = transfer[0]
    #G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
    
    for i in range(starty+1, fields[0].shape[0]-1, stridey):
        for j in range(startx+1, fields[0].shape[1]-1, stridex):
            #interpolating functions
            g = _g(phi[i][j])
            h = _h(phi[i][j])
            m = 1-h;
            M_q = 1e-6 + (M_qmax-1e-6)*m
            
            lq1 = grad2(q1[i][j], q1[i][j+1], q1[i][j-1], q1[i+1][j], q1[i-1][j], idx)
            lq4 = grad2(q4[i][j], q4[i][j+1], q4[i][j-1], q4[i+1][j], q4[i-1][j], idx)
            
            

            #this term is to evolve just the orientation, as done before the first real time step in the Dorr paper
            only_orientation = False

            if(only_orientation):
                deltac = 0
                deltaphi = 0
                t1 = eqbar*eqbar*lq1
                t4 = eqbar*eqbar*lq4
                lmbda = (q1[i][j]*t1+q4[i][j]*t4)
                deltaq1 = M_q*(t1-q1[i][j]*lmbda)
                deltaq4 = M_q*(t4-q4[i][j]*lmbda)

            else:
                #additional interpolating functions
                p = phi[i][j]*phi[i][j]
                pp = 2*phi[i][j]
                hprime = _hprime(phi[i][j])
                gprime = _gprime(phi[i][j])

                #bulk energy terms, using ideal solution model from Warren1995
                H_A = W_A*gprime - 30*L_A*(1/T-1/T_mA)*g
                H_B = W_B*gprime - 30*L_B*(1/T-1/T_mB)*g
                
                #quaternion gradient terms
                gq1_xm = (q1[i][j]-q1[i][j-1])*idx
                gq1_ym = (q1[i][j]-q1[i-1][j])*idx
                gq1_xp = (q1[i][j]-q1[i][j+1])*idx
                gq1_yp = (q1[i][j]-q1[i+1][j])*idx
                gq4_xm = (q4[i][j]-q4[i][j-1])*idx
                gq4_ym = (q4[i][j]-q4[i-1][j])*idx
                gq4_xp = (q4[i][j]-q4[i][j+1])*idx
                gq4_yp = (q4[i][j]-q4[i+1][j])*idx
                
                gqs_xm = gq1_xm*gq1_xm+gq4_xm*gq4_xm
                gqs_ym = gq1_ym*gq1_ym+gq4_ym*gq4_ym
                gqs_xp = gq1_xp*gq1_xp+gq4_xp*gq4_xp
                gqs_yp = gq1_yp*gq1_yp+gq4_yp*gq4_yp
                
                gqs = 0.5*(gqs_xm+gqs_ym+gqs_xp+gqs_yp)
                rgqs_0 = math.sqrt(gqs)
                
                beta = 1.5 #clipping value for grad-q-squared terms
                gqs_xm = max(gqs_xm, beta)
                gqs_ym = max(gqs_ym, beta)
                gqs_xp = max(gqs_xp, beta)
                gqs_yp = max(gqs_yp, beta)
                
                #these terms are also mag-grad-q in each respective direction, for f_ori_term function
                rgqs_xm = math.sqrt(gqs_xm)
                rgqs_ym = math.sqrt(gqs_ym)
                rgqs_xp = math.sqrt(gqs_xp)
                rgqs_yp = math.sqrt(gqs_yp)

                #change in c
                deltac = divagradb(temp[i][j], temp[i][j+1], temp[i][j-1], temp[i+1][j], temp[i-1][j], 
                                   phi[i][j], phi[i][j+1], phi[i][j-1], phi[i+1][j], phi[i-1][j], idx)
                deltac += divagradb(D_C[i][j], D_C[i][j+1], D_C[i][j-1], D_C[i+1][j], D_C[i-1][j], 
                                   c[i][j], c[i][j+1], c[i][j-1], c[i+1][j], c[i-1][j], idx)
                

                #change in phi
                lphi = grad2(phi[i][j], phi[i][j+1], phi[i][j-1], phi[i+1][j], phi[i-1][j], idx)
                pf_comp_x = 0.5*idx*(pf_comp_xmm[i][j+1] + pf_comp_xmm[i+1][j+1] - pf_comp_xmm[i][j] - pf_comp_xmm[i+1][j])
                pf_comp_y = 0.5*idx*(pf_comp_ymm[i+1][j] + pf_comp_ymm[i+1][j+1] - pf_comp_ymm[i][j] - pf_comp_ymm[i][j+1])
                M_phi = (1-c[i][j])*M_A + c[i][j]*M_B
                deltaphi = M_phi*(ebar*ebar*((1-3*y_e)*lphi + pf_comp_x + pf_comp_y)-(1-c[i][j])*H_A-c[i][j]*H_B-2*H*T*pp*rgqs_0)
                rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, threadId)
                alpha = 0.3
                deltaphi += M_phi*alpha*rand*(16*g)*((1-c[i][j])*H_A+c[i][j]*H_B)

                #changes in q, part 1
                base = 2*H*T
                D_q = base*p
                D_q_xp = base*fields[0][i][j+1]*fields[0][i][j+1]
                D_q_xm = base*fields[0][i][j-1]*fields[0][i][j-1]
                D_q_yp = base*fields[0][i+1][j]*fields[0][i+1][j]
                D_q_ym = base*fields[0][i-1][j]*fields[0][i-1][j]
                
                ######REACHED HERE FOR RETRANSLATION######

                gaq1 = f_ori_term(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, rgqs_xp, rgqs_xm, rgqs_yp, rgqs_ym, 
                                  q1[i][j], q1[i][j+1], q1[i][j-1], q1[i+1][j], q1[i-1][j], idx)
                gaq4 = f_ori_term(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, rgqs_xp, rgqs_xm, rgqs_yp, rgqs_ym, 
                                  q4[i][j], q4[i][j+1], q4[i][j-1], q4[i+1][j], q4[i-1][j], idx)

                cc_t1_temp = 0.25*(t1_temp[i][j]+t1_temp[i+1][j]+t1_temp[i][j+1]+t1_temp[i+1][j+1])
                cc_t4_temp = 0.25*(t4_temp[i][j]+t4_temp[i+1][j]+t4_temp[i][j+1]+t4_temp[i+1][j+1])

                t1 = eqbar*eqbar*lq1+(gaq1) + cc_t1_temp
                t4 = eqbar*eqbar*lq4+(gaq4) + cc_t4_temp
                lmbda = (q1[i][j]*t1+q4[i][j]*t4)
                deltaq1 = M_q*(t1-q1[i][j]*lmbda)
                deltaq4 = M_q*(t4-q4[i][j]*lmbda)

            #changes in q



            #apply changes
            phi_out[i][j] = phi[i][j] + deltaphi*dt
            q1_out[i][j] = q1[i][j] + deltaq1*dt
            q4_out[i][j] = q4[i][j] + deltaq4*dt
            c_out[i][j] = c[i][j] + deltac*dt
            renorm = math.sqrt((q1_out[i][j]**2+q4_out[i][j]**2))
            q1_out[i][j] = q1_out[i][j]/renorm
            q4_out[i][j] = q4_out[i][j]/renorm

@numba.jit
def get_thermodynamics(ufunc, array):
    if(len(array) == 3):
        G = ufunc(array[0], array[1], array[2])
        dGdc = 10000000.*(ufunc(array[0]+0.0000001, array[1]-0.0000001, array[2])-G)
    return G, dGdc
            
@cuda.jit
def AnisoDorr_helper_kernel(fields, T, transfer, rng_states, ufunc_array, params, c_params):
    #initializes certain arrays that are used in div-grad terms, to avoid recomputing terms many times
    #transfer[0] is pf_comp_x, defined at the vertex mm
    #transfer[1] is pf_comp_y, defined at the vertex mm
    #transfer[2] is t1_temp, defined at the vertex mm
    #transfer[3] is t4_temp, defined at the vertex mm
    #transfer[4] is "temp", the term used in the dcdt term for grad phi
    #transfer[5] is "D_C", the term used in the dcdt term for grad c
    startx, starty = cuda.grid(2)     
    stridex, stridey = cuda.gridsize(2) 
    threadId = cuda.grid(1)
    
    #v_m = params[2]
    #D_L = params[7]
    #D_S = params[8]
    #W = c_params[4]
    
    #temperature
    T = 1574.

    #bcc = 0L = e, fcc = 1S = d
    #material parameters, J, cm, K, s (except for R and Q terms, which use joules)
    M_qmax = 80000000. #maximum mobility of orientation, 1/(s*J)
    H = 1e-11 #interfacial energy term for quaternions, J/(K*cm)

    #material parameters, from Warren1995
    T_mA = 1728. #melting point of nickel
    T_mB = 1358. #melting point of copper
    L_A = 2350. #latent heat of nickel, J/cm^3
    L_B = 1728. #latent heat of copper, J/cm^3
    s_A = 0.000037 #surface energy of nickel, J/cm^2
    s_B = 0.000029 #surface energy of copper, J/cm^2
    D_L = 1e-5 #diffusion in liquid, cm^2/s
    D_S = 1e-9 #diffusion in solid, cm^2/s
    B_A = 0.33 #linear kinetic coefficient of nickel, cm/K/s
    B_B = 0.39 #linear kinetic coefficient of copper, cm/K/s
    v_m = 7.42 #molar volume, cm^3/mol
    R = 8.314 #gas constant, J/mol*K
    y_e = 0.12 #anisotropy

    #discretization params
    dx = 4.6e-6 #spacial division, cm
    idx = 1./dx
    dt = dx*dx/5./D_L/8.
    d = dx/0.94 #interfacial thickness

    #discretization dependent params, since d is constant, we compute them here for now
    ebar = math.sqrt(6.*math.sqrt(2.)*s_A*d/T_mA) #baseline energy
    eqbar = 0.5*ebar
    W_A = 3.*s_A/(math.sqrt(2.)*T_mA*d)
    W_B = 3.*s_B/(math.sqrt(2.)*T_mB*d)
    M_A = T_mA*T_mA*B_A/(6.*math.sqrt(2.)*L_A*d)
    M_B = T_mB*T_mB*B_B/(6.*math.sqrt(2.)*L_B*d)
    
    phi = fields[0]
    q1 = fields[1]
    q4 = fields[2]
    c = fields[3]
    
    pf_comp_xmm = transfer[0]
    pf_comp_ymm = transfer[1]
    t1_temp = transfer[2]
    t4_temp = transfer[3]
    temp = transfer[4]
    D_C = transfer[5]
    
    for i in range(startx, fields[0].shape[0], stridex):
        for j in range(starty, fields[0].shape[1], stridey):
            g = _g(phi[i][j])
            gprime = _gprime(phi[i][j])
            h = _h(phi[i][j])
            m = 1-h;
            H_A = W_A*gprime - 30*L_A*(1/T-1/T_mA)*g
            H_B = W_B*gprime - 30*L_B*(1/T-1/T_mB)*g
            
            #vertex_averaged_gphi
            gphi_xmm = 0.5*(phi[i][j]-phi[i][j-1]+phi[i-1][j]-phi[i-1][j-1])*idx
            gphi_ymm = 0.5*(phi[i][j]-phi[i-1][j]+phi[i][j-1]-phi[i-1][j-1])*idx
            
            #vertex_averaged_q1
            q1_mm = 0.25*(q1[i][j]+q1[i-1][j]+q1[i][j-1]+q1[i-1][j-1])
            q4_mm = 0.25*(q4[i][j]+q4[i-1][j]+q4[i][j-1]+q4[i-1][j-1])

            a2_b2 = q1_mm*q1_mm-q4_mm*q4_mm
            ab2 = 2.*q1_mm*q4_mm

            gpsi_xmm = a2_b2*gphi_xmm - ab2*gphi_ymm
            gpsi_ymm = a2_b2*gphi_ymm + ab2*gphi_xmm

            psi_xxy = gpsi_xmm*gpsi_xmm*gpsi_ymm
            psi_xyy = gpsi_xmm*gpsi_ymm*gpsi_ymm
            psi_xxyy = psi_xxy*gpsi_ymm

            mgphi2_mm = gpsi_xmm*gpsi_xmm + gpsi_ymm*gpsi_ymm
            mgphi2_mm = max(mgphi2_mm, 0.000000001)

            #change in c
            D_C[i][j] = D_S+m*(D_L-D_S)
            temp[i][j] = D_C*v_m*c*(1-c)*(H_B-H_A)/R

            #change in phi
            psix3 = gpsi_xmm*gpsi_xmm*gpsi_xmm
            psiy3 = gpsi_ymm*gpsi_ymm*gpsi_ymm
            pf_comp_xmm[i][j] = 4*y_e*((2*a2_b2*psix3 + 2*ab2*psiy3)/mgphi2_mm - gphi_xmm*(psix3*gphi_xmm + psiy3*gphi_ymm)/(mgphi2_mm*mgphi2_mm))
            pf_comp_ymm[i][j] = 4*y_e*((2*a2_b2*psiy3 - 2*ab2*psix3)/mgphi2_mm - gphi_ymm*(psix3*gphi_xmm + psiy3*gphi_ymm)/(mgphi2_mm*mgphi2_mm))

            q1px = q1_mm*gphi_xmm
            q1py = q1_mm*gphi_ymm
            q4px = q4_mm*gphi_xmm
            q4py = q4_mm*gphi_ymm

            t1_temp[i][j] = (16*ebar*ebar*y_e/mgphi2_mm)*(psi_xyy*(q1px - q4py) + psi_xxy*(q1py + q4px))
            t4_temp[i][j] = (16*ebar*ebar*y_e/mgphi2_mm)*(psi_xyy*(-q4px - q1py) + psi_xxy*(-q4py + q1px))
                
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
    
def engine_AnisoDorrGPU(sim):
    
    cuda.synchronize()
    AnisoDorr_helper_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device, sim.temperature_gpu_device,
                                                                          sim.transfer_gpu_device, sim.rng_states, 
                                                                          sim.ufunc_array, sim.params, sim.c_params)
    cuda.synchronize()
    AnisoDorr_kernel[sim.cuda_blocks, sim.cuda_threads_per_block](sim.fields_gpu_device, sim.temperature_gpu_device, 
                                                                   sim.transfer_gpu_device, sim.fields_out_gpu_device,
                                                                   sim.rng_states, sim.params, sim.c_params)
    cuda.synchronize()
    sim.fields_gpu_device, sim.fields_out_gpu_device = sim.fields_out_gpu_device, sim.fields_gpu_device
    
    
                
def init_AnisoDorrGPU(sim, dim=[200,200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb", temperature_type="isothermal", 
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
    #sim.load_tdb(tdb_path)
    sim.set_cell_spacing(cell_spacing)
    #sim.d = sim.get_cell_spacing()*d_ratio
    sim.set_engine(engine_AnisoDorrGPU)
    #init_tdb_parameters(sim)
    #if(temperature_type=="isothermal"):
    sim.set_temperature_isothermal(initial_temperature)
    #elif(temperature_type=="gradient"):
        #sim.set_temperature_gradient(initial_temperature, temperature_gradient, cooling_rate)
    #elif(temperature_type=="file"):
        #sim.set_temperature_file(temperature_file_path)
    #else:
        #print("Temperature type of "+temperature_type+" is not recognized, defaulting to isothermal with a temperature of "+str(initial_temperature))
        #sim.set_temperature_isothermal(initial_temperature)
    sim._components = ["CU", "NI"]
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
    #params.append(sim.get_cell_spacing())
    #params.append(sim.d)
    #params.append(sim.v_m)
    #params.append(sim.M_qmax)
    #params.append(sim.H)
    #params.append(sim.y_e)
    #params.append(sim.beta)
    #params.append(sim.D_L)
    #params.append(sim.D_S)
    #params.append(sim.get_time_step_length())
    #c_params.append(sim.L)
    #c_params.append(sim.T_M)
    #c_params.append(sim.S)
    #c_params.append(sim.B)
    #c_params.append(sim.W)
    #c_params.append(sim.M)
    sim.params = np.array(params)
    sim.c_params = np.array(c_params)
    sim.rng_states = create_xoroshiro128p_states(256*256, seed=3)
    out_dim = dim.copy()
    #out_dim.insert(0, len(sim._components)+2)
    out_dim.insert(0, 4)
    sim.fields_out_gpu_device = cuda.device_array(out_dim)
    transfer_dim = dim.copy()
    #transfer_dim.insert(0, 2*len(sim._components))
    transfer_dim.insert(0, 6)
    sim.transfer_gpu_device = cuda.device_array(transfer_dim)
    ufunc_array_dim = dim.copy()
    ufunc_array_dim.append(len(sim._components)+1)
    sim.ufunc_array = cuda.device_array(ufunc_array_dim)