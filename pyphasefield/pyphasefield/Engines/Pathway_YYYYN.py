import numpy as np
import sympy as sp
from scipy.sparse.linalg import gmres
from numba import cuda
import numba
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

try:
    #import from within Engines folder
    from ..field import Field
    from ..simulation import Simulation
    from ..ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
except:
    try:
        #import classes from pyphasefield library
        from pyphasefield.field import Field
        from pyphasefield.simulation import Simulation
        from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
    except:
        raise ImportError("Cannot import from pyphasefield library!")
        
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
    
    phi = fields[0]
    q1 = fields[1]
    q4 = fields[2]
    c = fields[3]
    
    phi_out = fields_out[0]
    q1_out = fields_out[1]
    q4_out = fields_out[2]
    c_out = fields_out[3]
    
    G_L = transfer[0]
    G_S = transfer[1]
    pf_comp_xmm = transfer[2]
    pf_comp_ymm = transfer[3]
    t1_temp = transfer[4]
    t4_temp = transfer[5]
    #M_c is transfer 6 to len(fields)+2 (for 2 components, eg Ni and Cu, M_c is just 6
    #dFdc is transfer len(fields)+3 to 2*len(fields)-1 (for 2 components, eg Ni and Cu, dFdc is just 7
    
    idx = 1./dx
    ebar2 = 6.*math.sqrt(2.)*S[1]*d/T_M[1]
    eqbar2 = 0.25*ebar2
    
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
                t1 = eqbar2*lq1
                t4 = eqbar2*lq4
                lmbda = (q1[i][j]*t1+q4[i][j]*t4)
                deltaq1 = M_q*(t1-q1[i][j]*lmbda)
                deltaq4 = M_q*(t4-q4[i][j]*lmbda)

            else:
                #additional interpolating functions
                p = phi[i][j]*phi[i][j]
                pp = 2*phi[i][j]
                hprime = _hprime(phi[i][j])
                gprime = _gprime(phi[i][j])
                
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
                
                #dcidt
                for l in range(3, len(fields)):
                    c_i_out = fields_out[l]
                    M_c = transfer[l+3]
                    dFdci = transfer[l+len(fields)]
                    c_i_out[i][j] = (divagradb(M_c[i][j], M_c[i][j+1], M_c[i][j-1], M_c[i+1][j], M_c[i-1][j], 
                                               dFdci[i][j], dFdci[i][j+1], dFdci[i][j-1], dFdci[i+1][j], dFdci[i-1][j], idx))
                    for m in range(3, len(fields)):
                        c_j = fields[m]
                        dFdcj = transfer[m+len(fields)]
                        c_i_out[i][j] -= divagradb(M_c[i][j]*c_j[i][j], M_c[i][j+1]*c_j[i][j+1], M_c[i][j-1]*c_j[i][j-1], 
                                                   M_c[i+1][j]*c_j[i+1][j], M_c[i-1][j]*c_j[i-1][j], 
                                                   dFdcj[i][j], dFdcj[i][j+1], dFdcj[i][j-1], dFdcj[i+1][j], dFdcj[i-1][j], idx)
                

                #change in phi
                lphi = grad2(phi[i][j], phi[i][j+1], phi[i][j-1], phi[i+1][j], phi[i-1][j], idx)
                pf_comp_x = 0.5*idx*(pf_comp_xmm[i][j+1] + pf_comp_xmm[i+1][j+1] - pf_comp_xmm[i][j] - pf_comp_xmm[i+1][j])
                pf_comp_y = 0.5*idx*(pf_comp_ymm[i+1][j] + pf_comp_ymm[i+1][j+1] - pf_comp_ymm[i][j] - pf_comp_ymm[i][j+1])
                deltaphi = ebar2*((1-3*y_e)*T[i][j]*lphi + pf_comp_x + pf_comp_y)-2*H*T[i][j]*pp*rgqs_0
                deltaphi -= hprime*(G_S[i][j] - G_L[i][j])/v_m
                deltaphi -= gprime*T[i][j]*cW
                deltaphi *= M_phi

                #changes in q, part 1
                base = 2*H*T[i][j]
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

                t1 = eqbar2*lq1+(gaq1) #+ cc_t1_temp
                t4 = eqbar2*lq4+(gaq4) #+ cc_t4_temp
                lmbda = (q1[i][j]*t1+q4[i][j]*t4)
                deltaq1 = M_q*(t1-q1[i][j]*lmbda)
                deltaq4 = M_q*(t4-q4[i][j]*lmbda)

            #changes in q



            #apply changes
            phi_out[i][j] = phi[i][j] + deltaphi*dt
            q1_out[i][j] = q1[i][j] + deltaq1*dt
            q4_out[i][j] = q4[i][j] + deltaq4*dt
            c_out[i][j] *= dt
            c_out[i][j] += c[i][j]
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
    
    dx = params[0]
    d = params[1]
    v_m = params[2]
    y_e = params[5]
    D_L = params[7]
    D_S = params[8]
    T_M = c_params[1]
    S = c_params[2]
    W = c_params[4]
    
    phi = fields[0]
    q1 = fields[1]
    q4 = fields[2]
    c = fields[3]
    
    G_L = transfer[0]
    G_S = transfer[1]
    pf_comp_xmm = transfer[2]
    pf_comp_ymm = transfer[3]
    t1_temp = transfer[4]
    t4_temp = transfer[5]
    
    idx = 1./dx
    ebar2 = 6.*math.sqrt(2.)*S[1]*d/T_M[1]
    
    for i in range(starty, fields[0].shape[0], stridey):
        for j in range(startx, fields[0].shape[1], stridex):
            c_N = 1.
            for l in range(3, len(fields)):
                ufunc_array[i][j][l-3] = fields[l][i][j]
                c_N -= fields[l][i][j]
            ufunc_array[i][j][len(fields)-3] = c_N
            ufunc_array[i][j][len(fields)-2] = T[i][j]
            #dGdc = numba.cuda.local.array((2,1), numba.float64)
            G_L[i][j], dGLdc = get_thermodynamics(ufunc_g_l, ufunc_array[i][j])
            G_S[i][j], dGSdc = get_thermodynamics(ufunc_g_s, ufunc_array[i][j])
            
            g = _g(phi[i][j])
            gprime = _gprime(phi[i][j])
            h = _h(phi[i][j])
            m = 1-h;
            
            noise_c = math.sqrt(2.*8.314*T[i][j]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
            
            for l in range(3, len(fields)):
                M_c = transfer[l+3]
                dFdc = transfer[l+len(fields)]
                M_c[i][j] = v_m*fields[l][i][j]*(D_L + h*(D_S - D_L))/(8.314*T[i][j])
                dFdc[i][j] = (dGLdc + h*(dGSdc-dGLdc))/v_m + (W[l-3]-W[len(fields)-3])*g*T[i][j]+noise_c

    for i in range(starty+1, fields[0].shape[0], stridey):
        for j in range(startx+1, fields[0].shape[1], stridex):
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
            
            #change in phi
            psix3 = gpsi_xmm*gpsi_xmm*gpsi_xmm
            psiy3 = gpsi_ymm*gpsi_ymm*gpsi_ymm
            pf_comp_xmm[i][j] = 4*T[i][j]*y_e*((2*a2_b2*psix3 + 2*ab2*psiy3)/mgphi2_mm - gphi_xmm*(psix3*gphi_xmm + psiy3*gphi_ymm)/(mgphi2_mm*mgphi2_mm))
            pf_comp_ymm[i][j] = 4*T[i][j]*y_e*((2*a2_b2*psiy3 - 2*ab2*psix3)/mgphi2_mm - gphi_ymm*(psix3*gphi_xmm + psiy3*gphi_ymm)/(mgphi2_mm*mgphi2_mm))
            
            q1px = q1_mm*gphi_xmm
            q1py = q1_mm*gphi_ymm
            q4px = q4_mm*gphi_xmm
            q4py = q4_mm*gphi_ymm

            t1_temp[i][j] = (16*ebar2*T[i][j]*y_e/mgphi2_mm)*(psi_xyy*(q1px - q4py) + psi_xxy*(q1py + q4px))
            t4_temp[i][j] = (16*ebar2*T[i][j]*y_e/mgphi2_mm)*(psi_xyy*(-q4px - q1py) + psi_xxy*(-q4py + q1px))
                
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
    
    
class Pathway_YYYYN(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #additional initialization code goes below
        #runs *before* tdb, thermal, fields, and boundary conditions are loaded/initialized
        self.uses_gpu = True
        self._framework = "GPU_SERIAL" #must be this framework for this engine
        self.user_data["d_ratio"] = 1./0.94 #default value
        
    def init_tdb_params(self):
        super().init_tdb_params()
        #additional tdb-related code goes below
        #runs *after* tdb file is loaded, tdb_phases and tdb_components are initialized
        #runs *before* thermal, fields, and boundary conditions are loaded/initialized
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
            self.user_data["eqbar"] = 0.5*self.user_data["ebar"]
            self.set_time_step_length(self.get_cell_spacing()**2/5./self.user_data["D_L"]/8)
            self.user_data["beta"] = 1.5
        except Exception as e:
            print("Could not load every parameter required from the TDB file!")
            print(e)
            raise Exception()
            
    def init_fields(self):
        #initialization of fields code goes here
        #runs *after* tdb and thermal data is loaded/initialized
        #runs *before* boundary conditions are initialized
        self._num_transfer_arrays = 2*len(self._tdb_components)+4
        self._tdb_ufunc_input_size = len(self._tdb_components)+1
        self.user_data["rng_states"] = create_xoroshiro128p_states(256*256, seed=3446621627)
        dim = self.dimensions
        try:
            sim_type = self.user_data["sim_type"]
            if(sim_type == "seed"):
                #initialize phi, q1, q4
                
                phi = np.zeros(dim)
                q1 = np.zeros(dim)
                q4 = np.zeros(dim)
                try:
                    melt_angle = self.user_data["melt_angle"]
                except:
                    print("self.user_data[\"melt_angle\"] not defined, defaulting to 0")
                    melt_angle = 0*np.pi/8
                #angle is halved because that is how quaternions do
                q1 += np.cos(0.5*melt_angle)
                q4 += np.sin(0.5*melt_angle)
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
                    assert(len(initial_concentration_array) == 1)
                    c_n = np.zeros(dim)
                    c_n += initial_concentration_array[0]
                    self.add_field(c_n, "c_CU", colormap=COLORMAP_OTHER)
                        
                except: #initial_concentration array isnt defined?
                    c_n = np.zeros(dim)
                    c_n += 0.5
                    self.add_field(c_n, "c_CU", colormap=COLORMAP_OTHER)
                        
            elif(sim_type=="seeds"):
                #initialize phi, q1, q4
                phi = np.zeros(dim)
                q1 = np.zeros(dim)
                q4 = np.zeros(dim)
                melt_angle = 0*np.pi/8
                #angle is halved because that is how quaternions do
                q1 += np.cos(0.5*melt_angle)
                q4 += np.sin(0.5*melt_angle)

                for j in range(number_of_seeds):
                    seed_angle = (np.random.rand()-0.5)*np.pi/2
                    x_pos = int(np.random.rand()*dim[1])
                    y_pos = int(np.random.rand()*dim[0])
                    phi, q1, q4 = make_seed(phi, q1, q4, x_pos, y_pos, seed_angle, 5)

                self.add_field(phi, "phi", colormap=COLORMAP_PHASE)
                self.add_field(q1, "q1")
                self.add_field(q4, "q4")

                #initialize concentration array(s)
                try:
                    initial_concentration_array = self.user_data["initial_concentration_array"]
                    assert(len(initial_concentration_array) == 1)
                    c_n = np.zeros(dim)
                    c_n += initial_concentration_array[0]
                    self.add_field(c_n, "c_CU", colormap=COLORMAP_OTHER)
                        
                except: #initial_concentration array isnt defined?
                    c_n = np.zeros(dim)
                    c_n += 0.5
                    self.add_field(c_n, "c_CU", colormap=COLORMAP_OTHER)
        
        except:
            phi = np.zeros(dim)
            q1 = np.zeros(dim)
            q4 = np.zeros(dim)
            melt_angle = 0
            #angle is halved because that is how quaternions do
            q1 += np.cos(0.5*melt_angle)
            q4 += np.sin(0.5*melt_angle)
            self.add_field(phi, "phi", colormap=COLORMAP_PHASE)
            self.add_field(q1, "q1")
            self.add_field(q4, "q4")
            #initialize concentration array(s)
            try:
                initial_concentration_array = self.user_data["initial_concentration_array"]
                assert(len(initial_concentration_array) == 1)
                c_n = np.zeros(dim)
                c_n += initial_concentration_array[0]
                self.add_field(c_n, "c_CU", colormap=COLORMAP_OTHER)

            except: #initial_concentration array isnt defined?
                c_n = np.zeros(dim)
                c_n += 0.5
                self.add_field(c_n, "c_CU", colormap=COLORMAP_OTHER)
        
    def initialize_fields_and_imported_data(self):
        super().initialize_fields_and_imported_data()
        #initialization of fields/imported data goes below
        #runs *after* tdb, thermal, fields, and boundary conditions are loaded/initialized
                        
    def just_before_simulating(self):
        super().just_before_simulating()
        #additional code to run just before beginning the simulation goes below
        #runs immediately before simulating, no manual changes permitted to changes implemented here
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
        c_params.append(self.user_data["L"])
        c_params.append(self.user_data["T_M"])
        c_params.append(self.user_data["S"])
        c_params.append(self.user_data["B"])
        c_params.append(self.user_data["W"])
        c_params.append(self.user_data["M"])
        self.user_data["params"] = np.array(params)
        self.user_data["c_params"] = np.array(c_params)
        
    def simulation_loop(self):
        #code to run each simulation step goes here
        cuda.synchronize()
        if(len(self.dimensions) == 1):
            AnisoDorr_helper_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            AnisoDorr_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
        elif(len(self.dimensions) == 2):
            AnisoDorr_helper_kernel[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            AnisoDorr_kernel[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
        elif(len(self.dimensions) == 3):
            AnisoDorr_helper_kernel[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params"], self.user_data["c_params"])
            cuda.synchronize()
            AnisoDorr_kernel[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params"], self.user_data["c_params"])
        cuda.synchronize()
        self._fields_gpu_device, self._fields_out_gpu_device = self._fields_out_gpu_device, self._fields_gpu_device