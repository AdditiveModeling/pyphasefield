import numpy as np
import sympy as sp
from ..field import Field

def find_Pn(T_M, T, Q ,dt):
    #finding the probability of forming a critical nucleus, nucleating only every 500 time steps
    #input: T_M -- temperature of the liquidus, T -- Temperature, Q -- activation energy for migration
    #choose free parameters a and b
    #code by Vera Titze
    a=10**28
    b=2.8*10**3
    e=2.7182818284590452353
    R=8.314
    J0=a*e**(-b/(T_M-T))*e**(-Q/(R*T))
    Pn=1-e**(-J0*dt*500)
    return J0,Pn

def add_nuclei(phi, q1, q4, p11, size):
    #adds nuclei to the phi, q1, and q4 fields for a given probability array, p11
    #code by Vera Titze
    random=np.random.random((size, size))
    nuclei_centers=np.argwhere(random<p11)
    print('number of nuclei added: ',len(nuclei_centers))
    for center in nuclei_centers:
        angle=np.random.random()
        for i in range((int)(center[0]-5), (int)(center[0]+5)):
            for j in range((int)(center[1]-5), (int)(center[1]+5)):
                if (i>=0 and i<size and j<size and j>=0):
                    if((i-center[0])*(i-center[0])+(j-center[1])*(j-center[1]) < 25):
                        if(phi[i][j]<0.2):
                            phi[i][j] = 1
                            q1[i][j] = np.cos(angle*2*np.pi)
                            q4[i][j] = np.sin(angle*2*np.pi)
    return phi, q1, q4

def compute_tdb_energy_nc(sim, temps, c, phase):
    """
    Computes Gibbs Free Energy and its derivative*S* w.r.t. composition, for a given temperature field and list of composition fields
    Derivatives are computed by holding all other explicit composition variables constant
    c_i is increased, c_N is decreased (the implicitly-defined last composition variable which equals 1-sum(c_i) )
    
    Input parameters:
        - sim: the Simulation object. The method retrieves the TDB pycalphad object (sim._tdb) and the components used (sim._components)
            from this variable.
        - temps: the temperature array. Could also be retrieved as sim.temperature
        - c: the list of composition arrays. The format is a python list of numpy ndarrays
        - phase: the String which corresponds to a particular phase in the TDB file. E.g.: "FCC_A1" or "LIQUID"
    
    Returns GM (Molar Gibbs Free Energy) and dGdci (list of derivatives of GM, w.r.t. c_i)
    """
    import pycalphad as pyc
    #alphabetical order of components!
    fec = [] #flattened expanded c
    for i in range(len(c)):
            fec.append(np.expand_dims(c[i].flatten(), axis=1))
    fec_n_comp = np.ones(fec[0].shape)
    for i in range(len(c)):
        fec_n_comp -= fec[i]
    for i in range(len(c)):
        fec_n_comp = np.concatenate((fec_n_comp, fec[i]), axis=1)
    #move final component to end, maybe ill find a way to write this better in the future...
    fec_n_comp = np.roll(fec_n_comp, -1, axis=1)
    #offset composition, for computing slope of GM w.r.t. comp
    fec_nc_offset = []
    for i in range(len(c)):
        fec_offset = np.zeros([len(c)+1])
        fec_offset[i] = 0.0000001
        fec_offset[len(c)] = -0.0000001
        fec_nc_offset.append(fec_n_comp+fec_offset)
    flattened_t = temps.flatten()
    GM = pyc.calculate(sim._tdb, sim._components, phase, P=101325, T=flattened_t, points=fec_n_comp, broadcast=False).GM.values.reshape(c[0].shape)
    GM_derivs = []
    for i in range(len(c)):
        GM_derivs.append((pyc.calculate(sim._tdb, sim._components, phase, P=101325, T=flattened_t, points=fec_nc_offset[i], broadcast=False).GM.values.reshape(c[0].shape)-GM)*(10000000.))
    return GM, GM_derivs

def __h(phi):
    #h function from Dorr2010
    return phi*phi*phi*(10-15*phi+6*phi*phi)

def __hprime(phi):
    #derivative of the h function from Dorr2010, w.r.t phi
    return (30*phi*phi*(1-phi)*(1-phi))

def __g(phi):
    #g function from Warren1995. Similar to that used in Dorr2010 and Granasy2014
    return (phi*phi*(1-phi)*(1-phi))

def __gprime(phi):
    #derivative of g function, w.r.t phi
    return (4*phi*phi*phi - 6*phi*phi +2*phi)

#Numpy vectorized versions of above functions
_h = np.vectorize(__h)
_hprime = np.vectorize(__hprime)
_g = np.vectorize(__g) 
_gprime = np.vectorize(__gprime)

def grad_l(phi, dx, dim):
    r = []
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        r.append((phi-phim)/(dx))
    return r

def grad_r(phi, dx, dim):
    r = []
    for i in range(dim):
        phip = np.roll(phi, -1, i)
        r.append((phip-phi)/(dx))
    return r

def grad2(phi, dx, dim):
    r = np.zeros_like(phi)
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        phip = np.roll(phi, -1, i)
        r += (phip+phim-2*phi)/(dx*dx)
    return r

def divagradb(a, b, dx, dim):
    r = np.zeros_like(b)
    for i in range(dim):
        agradb = ((a + np.roll(a, -1, i))/2)*(np.roll(b, -1, i) - b)/dx
        r += (agradb - np.roll(agradb, 1, i))/dx
    return r

def gaq(gql, gqr, rgqsl, rgqsr, dqc, dx, dim):
    r = np.zeros_like(dqc)
    for i in range(dim):
        r += ((0.5*(dqc+np.roll(dqc, -1, i))*gqr[i]/rgqsr[i])-(0.5*(dqc+np.roll(dqc, 1, i))*gql[i]/rgqsl[i]))/(dx)
    return r

def renormalize(q1, q4):
    q = np.sqrt(q1*q1+q4*q4)
    return q1/q, q4/q
    
def npvalue(var, string, tdb):
    """
    Returns a numpy float from the sympy expression gotten from pycalphad
    Reason: some numpy functions (i.e. sqrt) are incompatible with sympy floats!
    """
    return sp.lambdify(var, tdb.symbols[string], 'numpy')(1000)

def engine_NComponent(sim):
    #load fields to easier-to-reference variables
    phi = sim.fields[0].data
    q1 = sim.fields[1].data
    q4 = sim.fields[2].data
    c = []
    for i in range(3, len(sim.fields)):
        c.append(sim.fields[i].data)
    T = sim.temperature.data
    dim = len(phi.shape)
    dx = sim.get_cell_spacing()
    
    g = _g(phi)
    h = _h(phi)
    m = 1-h;
    M_q = 1e-6 + (sim.M_qmax-1e-6)*m

    lq1 = grad2(q1, dx, dim)
    lq4 = grad2(q4, dx, dim)

    #additional interpolating functions
    p = phi*phi
    hprime = _hprime(phi)
    gprime = _gprime(phi)

    #quaternion gradient terms
    gq1l = grad_l(q1, dx, dim)
    gq4l = grad_l(q4, dx, dim)
    gqsl = []
    for j in range(dim):
        gqsl.append(gq1l[j]*gq1l[j]+gq4l[j]*gq4l[j])

    gq1r = grad_r(q1, dx, dim)
    gq4r = grad_r(q4, dx, dim)

    gqsr = []
    for j in range(dim):
        gqsr.append(np.roll(gqsl[j], -1, j))

    gqs = (gqsl[0]+gqsr[0])/2
    for j in range(1, dim):
        gqs += (gqsl[j]+gqsr[j])/2
    rgqs_0 = np.sqrt(gqs)

    gphi = grad_l(phi, dx, 2)
    vertex_averaged_gphi = []
    vertex_averaged_gphi.append((gphi[0]+np.roll(gphi[0], 1, 1))/2.)
    vertex_averaged_gphi.append((gphi[1]+np.roll(gphi[1], 1, 0))/2.)
    vertex_averaged_q1 = (q1 + np.roll(q1, 1, 0))/2.
    vertex_averaged_q1 = (vertex_averaged_q1 + np.roll(vertex_averaged_q1, 1, 1))/2.
    vertex_averaged_q4 = (q4 + np.roll(q4, 1, 0))/2.
    vertex_averaged_q4 = (vertex_averaged_q4 + np.roll(vertex_averaged_q4, 1, 1))/2.
    vertex_averaged_T = (T + np.roll(T, 1, 0))/2.
    vertex_averaged_T = (vertex_averaged_T + np.roll(vertex_averaged_T, 1, 1))/2.

    a2_b2 = vertex_averaged_q1*vertex_averaged_q1-vertex_averaged_q4*vertex_averaged_q4
    ab2 = 2.*vertex_averaged_q1*vertex_averaged_q4

    vertex_centered_gpsi = []
    vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[0] - ab2*vertex_averaged_gphi[1])
    vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[1] + ab2*vertex_averaged_gphi[0])

    psi_xxy = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[1]
    psi_xyy = vertex_centered_gpsi[0]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
    psi_xxyy = psi_xxy*vertex_centered_gpsi[1]

    vertex_centered_mgphi2 = vertex_averaged_gphi[0]*vertex_averaged_gphi[0] + vertex_averaged_gphi[1]*vertex_averaged_gphi[1]

    #"clip" the grid: if values are smaller than sim.beta, set them equal to sim.beta
    #also clip the mgphi values to avoid divide by zero errors!
    for j in range(dim):
        gqsl[j] = np.clip(gqsl[j], sim.beta, np.inf)
        gqsr[j] = np.clip(gqsr[j], sim.beta, np.inf)

    vertex_centered_mgphi2 = np.clip(vertex_centered_mgphi2, 0.000000001, np.inf)

    rgqsl = []
    rgqsr = []
    for j in range(dim):
        rgqsl.append(np.sqrt(gqsl[j]))
        rgqsr.append(np.sqrt(gqsr[j]))

    #compute values from tdb
    G_L, dGLdc = compute_tdb_energy_nc(sim, T, c, "LIQUID")
    G_S, dGSdc = compute_tdb_energy_nc(sim, T, c, "FCC_A1")

    #change in c1, c2
    M_c = []
    dFdc = []
    deltac = []
    #find the standard deviation as an array 
    std_c=np.sqrt(np.absolute(2*sim.R*T/sim.v_m))
    for j in range(len(c)):
        #find the actual random noise
        noise_c=np.random.normal(0, std_c, phi.shape)
        M_c.append(sim.v_m*c[j]*(sim.D_S+m*(sim.D_L-sim.D_S))/sim.R/1574.)
        #add the change in noise inside the functional
        dFdc.append((dGSdc[j] + m*(dGLdc[j]-dGSdc[j]))/sim.v_m + (sim.W[j]-sim.W[len(c)])*g*T+noise_c)
    for j in range(len(c)):
        deltac.append(divagradb(M_c[j]*(1-c[j]), dFdc[j], dx, dim))
        for k in range(len(c)):
            if not (j == k):
                deltac[j] -= divagradb(M_c[j]*c[k], dFdc[k], dx, dim)

    #change in phi
    divTgradphi = divagradb(T, phi, dx, dim)

    #compute overall order mobility, from order mobility coefficients
    c_N = 1-np.sum(c, axis=0)
    M_phi = c_N*sim.M[len(c)]
    for j in range(len(c)):
        M_phi += c[j]*sim.M[j]

    #compute well size term for N-components
    well = c_N*sim.W[len(c)]
    for j in range(len(c)):
        well += c[j]*sim.W[j]
    well *= (T*gprime)

    psix3 = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[0]
    psiy3 = vertex_centered_gpsi[1]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
    pf_comp_x = 8*sim.y_e*T*((2*a2_b2*psix3 + 2*ab2*psiy3)/vertex_centered_mgphi2 - vertex_averaged_gphi[0]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
    pf_comp_x = (np.roll(pf_comp_x, -1, 0) - pf_comp_x)/dx
    pf_comp_x = (np.roll(pf_comp_x, -1, 1) + pf_comp_x)/2.
    pf_comp_y = 8*sim.y_e*T*((2*a2_b2*psiy3 - 2*ab2*psix3)/vertex_centered_mgphi2 - vertex_averaged_gphi[1]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
    pf_comp_y = (np.roll(pf_comp_y, -1, 1) - pf_comp_y)/dx
    pf_comp_y = (np.roll(pf_comp_y, -1, 0) + pf_comp_y)/2.
    deltaphi = M_phi*(sim.ebar*sim.ebar*((1-3*sim.y_e)*divTgradphi + pf_comp_x + pf_comp_y)-30*g*(G_S-G_L)/sim.v_m-well-4*sim.H*T*phi*rgqs_0)

    #old noise from Warren1995:
    #randArray = 2*np.random.random_sample(shape)-1
    #alpha = 0.3
    #deltaphi += M_phi*alpha*randArray*(16*g)*(30*g*(G_S-G_L)/v_m+well)

    #noise in phi, based on Langevin Noise
    std_phi=np.sqrt(np.absolute(2*sim.R*M_phi*T/sim.v_m))
    noise_phi=np.random.normal(0, std_phi, phi.shape)
    deltaphi += noise_phi

    #changes in q, part 1
    dq_component = 2*sim.H*T*p

    gaq1 = gaq(gq1l, gq1r, rgqsl, rgqsr, dq_component, dx, dim)
    gaq4 = gaq(gq4l, gq4r, rgqsl, rgqsr, dq_component, dx, dim)

    q1px = vertex_averaged_q1*vertex_averaged_gphi[0]
    q1py = vertex_averaged_q1*vertex_averaged_gphi[1]
    q4px = vertex_averaged_q4*vertex_averaged_gphi[0]
    q4py = vertex_averaged_q4*vertex_averaged_gphi[1]

    t1_temp = (16*sim.ebar*sim.ebar*T*sim.y_e/vertex_centered_mgphi2)*(psi_xyy*(q1px - q4py) + psi_xxy*(q1py + q4px))
    t4_temp = (16*sim.ebar*sim.ebar*T*sim.y_e/vertex_centered_mgphi2)*(psi_xyy*(-q4px - q1py) + psi_xxy*(-q4py + q1px))
    cc_t1_temp = (t1_temp + np.roll(t1_temp, -1, 0))/2.
    cc_t1_temp = (cc_t1_temp + np.roll(cc_t1_temp, -1, 1))/2.
    cc_t4_temp = (t4_temp + np.roll(t4_temp, -1, 0))/2.
    cc_t4_temp = (cc_t4_temp + np.roll(cc_t4_temp, -1, 1))/2.
    
    t1 = gaq1 + cc_t1_temp
    t4 = gaq4 + cc_t4_temp

    #add Dorr2010 term to quaternion field
    t1 += sim.eqbar*sim.eqbar*lq1
    t4 += sim.eqbar*sim.eqbar*lq4
    
    ### add noise to quaternion field ###
    ### NOT stable, unsure why at the moment ###
    
    #std_q1=np.sqrt(np.absolute(2*sim.R*T/sim.v_m))
    #noise_q1=np.random.normal(0, std_q1, q1.shape)
    #std_q4=np.sqrt(np.absolute(2*sim.R*T/sim.v_m))
    #noise_q4=np.random.normal(0, std_q4, q4.shape)
    #t1 += noise_q1
    #t4 += noise_q4
    
    lmbda = (q1*t1+q4*t4)
    deltaq1 = M_q*(t1-q1*lmbda)
    deltaq4 = M_q*(t4-q4*lmbda)

    #changes in q



    #apply changes
    dt = sim.get_time_step_length()
    for j in range(len(c)):
        sim.fields[3+j].data += deltac[j]*dt
    sim.fields[0].data += deltaphi*dt
    sim.fields[1].data += deltaq1*dt
    sim.fields[2].data += deltaq4*dt
    
    #always renormalize quaternion fields after every step
    sim.fields[1].data, sim.fields[2].data = renormalize(sim.fields[1].data, sim.fields[2].data)

    #This code segment prints the progress after every 5% of the simulation is done (for convenience)
    #disabled for now, may bring it back in the simulate function of simulation.py
    #if(steps > 19):
        #if(i%(steps/20) == 0):
            #print(str(5*i/(steps/20))+"% done...")

    #This code segment adds nuclei
    #note: nuclei are added *before* saving the data, so stray nuclei may be found before evolving the system
    #step = sim.get_time_step_counter()
    #if(step%500 == 0):
        #find the stochastic nucleation critical probabilistic cutoff
        #attn -- Q and T_liq are hard coded parameters for Ni-10%Cu
        #Q0=8*10**5 #activation energy of migration
        #T_liq=1697 #Temperature of Liquidus (K)
        #J0,p11=find_Pn(T_liq, T, Q0, dt) 
        #print(J0)
        #phi, q1, q4=add_nuclei(phi, q1, q4, p11, len(phi))
        #saveArrays_nc(data_path, step, phi, c, q1, q4)
        
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

def init_tdb_parameters(sim):
    """
    Modifies the global vars which are parameters for the engine. Called from the function utils.preinitialize
    Returns True if variables are loaded successfully, False if certain variables dont exist in the TDB
    If false, preinitialize will print an error saying the TDB doesn't have enough info to run the sim
    """
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
            sim.W.append(3*sim.S[i]/(np.sqrt(2)*sim.T_M[i]*sim.d))
            sim.M.append(sim.T_M[i]*sim.T_M[i]*sim.B[i]/(6*np.sqrt(2)*sim.L[i]*sim.d)/1574.)
        sim.D_S = npvalue(T, "D_S", tdb)
        sim.D_L = npvalue(T, "D_L", tdb)
        sim.v_m = npvalue(T, "V_M", tdb)
        sim.M_qmax = npvalue(T, "M_Q", tdb)
        sim.H = npvalue(T, "H", tdb)
        sim.y_e = npvalue(T, "Y_E", tdb)
        sim.ebar = np.sqrt(6*np.sqrt(2)*sim.S[1]*sim.d/sim.T_M[1])
        sim.eqbar = 0.5*sim.ebar
        sim.set_time_step_length(sim.get_cell_spacing()**2/5./sim.D_L/20)
        sim.beta = 1.5
        return True
    except Exception as e:
        print("Could not load every parameter required from the TDB file!")
        print(e)
        return False
    
def functional_NComponent():
    print("The functional for this engine is given by the following LaTeX expressions:")
    print("$$ F = \\int_\\Omega (f_{int} + f_{bulk} + f_{well} + f_{ori} + f_{q2} + \\lambda(1-\\sqrt{\\sum q_i^2}) $$")
    print("$$ f_{int} = \\frac{\\epsilon_\\phi^2\\eta T}{2}|\\nabla \\phi|^2 $$")
    print("$$ f_{bulk} = G_L(\\textbf{c}, T) + h(\\phi)(G_S(\\textbf{c}, T) - G_L(\\textbf{c}, T)) $$")
    print("$$ f_{well} = \\sum_i c_iW_ig(\\phi) $$")
    print("$$ f_{ori} = 2HTp(\\phi)|\\nabla \\textbf{q}| $$")
    print("$$ f_{q2} = \\frac{\\epsilon_q^2}{2}|\\nabla \\textbf{q}|^2 $$")
    print("$$ h(\\phi) = \\phi^3(6\\phi^2 - 15\\phi + 10) $$")
    print("$$ g(\\phi) = \\phi^2(1-\\phi)^2 $$")
    print("$$ p(\\phi) = \\phi^2 $$")
    print("$$ \\eta = 1 - 3\\gamma_\\epsilon + 4\\gamma_\\epsilon\\frac{\\psi_x^4 + \\psi_y^4 + \\psi_z^4}{|\\nabla \\phi|^4} $$")
    
def engine_params_NComponent(sim):
    #load fields to easier-to-reference variables
    phi = sim.fields[0].data
    q1 = sim.fields[1].data
    q4 = sim.fields[2].data
    c = []
    for i in range(3, len(sim.fields)):
        c.append(sim.fields[i].data)
    T = sim.temperature.data
    dim = len(phi.shape)
    dx = sim.get_cell_spacing()
    
    g = _g(phi)
    h = _h(phi)
    m = 1-h;
    M_q = 1e-6 + (sim.M_qmax-1e-6)*m

    lq1 = grad2(q1, dx, dim)
    lq4 = grad2(q4, dx, dim)

    #additional interpolating functions
    p = phi*phi
    hprime = _hprime(phi)
    gprime = _gprime(phi)

    #quaternion gradient terms
    gq1l = grad_l(q1, dx, dim)
    gq4l = grad_l(q4, dx, dim)
    gqsl = []
    for j in range(dim):
        gqsl.append(gq1l[j]*gq1l[j]+gq4l[j]*gq4l[j])

    gq1r = grad_r(q1, dx, dim)
    gq4r = grad_r(q4, dx, dim)

    gqsr = []
    for j in range(dim):
        gqsr.append(np.roll(gqsl[j], -1, j))

    gqs = (gqsl[0]+gqsr[0])/2
    for j in range(1, dim):
        gqs += (gqsl[j]+gqsr[j])/2
    rgqs_0 = np.sqrt(gqs)

    gphi = grad_l(phi, dx, 2)
    vertex_averaged_gphi = []
    vertex_averaged_gphi.append((gphi[0]+np.roll(gphi[0], 1, 1))/2.)
    vertex_averaged_gphi.append((gphi[1]+np.roll(gphi[1], 1, 0))/2.)
    vertex_averaged_q1 = (q1 + np.roll(q1, 1, 0))/2.
    vertex_averaged_q1 = (vertex_averaged_q1 + np.roll(vertex_averaged_q1, 1, 1))/2.
    vertex_averaged_q4 = (q4 + np.roll(q4, 1, 0))/2.
    vertex_averaged_q4 = (vertex_averaged_q4 + np.roll(vertex_averaged_q4, 1, 1))/2.
    vertex_averaged_T = (T + np.roll(T, 1, 0))/2.
    vertex_averaged_T = (vertex_averaged_T + np.roll(vertex_averaged_T, 1, 1))/2.

    a2_b2 = vertex_averaged_q1*vertex_averaged_q1-vertex_averaged_q4*vertex_averaged_q4
    ab2 = 2.*vertex_averaged_q1*vertex_averaged_q4

    vertex_centered_gpsi = []
    vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[0] - ab2*vertex_averaged_gphi[1])
    vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[1] + ab2*vertex_averaged_gphi[0])

    psi_xxy = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[1]
    psi_xyy = vertex_centered_gpsi[0]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
    psi_xxyy = psi_xxy*vertex_centered_gpsi[1]

    vertex_centered_mgphi2 = vertex_averaged_gphi[0]*vertex_averaged_gphi[0] + vertex_averaged_gphi[1]*vertex_averaged_gphi[1]

    #"clip" the grid: if values are smaller than sim.beta, set them equal to sim.beta
    #also clip the mgphi values to avoid divide by zero errors!
    for j in range(dim):
        gqsl[j] = np.clip(gqsl[j], sim.beta, np.inf)
        gqsr[j] = np.clip(gqsr[j], sim.beta, np.inf)

    vertex_centered_mgphi2 = np.clip(vertex_centered_mgphi2, 0.000000001, np.inf)

    rgqsl = []
    rgqsr = []
    for j in range(dim):
        rgqsl.append(np.sqrt(gqsl[j]))
        rgqsr.append(np.sqrt(gqsr[j]))

    #compute values from tdb
    G_L, dGLdc = compute_tdb_energy_nc(sim, T, c, "LIQUID")
    G_S, dGSdc = compute_tdb_energy_nc(sim, T, c, "FCC_A1")

    #change in c1, c2
    M_c = []
    dFdc = []
    deltac = []
    #find the standard deviation as an array 
    std_c=np.sqrt(np.absolute(2*sim.R*T/sim.v_m))
    for j in range(len(c)):
        #find the actual random noise
        noise_c=np.random.normal(0, std_c, phi.shape)
        M_c.append(sim.v_m*c[j]*(sim.D_S+m*(sim.D_L-sim.D_S))/sim.R/1574.)
        #add the change in noise inside the functional
        dFdc.append((dGSdc[j] + m*(dGLdc[j]-dGSdc[j]))/sim.v_m + (sim.W[j]-sim.W[len(c)])*g*T+noise_c)
    for j in range(len(c)):
        deltac.append(divagradb(M_c[j]*(1-c[j]), dFdc[j], dx, dim))
        for k in range(len(c)):
            if not (j == k):
                deltac[j] -= divagradb(M_c[j]*c[k], dFdc[k], dx, dim)

    #change in phi
    divTgradphi = divagradb(T, phi, dx, dim)

    #compute overall order mobility, from order mobility coefficients
    c_N = 1-np.sum(c, axis=0)
    M_phi = c_N*sim.M[len(c)]
    for j in range(len(c)):
        M_phi += c[j]*sim.M[j]

    #compute well size term for N-components
    well = c_N*sim.W[len(c)]
    for j in range(len(c)):
        well += c[j]*sim.W[j]
    well *= (T*gprime)

    psix3 = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[0]
    psiy3 = vertex_centered_gpsi[1]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
    pf_comp_x = 8*sim.y_e*T*((2*a2_b2*psix3 + 2*ab2*psiy3)/vertex_centered_mgphi2 - vertex_averaged_gphi[0]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
    pf_comp_x = (np.roll(pf_comp_x, -1, 0) - pf_comp_x)/dx
    pf_comp_x = (np.roll(pf_comp_x, -1, 1) + pf_comp_x)/2.
    pf_comp_y = 8*sim.y_e*T*((2*a2_b2*psiy3 - 2*ab2*psix3)/vertex_centered_mgphi2 - vertex_averaged_gphi[1]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
    pf_comp_y = (np.roll(pf_comp_y, -1, 1) - pf_comp_y)/dx
    pf_comp_y = (np.roll(pf_comp_y, -1, 0) + pf_comp_y)/2.
    deltaphi = M_phi*(sim.ebar*sim.ebar*((1-3*sim.y_e)*divTgradphi + pf_comp_x + pf_comp_y)-30*g*(G_S-G_L)/sim.v_m-well-4*sim.H*T*phi*rgqs_0)

    #noise in phi, based on Langevin Noise
    std_phi=np.sqrt(np.absolute(2*sim.R*M_phi*T/sim.v_m))
    noise_phi=np.random.normal(0, std_phi, phi.shape)
    deltaphi += noise_phi

    #changes in q, part 1
    dq_component = 2*sim.H*T*p

    gaq1 = gaq(gq1l, gq1r, rgqsl, rgqsr, dq_component, dx, dim)
    gaq4 = gaq(gq4l, gq4r, rgqsl, rgqsr, dq_component, dx, dim)

    q1px = vertex_averaged_q1*vertex_averaged_gphi[0]
    q1py = vertex_averaged_q1*vertex_averaged_gphi[1]
    q4px = vertex_averaged_q4*vertex_averaged_gphi[0]
    q4py = vertex_averaged_q4*vertex_averaged_gphi[1]

    t1_temp = (16*sim.ebar*sim.ebar*T*sim.y_e/vertex_centered_mgphi2)*(psi_xyy*(q1px - q4py) + psi_xxy*(q1py + q4px))
    t4_temp = (16*sim.ebar*sim.ebar*T*sim.y_e/vertex_centered_mgphi2)*(psi_xyy*(-q4px - q1py) + psi_xxy*(-q4py + q1px))
    cc_t1_temp = (t1_temp + np.roll(t1_temp, -1, 0))/2.
    cc_t1_temp = (cc_t1_temp + np.roll(cc_t1_temp, -1, 1))/2.
    cc_t4_temp = (t4_temp + np.roll(t4_temp, -1, 0))/2.
    cc_t4_temp = (cc_t4_temp + np.roll(cc_t4_temp, -1, 1))/2.
    
    t1 = gaq1 + cc_t1_temp
    t4 = gaq4 + cc_t4_temp

    #add Dorr2010 term to quaternion field
    t1 += sim.eqbar*sim.eqbar*lq1
    t4 += sim.eqbar*sim.eqbar*lq4
    
    lmbda = (q1*t1+q4*t4)
    deltaq1 = M_q*(t1-q1*lmbda)
    deltaq4 = M_q*(t4-q4*lmbda)


    #make dict to return, containing variables of interest
    rd = {}
    rd["deltaphi"] = {}
    rd["deltac"] = {}
    rd["deltaq1"] = {}
    rd["deltaq4"] = {}
    rd["energy"] = {}
    
    #add energetic terms
    psix4 = psix3*vertex_centered_gpsi[0]
    psiy4 = psiy3*vertex_centered_gpsi[1]
    psix4y4 = (psix4+psiy4)/vertex_centered_mgphi2
    psix4y4 = 0.5*(psix4y4 + np.roll(psix4y4, -1, 0))
    psix4y4 = 0.5*(psix4y4 + np.roll(psix4y4, -1, 1))
    eta = 1-3*sim.y_e+4*sim.y_e*(psix4y4)
    rd["energy"]["f_int"] = sim.ebar**2/2*eta*T
    wellenergy = c_N*sim.W[len(c)]
    for j in range(len(c)):
        wellenergy += c[j]*sim.W[j]
    wellenergy *= (T*g)
    rd["energy"]["f_well"] = wellenergy
    rd["energy"]["f_bulk"] = G_S + h*(G_L-G_S)
    rd["energy"]["f_ori"] = 2*sim.H*T*p*rgqs_0
    rd["energy"]["f_q2"] = 0.5*sim.eqbar**2 * (rgqs_0)**2
    
    return rd
    
        
def init_NComponent(sim, dim=[200,200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb", thermal_type="isothermal", 
                           initial_temperature=1574, thermal_gradient=0, cooling_rate=0, thermal_file_path="T.xdmf", 
                           initial_concentration_array=[0.40831], cell_spacing=0.0000046, d_ratio=1/0.94):
    if(len(dim) == 1):
        dim.append(1)
    sim.set_dimensions(dim)
    sim.load_tdb(tdb_path)
    sim.set_cell_spacing(cell_spacing)
    sim.d = sim.get_cell_spacing()*d_ratio
    sim.set_engine(engine_NComponent)
    init_tdb_parameters(sim)
    if(thermal_type=="isothermal"):
        sim.set_thermal_isothermal(initial_temperature)
    elif(thermal_type=="gradient"):
        sim.set_thermal_gradient(initial_temperature, thermal_gradient, cooling_rate)
    elif(thermal_type=="file"):
        sim.set_thermal_file(thermal_file_path)
    else:
        print("thermal type of "+thermal_type+" is not recognized, defaulting to isothermal with a temperature of "+str(initial_temperature))
        sim.set_thermal_isothermal(initial_temperature)
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
        phi_field = Field(data=phi, name="phi", simulation=sim)
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
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim)
                sim.add_field(c_n_field)
        else:
            assert((len(initial_concentration_array)+1) == len(sim._components))
            for i in range(len(initial_concentration_array)):
                c_n = np.zeros(dim)
                c_n += initial_concentration_array[i]
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim)
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
        
        phi_field = Field(data=phi, name="phi", simulation=sim)
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
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim)
                sim.add_field(c_n_field)
        else:
            assert((len(initial_concentration_array)+1) == len(sim._components))
            for i in range(len(initial_concentration_array)):
                c_n = np.zeros(dim)
                c_n += initial_concentration_array[i]
                c_n_field = Field(data=c_n, name="c_"+sim._components[i], simulation=sim)
                sim.add_field(c_n_field)
        
        
            