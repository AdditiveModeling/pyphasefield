import numpy as np
import sympy as sp

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

def load_tdb(tdb_path):
    """
    loads the TDB file at the specified path, and updates global variables accordingly
    """
    global tdb
    global phases
    global components
    if not os.path.isfile(root_folder+"/TDB/"+tdb_path):
        print("utils.load_tdb Error: TDB file does not exist!")
        return False
    tdb = pyc.Database(root_folder + '/TDB/' + tdb_path)
    
    #update phases
    # will automatically update "phases" in multiphase model. For now, phases is hardcoded
    
    #update components
    components = list(tdb.elements)
    components.sort()
    return True

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

def grad(phi, dx, dim):
    r = []
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        phip = np.roll(phi, -1, i)
        r.append((phip-phim)/(2*dx))
    return r

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

def partial_l(phi, dx, i):
    phim = np.roll(phi, 1, i)
    return (phi-phim)/dx

def partial_r(phi, dx, i):
    phip = np.roll(phi, -1, i)
    return (phip-phi)/dx

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

def loadArrays_nc(data_path, timestep):
    _q1 = np.load(root_folder+"/data/"+data_path+'/q1_'+str(timestep)+'.npy')
    _q4 = np.load(root_folder+"/data/"+data_path+'/q4_'+str(timestep)+'.npy')
    _c = []
    for i in range(len(components)-1):
        _c.append(np.load(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(timestep)+'.npy'))
    _phi = np.load(root_folder+"/data/"+data_path+'/phi_'+str(timestep)+'.npy')
    return timestep, _phi, _c, _q1, _q4

def saveArrays_nc(data_path, timestep, phi, c, q1, q4):
    np.save(root_folder+"/data/"+data_path+'/phi_'+str(timestep), phi)
    for i in range(len(c)):
        np.save(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(timestep), c[i])
    np.save(root_folder+"/data/"+data_path+'/q1_'+str(timestep), q1)
    np.save(root_folder+"/data/"+data_path+'/q4_'+str(timestep), q4)

def applyBCs_nc(phi, c, q1, q4, nbc):
    if(nbc[0]):
        for i in range(len(c)):
            c[i][:,0] = c[i][:,1]
            c[i][:,-1] = c[i][:,-2]
        phi[:,0] = phi[:,1]
        phi[:,-1] = phi[:,-2]
        q1[:,0] = q1[:,1]
        q1[:,-1] = q1[:,-2]
        q4[:,0] = q4[:,1]
        q4[:,-1] = q4[:,-2]
    if(nbc[1]):
        for i in range(len(c)):
            c[i][0,:] = c[i][1,:]
            c[i][-1,:] = c[i][-2,:]
        phi[0,:] = phi[1,:]
        phi[-1,:] = phi[-2,:]
        q1[0,:] = q1[1,:]
        q1[-1,:] = q1[-2,:]
        q4[0,:] = q4[1,:]
        q4[-1,:] = q4[-2,:]

def coreSection(array, nbc):
    """
    Returns only the region of interest for plotting. 
    Removes the buffer cells used for Neumann Boundary Conditions
    """
    returnArray = array
    if(nbc[0]):
        returnArray = returnArray[:, 1:-1]
    if(nbc[1]):
        returnArray = returnArray[1:-1, :]
    return returnArray

def plotImages_nc(phi, c, q4, nbc, data_path, step):
    """
    Plots the phi (order), c (composition), and q4 (orientation component) fields for a given step
    Saves images to the defined path
    """
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('rgb', colors)
    colors2 = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]
    cm2 = LinearSegmentedColormap.from_list('rgb', colors2)

    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = 4, 4
    plt.title('phi')
    cax = plt.imshow(coreSection(phi, nbc), cmap=cm2)
    cbar = fig.colorbar(cax, ticks=[np.min(phi), np.max(phi)])
    plt.savefig(root_folder+"/data/"+data_path+'/phi_'+str(step)+'.png')
    for i in range(len(c)):
        fig, ax = plt.subplots()
        plt.title('c_'+components[i])
        cax = plt.imshow(coreSection(c[i], nbc), cmap=cm)
        cbar = fig.colorbar(cax, ticks=[np.min(c[i]), np.max(c[i])])
        plt.savefig(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(step)+'.png')
    c_N = 1-np.sum(c, axis=0)
    fig, ax = plt.subplots()
    plt.title('c_'+components[len(c)])
    cax = plt.imshow(coreSection(c_N, nbc), cmap=cm)
    cbar = fig.colorbar(cax, ticks=[np.min(c_N), np.max(c_N)])
    plt.savefig(root_folder+"/data/"+data_path+'/c'+str(len(c)+1)+'_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('q4')
    cax = plt.imshow(coreSection(q4, nbc), cmap=cm2)
    cbar = fig.colorbar(cax, ticks=[np.min(q4), np.max(q4)])
    plt.savefig(root_folder+"/data/"+data_path+'/q4_'+str(step)+'.png')
    
def npvalue(var, string, tdb):
    """
    Returns a numpy float from the sympy expression gotten from pycalphad
    Reason: some numpy functions (i.e. sqrt) are incompatible with sympy floats!
    """
    return sp.lambdify(var, tdb.symbols[string], 'numpy')(1000)

def NComponent(sim):
    #load fields to easier-to-reference variables
    phi = sim.fields[0]
    q1 = sim.fields[1]
    q4 = sim.fields[2]
    c = []
    for i in range(3, len(sim.fields)):
        c.append(sim.fields[i])
    T = sim.temperature
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

    #"clip" the grid: if values are smaller than "smallest", set them equal to "smallest"
    #also clip the mgphi values to avoid divide by zero errors!
    smallest = 1.5
    for j in range(dim):
        gqsl[j] = np.clip(gqsl[j], smallest, np.inf)
        gqsr[j] = np.clip(gqsr[j], smallest, np.inf)

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
    std_c=np.sqrt(np.absolute(2*R*T/v_m))
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
    deltaphi = M_phi*(sim.ebar*sim.ebar*((1-3*sim.y_e)*divTgradphi + pf_comp_x + pf_comp_y)-30*g*(G_S-G_L)/v_m-well-4*sim.H*T*phi*rgqs_0*1574.)

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

    t1 = sim.eqbar*sim.eqbar*lq1+(gaq1)*1574. + cc_t1_temp
    t4 = sim.eqbar*sim.eqbar*lq4+(gaq4)*1574. + cc_t4_temp
    lmbda = (q1*t1+q4*t4)
    deltaq1 = M_q*(t1-q1*lmbda)
    deltaq4 = M_q*(t4-q4*lmbda)

    #changes in q



    #apply changes
    for j in range(len(c)):
        sim.fields[3+j] += deltac[j]*dt
    sim.fields[0] += deltaphi*dt
    sim.fields[1] += deltaq1*dt
    sim.fields[2] += deltaq4*dt
    if(i%10 == 0):
        q1, q4 = renormalize(q1, q4)

    #This code segment prints the progress after every 5% of the simulation is done (for convenience)
    if(steps > 19):
        if(i%(steps/20) == 0):
            print(str(5*i/(steps/20))+"% done...")

    #This code segment saves the arrays every 500 steps, and adds nuclei
    #note: nuclei are added *before* saving the data, so stray nuclei may be found before evolving the system
    if(step%500 == 0):
        #find the stochastic nucleation critical probabilistic cutoff
        #attn -- Q and T_liq are hard coded parameters for Ni-10%Cu
        Q0=8*10**5 #activation energy of migration
        T_liq=1697 #Temperature of Liquidus (K)
        J0,p11=find_Pn(T_liq, T, Q0, dt) 
        #print(J0)
        phi, q1, q4=add_nuclei(phi, q1, q4, p11, len(phi))
        #saveArrays_nc(data_path, step, phi, c, q1, q4)
        
def make_seed(phi, q1, q4, x, y, angle, seed_radius):
    shape = phi.shape
    x_size = shape[1]
    y_size = shape[0]
    for i in range((int)(y-seed_radius), (int)(y+seed_radius)):
        for j in range((int)(x-seed_radius), (int)(x+seed_radius)):
            if((i-y)*(i-y)+(j-x)*(j-x) < (seed_radius**2)):
                phi[i%y_size][j%x_size] = 1
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
        sim.set_time_step_length(sim.get_cell_spacing()**2/5./sim.D_L/8)
        return True
    except Exception as e:
        print("Could not load every parameter required from the TDB file!")
        print(e)
        return False
        
def init_NComponent(sim, dim=[200,200], sim_type="seed", tdb_path="Ni-Cu_Ideal.tdb", thermal_type="isothermal", 
                           initial_temperature=1574, thermal_gradient=0, cooling_rate=0, thermal_file_path="T.xdmf", 
                           initial_concentration_array=[0.40831]):
    sim.set_dimensions(dim)
    sim.load_tdb(tdb_path)
    sim.set_cell_spacing(0.0000046)
    sim.d = sim.get_cell_spacing()/0.94
    sim.set_engine(NComponent)
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
        seed_angle = 1*np.pi*8
        phi, q1, q4 = make_seed(phi, q1, q4, dim[1]/2, dim[0]/2, seed_angle, 5)
        sim.add_field(phi)
        sim.add_field(q1)
        sim.add_field(q4)
        
        #initialize concentration array(s)
        if(initial_concentration_array == None):
            for i in range(len(sim.components)-1):
                c_n = np.zeros(dim)
                c_n += 1./len(sim.components)
                sim.add_field(c_n)
        else:
            assert((len(initial_concentration_array)+1) == len(sim._components))
            for i in range(len(initial_concentration_array)):
                c_n = np.zeros(dim)
                c_n += initial_concentration_array[i]
                sim.add_field(c_n)
        
        
            