def __h(phi):
    return phi*phi*phi*(10-15*phi+6*phi*phi)

def __hprime(phi):
    return (30*phi*phi*(1-phi)*(1-phi))

def __g(phi):
    return (16*phi*phi*(1-phi)*(1-phi))

def __gprime(phi):
    return (64*phi*phi*phi - 96*phi*phi +32*phi)

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

def grad_l(phi, dx, i):
    r = []
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        r.append((phi-phim)/(dx))
    return r

def grad_r(phi, dx, i):
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

def renormalize(q1, q2, q3, q4):
    q = np.sqrt(q1*q1+q2*q2+q3*q3+q4*q4)
    return q1/q, q2/q, q3/q, q4/q

def Dorr2010(sim):
    
    
def init_Dorr2010(sim):
    sim.T = 873.

    #bcc = 0L = e, fcc = 1S = d
    #material parameters, pJ, um, K, s (except for R and Q terms, which use joules)
    sim.R = 8.314 #gas constant, J/mol*K
    sim.Q_d = 156400. #activation term in diffusivity in fcc region, J/mol
    sim.Q_e = 55290. #activation term in diffusivity in bcc region, J/mol
    sim.ceq_d = 0.1 #equilibrium concentration in fcc
    sim.ceq_e = 0.05 #equilibrium concentration in bcc
    sim.D_d0 = 1.3e8 #scaling constant of diffusion in fcc, um^2/s
    sim.D_e0 = 5.6e4 #scaling constant of diffusion in fcc, um^2/s
    sim.M_phi = 200 #mobility of phase, 1/(s*pJ)
    sim.M_qmax = 200 #maximum mobility of orientation, 1/(s*pJ)
    sim.H = 1e-3 #interfacial energy term for quaternions, pJ/(K*um)
    sim.A = 666. #energetic term in HBSM binary alloy model, A_d = A_e
    sim.w = 0.4125 #scaling of energetic hump forcing phase to be 0,1

    #Temperature dependent params, since T is constant, we compute them here for now
    sim.D_d = sim.D_d0 * np.exp(-sim.Q_d/(sim.R*sim.T))
    sim.D_e = sim.D_e0 * np.exp(-sim.Q_e/(sim.R*sim.T))
    
    #simulation region and interfacial params
    sim.dx = 0.05
    #dt = dx*dx/5./D_e #old value for Courant stability, not sufficient for quaternion stability. Use beta = 1.5ish
    sim.dt = 1.48e-7 #10 million time steps to run Dorr's simulation
    sim.d = dx*4 #interfacial thickness
    sim.ebar = 0.165 
    sim.eqbar = 0.1