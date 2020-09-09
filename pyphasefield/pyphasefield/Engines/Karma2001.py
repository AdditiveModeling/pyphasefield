import numpy as np
from ..field import Field
from ..utils import COLORMAP_OTHER, COLORMAP_PHASE

def _q(phi):
    return (1.-phi)/2.

def _q2(h, k, phi):
    return (1.-phi)/(1+k-(1-k)*h)

def _h(phi):
    return phi

def _hprime(phi):
    return 1

def _g(phi):
    phi2 = phi*phi
    phi3 = phi2*phi
    phi5 = phi2*phi3
    return phi5/5. - 2.*phi3/3. + phi - 8./15.

def _gprime(phi):
    phi2m1 = phi*phi-1.
    return phi2m1*phi2m1
    
def _f(phi):
    phi2 = phi*phi
    phi4 = phi2*phi2
    return phi4/4. - phi2/2.

def _fprime(phi):
    return phi*phi*phi - phi

def _u(c, c_L, h, k):
    return np.nan_to_num(np.log(2.*c/(c_L*(1+k-(1-k)*h))))

def _eu(c, c_L, h, k):
    return 2.*c/(c_L*(1+k-(1-k)*h))

def _deudphi(c, c_L, h, hprime, k):
    return 2*c*(1-k)*hprime/(c_L*(1+k-(1-k)*h)**2)

def _dcdt(a_t, c, c_L, D, dphidt, dx, eu, k, phi, q, u, W):
    #compute gradients
    grad2_u = grad2(u, dx)
    grad2_phi = grad2(phi, dx)
    grad_phi_x, grad_phi_y = grad(phi, dx)
    mag_grad_phi = np.sqrt(grad_phi_x**2. + grad_phi_y**2.)+0.000000000001
    grad_u_x, grad_u_y = grad(u, dx)
    grad_cq_x, grad_cq_y = grad(c*q, dx)
    grad_eudphidtmgp_x, grad_eudphidtmgp_y = grad(eu*dphidt/mag_grad_phi, dx)
    grad_cq_dot_grad_u = grad_cq_x * grad_u_x + grad_cq_y * grad_u_y
    grad_eudphidtmgp_dot_grad_phi = grad_eudphidtmgp_x * grad_phi_x + grad_eudphidtmgp_y * grad_phi_y
    
    #return dcdt
    return D*c*q*grad2_u + D*grad_cq_dot_grad_u + a_t*W*c_L*(1-k)*(eu*dphidt*grad2_phi/mag_grad_phi + grad_eudphidtmgp_dot_grad_phi)

def _dphidt(D, deudphi, dx, e4, eu, fprime, g, gprime, k, lmbda, phi, tau, W):
    
    #gradients
    grad_phi_x, grad_phi_y = grad(phi, dx)
    
    #functions
    theta = np.arctan2(grad_phi_y, grad_phi_x)
    a_s = 1+e4*np.cos(4*theta)
    taufunc = 1./(tau*a_s*a_s)
    Wfunc = W*a_s
    W2 = Wfunc*Wfunc
    Wprimefunc = -4*W*e4*np.sin(4*theta)
    WWprime = Wfunc * Wprimefunc
    
    #more gradients
    grad2_phi = grad2(phi, dx)
    grad_W2_x, grad_W2_y = grad(W2, dx)
    grad_WWprime_x, grad_WWprime_y = grad(WWprime, dx)
    divW2dphi = W2*grad2_phi + grad_W2_x*grad_phi_x + grad_W2_y*grad_phi_y
    
    
    #return dphidt
    return taufunc*(-fprime - lmbda/(1-k)*(gprime*(eu-1.) + g*deudphi) + divW2dphi - grad_WWprime_x*grad_phi_y + grad_WWprime_y*grad_phi_x)



def grad(phi, dx):
    phim = np.roll(phi, -1, 0)
    phip = np.roll(phi, 1, 0)
    grady = (phip-phim)/(2*dx)
    phim = np.roll(phi, -1, 1)
    phip = np.roll(phi, 1, 1)
    gradx = (phip-phim)/(2*dx)
    return gradx, grady

def grad2(phi, dx):
    r = 0
    for i in range(2):
        phim = np.roll(phi, -1, 0)
        phip = np.roll(phi, 1, 0)
        r += (phim + phip - 2*phi)
    return r/(dx*dx)
                    
def gradx(phi, dx):
    phim = np.roll(phi, -1, 1)
    phip = np.roll(phi, 1, 1)
    return (phip-phim)/(2*dx)
                    
def grady(phi, dx):
    phim = np.roll(phi, -1, 0)
    phip = np.roll(phi, 1, 0)
    return (phip-phim)/(2*dx)

def Karma2001(sim):
    dt = sim.get_time_step_length()
    dx = sim.get_cell_spacing()
    phi = sim.fields[0].data
    c = sim.fields[1].data
    
    #constants
    k = sim.k
    D = sim.D
    a_t = sim.a_t
    e4 = sim.e4
    a2 = sim.a2
    W = sim.W
    tau = sim.tau
    c_L = sim.c_L
    lmbda = D*tau/(a2*W*W)
    
    #interpolating functions
    h = _h(phi)
    q = _q2(h, k, phi)
    g = _g(phi)
    hprime = _hprime(phi)
    gprime = _gprime(phi)
    fprime = _fprime(phi)
    
    #other quantities
    u = _u(c, c_L, h, k)
    eu = _eu(c, c_L, h, k)
    deudphi = _deudphi(c, c_L, h, hprime, k)
    
    #rate equations
    dphidt = _dphidt(D, deudphi, dx, e4, eu, fprime, g, gprime, k, lmbda, phi, tau, W)
    dcdt = _dcdt(a_t, c, c_L, D, dphidt, dx, eu, k, phi, q, u, W)
    
    #apply rate equations
    sim.fields[0].data += dphidt*dt
    sim.fields[1].data += dcdt*dt
    
def init_Karma2001(sim, dim, radius=22):
    sim.set_dimensions(dim)
    
    sim.k = 0.15
    sim.W = 1.
    sim.tau = 1.
    sim.D = 1.
    sim.a_t = 1./(2.*np.sqrt(2.))
    sim.a1 = 0.8839
    sim.a2 = 0.6267
    sim.e4 = 0.02
    sim.c_L = 0.5
    lmbda = sim.D*sim.tau/(sim.a2*sim.W*sim.W)
    d0 = sim.a1*sim.W/lmbda
    
    phi = np.zeros(dim)
    phi -= 1.
    len_x = dim[1]
    len_y = dim[0]
    for i in range(-radius-1, radius+1):
        for j in range(-radius-1, radius+1):
            if(i*i+j*j < radius*radius*d0*d0):
                phi[i+len_y//2][j+len_x//2] = 1.
    phi_field = Field(data=phi, name="phi", simulation=sim, colormap=COLORMAP_PHASE)
    sim.add_field(phi_field)
    
    
    
    c = np.zeros(dim)
    c += sim.c_L
    for i in range(len_y):
        for j in range(len_x):
            if(phi[i][j] > 0):
                c[i][j] = sim.k*sim.c_L
    c_field = Field(data=c, name="c", simulation=sim, colormap=COLORMAP_OTHER)
    sim.add_field(c_field)
    
    sim.set_cell_spacing(0.4)
    sim.set_time_step_length(0.008)
    sim.set_engine(Karma2001)