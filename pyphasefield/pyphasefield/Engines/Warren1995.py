import numpy as np
import dataclasses
import collections
from pyphasefield.io import plot_field, save_fields

def _p(phi):
    return phi*phi*phi*(10-15*phi+6*phi*phi)

def _g(phi):
    return (phi*phi*(1-phi)*(1-phi))

def _gprime(phi):
    return (4*phi*phi*phi - 6*phi*phi +2*phi)


def gradx(phi, dx):
    phim = np.roll(phi, -1, 0)
    phip = np.roll(phi, 1, 0)
    return (phip-phim)/(2*dx)


def grady(phi, dx):
    phim = np.roll(phi, -1, 1)
    phip = np.roll(phi, 1, 1)
    return (phip-phim)/(2*dx)


def gradxx(phi, dx):
    phim = np.roll(phi, -1, 0)
    phip = np.roll(phi, 1, 0)
    return (phip + phim - 2 * phi) / (dx * dx)


def gradyy(phi, dx):
    phim = np.roll(phi, -1, 1)
    phip = np.roll(phi, 1, 1)
    return (phip + phim - 2 * phi) / (dx * dx)


# These next four declarations will be moved somewhere else -MP:
Wfields = collections.namedtuple('Wfields', ['phi', 'c'])


def warren_diamond(dim, diamond_size):
    phi = np.full(dim, 1.)
    for i in range(diamond_size):
        phi[int(dim[0] / 2 - i):int(dim[0] / 2 + i),
        (int(dim[1] / 2 - (diamond_size - i))):int(dim[1] / 2 + (diamond_size - i))] = 0
    c = np.full(dim, 0.40831)
    return Wfields(phi, c)


@dataclasses.dataclass(frozen=True)
class Component:
    """A data class that holds physical properties for a component"""
    T_m: float  # Melting temperature (K)
    L: float  # Latent heat (J/m^3)
    S: float  # Surface energy (J/m^2)
    B: float  # Linear kinetic coefficient (m/K*s)

    def W(self, d):
        return 3 * self.S / 2 ** 0.5 * self.T_m * d

    def M(self, d):
        return self.T_m * self.T_m * self.B / (6 * 2 ** 0.5 * self.L * d)


def warren_eng(comp1, comp2, D_s, D_l, v_m, y_e, T, alpha, cell_spacing):
    # original Warren1995 model uses centimeters, values have been converted to meters!
    # Initialize fields with seeds
    R = 8.314  # gas constant, J/mol*K
    dt = cell_spacing ** 2 / 5. / D_l
    d = cell_spacing / 0.94  # Interfacial thickness
    ebar = np.sqrt(6 * np.sqrt(2) * comp1.S * d / comp1.T_m)  # Baseline energy

    def warren1995(phi, c):
        dx = cell_spacing

        g = _g(phi)
        p = _p(phi)
        gprime = _gprime(phi)
        H_A = comp1.W(d) * gprime + 30 * comp1.L * (1 / (T - 1) / comp1.T_m) * g
        H_B = comp2.W(d) * gprime + 30 * comp2.L * (1 / (T - 1) / comp2.T_m) * g
        phixx = gradxx(phi, dx)
        phiyy = gradyy(phi, dx)
        lphi = phixx + phiyy
        phix = gradx(phi, dx)
        phiy = grady(phi, dx)
        phixy = grady(phix, dx)

        # Change in c field
        D_C = D_s + p * (D_l - D_s)
        temp = D_C * v_m * c * (1 - c) * (H_B - H_A) / R
        deltac = D_C * (gradxx(c, dx) + gradyy(c, dx)) + (
                gradx(D_C, dx) * gradx(c, dx) + grady(D_C, dx) * grady(c, dx)) + temp * (lphi) + (
                         gradx(temp, dx) * phix + grady(temp, dx) * phiy)

        # Change in phi field
        theta = np.arctan2(phiy, phix)
        eta = 1 + y_e * np.cos(4 * theta)
        etap = -4 * y_e * np.sin(4 * theta)
        etapp = -16 * (eta - 1)
        c2 = np.cos(2 * theta)
        s2 = np.sin(2 * theta)
        M_phi = (1 - c) * comp1.M(d) + c * comp2.M(d)
        ebar2 = ebar ** 2
        deltaphi = M_phi * ((ebar2 * eta * eta * lphi - (1 - c) * H_A - c * H_B) + ebar2 * eta * etap * (
                s2 * (phiyy - phixx) + 2 * c2 * phixy) + 0.5 * ebar2 * (etap * etap + eta * etapp) * (
                                    -2 * s2 * phixy + lphi + c2 * (phiyy - phixx)))
        randArray = 2 * np.random.random(phi.shape) - 1
        deltaphi += M_phi * alpha * randArray * (16 * g) * ((1 - c) * H_A + c * H_B)

        # Apply changes to inputted fields
        phi += deltaphi * dt
        c += deltac * dt
        return phi, c

    return warren1995


def run_warren(phi, c, engine, steps, output=False):
    for step in range(steps):
        if output and not step % output:
            save_fields({'phi': phi, 'c': c}, step)
            plot_field(phi, "phi", step, 1)
        engine(phi, c)
    return 0
