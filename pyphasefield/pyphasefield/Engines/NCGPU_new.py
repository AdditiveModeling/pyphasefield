import numpy as np
import sympy as sp
import symengine as se
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import math
from pathlib import Path
import time
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE, make_seed

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

@cuda.jit(device=True)
def divagradb_3D(a, axp, axm, ayp, aym, azp, azm, b, bxp, bxm, byp, bym, bzp, bzm, idx): 
    return 0.5*idx*idx*((axp+a)*(bxp-b) - (a+axm)*(b-bxm) + (ayp+a)*(byp-b) - (a+aym)*(b-bym) + (azp+a)*(bzp-b) - (a+azm)*(b-bzm))

@cuda.jit(device=True)
def f_ori_term_3D(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, D_q_zp, D_q_zm, mgq_xp, mgq_xm, mgq_yp, mgq_ym, mgq_zp, mgq_zm, q, q_xp, q_xm, q_yp, q_ym, q_zp, q_zm, idx):
    term = (D_q+D_q_xp)*(q_xp-q)/mgq_xp - (D_q+D_q_xm)*(q-q_xm)/mgq_xm
    term += ((D_q+D_q_yp)*(q_yp-q)/mgq_yp - (D_q+D_q_ym)*(q-q_ym)/mgq_ym)
    term += ((D_q+D_q_zp)*(q_zp-q)/mgq_zp - (D_q+D_q_zm)*(q-q_zm)/mgq_zm)
    return 0.5*idx*idx*term

@cuda.jit
def NComponent_kernel_2D(fields, T, transfer, fields_out, rng_states, params, c_params):
    
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
    
@cuda.jit
def NComponent_sp_kernel_2D(fields, T, spa_gpu, save_points, timestep):
    
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    threadId = startx + starty*stridex
    
    
    for i in range(threadId, len(save_points[0]), stridex*stridey):
        for j in range(len(fields)):
            spa_gpu[i][j][timestep] = fields[j][save_points[1][i]][save_points[0][i]]
        spa_gpu[i][len(fields)][timestep] = T[save_points[1][i]][save_points[0][i]]
        
                
@numba.jit
def get_thermodynamics(ufunc, array):
    return ufunc(array)

@cuda.jit
def NComponent_noise_kernel_2D(fields, T, transfer, rng_states, ufunc_array, params, c_params):
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
def NComponent_helper_kernel_2D(fields, T, transfer, rng_states, ufunc_array, params, c_params):
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
            
@cuda.jit
def NComponent_helper_kernel_3D(fields, T, transfer, rng_states, ufunc_array, params, c_params):
    #initializes certain arrays that are used in div-grad terms, to avoid recomputing terms 4 or 6 times
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)
    threadId = startx + starty*stridex + startz*stridex*stridey
    
    v_m = params[2]
    D_L = params[7]
    D_S = params[8]
    noise_amp_c = params[13]
    thermo_finite_diff_incr = params[15]
    tfdi_inv = 1./thermo_finite_diff_incr
    W = c_params[4]
    
    phi = fields[0]
    q1 = fields[1]
    q2 = fields[2]
    q3 = fields[3]
    q4 = fields[4]
    #c is fields 5 to k, where k equals the number of components +3. 
    #For N components this is N-1 fields, the last one being implicitly defined
    
    phi_out = fields[0]
    q1_out = fields[1]
    q2_out = fields[2]
    q3_out = fields[3]
    q4_out = fields[4]
    
    G_L = transfer[0]
    G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-4 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-3 to 2*len(fields)-9 (for 2 components, eg Ni and Cu, dFdc is just 3
    
    for i in range(startz, phi.shape[0], stridez):
        for j in range(starty, phi.shape[1], stridey):
            for k in range(startx, phi.shape[2], stridex):
                c_N = 1.
                for l in range(5, len(fields)):
                    ufunc_array[i][j][k][l-5] = fields[l][i][j][k]
                    c_N -= fields[l][i][j][k]
                comps = len(fields)-4
                ufunc_array[i][j][k][len(fields)-5] = c_N
                ufunc_array[i][j][k][len(fields)-4] = T[i][j][k]
                #dGdc = numba.cuda.local.array((2,1), numba.float64)
                #NEEDS FIXING vvvvvvvvvv
                G_L[i][j][k] = get_thermodynamics(ufunc_g_l, ufunc_array[i][j][k])
                G_S[i][j][k] = get_thermodynamics(ufunc_g_s, ufunc_array[i][j][k])

                g = (phi[i][j][k]**2)*(1-phi[i][j][k])**2
                h = (phi[i][j][k]**3)*(6.*phi[i][j][k]**2 - 15.*phi[i][j][k] + 10.)

                ufunc_array[i][j][k][len(fields)-5] -= thermo_finite_diff_incr
                for l in range(5, len(fields)):
                    ufunc_array[i][j][k][l-5] += thermo_finite_diff_incr
                    dGLdc = tfdi_inv*(get_thermodynamics(ufunc_g_l, ufunc_array[i][j][k])-G_L[i][j][k])
                    dGSdc = tfdi_inv*(get_thermodynamics(ufunc_g_s, ufunc_array[i][j][k])-G_S[i][j][k])
                    M_c = transfer[l-3]
                    dFdc = transfer[l-3+len(fields)-5]
                    M_c[i][j][k] = v_m*fields[l][i][j][k]*(D_L + h*(D_S - D_L))/(8.314*T[i][j][k])
                    dFdc[i][j][k] = (dGLdc + h*(dGSdc-dGLdc))/v_m + (W[l-5]-W[len(fields)-5])*g*T[i][j][k]
                    ufunc_array[i][j][k][l-5] -= thermo_finite_diff_incr
                ufunc_array[i][j][k][len(fields)-5] += thermo_finite_diff_incr
                
@cuda.jit
def NComponent_sp_kernel_3D(fields, T, spa_gpu, save_points, timestep):
    
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)
    threadId = startx + starty*stridex + startz*stridex*stridey
    
    
    for i in range(threadId, len(save_points[0]), stridex*stridey*stridez):
        for j in range(len(fields)):
            spa_gpu[i][j][timestep] = fields[j][save_points[2][i]][save_points[1][i]][save_points[0][i]]
        spa_gpu[i][len(fields)][timestep] = T[save_points[2][i]][save_points[1][i]][save_points[0][i]]
                
@cuda.jit
def NComponent_noise_kernel_3D(fields, T, transfer, rng_states, ufunc_array, params, c_params):
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)
    threadId = startx + starty*stridex + startz*stridex*stridey
    
    v_m = params[2]
    D_L = params[7]
    D_S = params[8]
    noise_amp_c = params[13]
    W = c_params[4]
    
    #c is fields 3 to k, where k equals the number of components +1. 
    #For N components this is N-1 fields, the last one being implicitly defined
    
    phi = fields[0]
    
    G_L = transfer[0]
    G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
    
    #possibly check if boundary conditions are important here
    for i in range(startz, phi.shape[0], stridez):
        for j in range(starty, phi.shape[1], stridey):
            for k in range(startx, phi.shape[2], stridex):
                for l in range(5, len(fields)):
                    dFdc = transfer[l-3+len(fields)-5]
                    noise_c = noise_amp_c*math.sqrt(2.*8.314*T[i][j][k]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
                    dFdc[i][j][k] += noise_c
            
@cuda.jit
def NComponent_kernel_3D(fields, T, transfer, fields_out, rng_states, params, c_params):
    
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)
    threadId = startx + starty*stridex + startz*stridex*stridey
    
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
    q2 = fields[2]
    q3 = fields[3]
    q4 = fields[4]
    #c is fields 5 to k, where k equals the number of components +3. 
    #For N components this is N-1 fields, the last one being implicitly defined
    
    phi_out = fields_out[0]
    q1_out = fields_out[1]
    q2_out = fields_out[2]
    q3_out = fields_out[3]
    q4_out = fields_out[4]
    
    G_L = transfer[0]
    G_S = transfer[1]
    #M_c is transfer 2 to len(fields)-2 (for 2 components, eg Ni and Cu, M_c is just 2
    #dFdc is transfer len(fields)-1 to 2*len(fields)-5 (for 2 components, eg Ni and Cu, dFdc is just 3
        
    ebar2 = ebar*ebar
    eqbar2 = eqbar*eqbar
    for k in range(startx+1, phi.shape[2]-1, stridex):
        for j in range(starty+1, phi.shape[1]-1, stridey):
            for i in range(startz+1, phi.shape[0]-1, stridez):

                #interpolating functions
                g = (phi[i][j][k]**2)*(1-phi[i][j][k])**2
                h = (phi[i][j][k]**3)*(6.*phi[i][j][k]**2 - 15.*phi[i][j][k] + 10.)
                hprime = 30.*g
                gprime = 4.*phi[i][j][k]**3 - 6.*phi[i][j][k]**2 + 2*phi[i][j][k]

                #gradients
                idx = 1./dx
                dphidx = 0.5*(phi[i][j][k+1]-phi[i][j][k-1])*idx
                dphidy = 0.5*(phi[i][j+1][k]-phi[i][j-1][k])*idx
                dphidz = 0.5*(phi[i+1][j][k]-phi[i-1][j][k])*idx
                dTdx = 0.5*idx*(T[i][j][k+1]-T[i][j][k-1])
                dTdy = 0.5*idx*(T[i][j+1][k]-T[i][j-1][k])
                dTdz = 0.5*idx*(T[i+1][j][k]-T[i-1][j][k])
                d2phidx2 = (phi[i][j][k+1]+phi[i][j][k-1]-2.*phi[i][j][k])*idx*idx
                d2phidy2 = (phi[i][j+1][k]+phi[i][j-1][k]-2.*phi[i][j][k])*idx*idx
                d2phidz2 = (phi[i+1][j][k]+phi[i-1][j][k]-2.*phi[i][j][k])*idx*idx
                lphi = d2phidx2 + d2phidy2 + d2phidz2
                d2phidxy = 0.25*(phi[i][j+1][k+1]-phi[i][j+1][k-1]-phi[i][j-1][k+1]+phi[i][j-1][k-1])*idx*idx
                d2phidxz = 0.25*(phi[i+1][j][k+1]-phi[i+1][j][k-1]-phi[i-1][j][k+1]+phi[i-1][j][k-1])*idx*idx
                d2phidyz = 0.25*(phi[i+1][j+1][k]-phi[i+1][j-1][k]-phi[i-1][j+1][k]+phi[i-1][j-1][k])*idx*idx
                mag_grad_phi2 = dphidx**2 + dphidy**2 + dphidz**2
                if(mag_grad_phi2 < 1e-6):
                    mag_grad_phi2 = 1e-6
                mag_grad_phi4 = mag_grad_phi2**2
                mag_grad_phi6 = mag_grad_phi4*mag_grad_phi2

                dq1dxp = idx*(q1[i][j][k+1]-q1[i][j][k])
                dq1dxm = idx*(q1[i][j][k]-q1[i][j][k-1])
                dq1dx = 0.5*(dq1dxp+dq1dxm)
                dq1dyp = idx*(q1[i][j+1][k]-q1[i][j][k])
                dq1dym = idx*(q1[i][j][k]-q1[i][j-1][k])
                dq1dy = 0.5*(dq1dyp+dq1dym)
                dq1dzp = idx*(q1[i+1][j][k]-q1[i][j][k])
                dq1dzm = idx*(q1[i][j][k]-q1[i-1][j][k])
                dq1dz = 0.5*(dq1dzp+dq1dzm)
                dq2dxp = idx*(q2[i][j][k+1]-q2[i][j][k])
                dq2dxm = idx*(q2[i][j][k]-q2[i][j][k-1])
                dq2dx = 0.5*(dq2dxp+dq2dxm)
                dq2dyp = idx*(q2[i][j+1][k]-q2[i][j][k])
                dq2dym = idx*(q2[i][j][k]-q2[i][j-1][k])
                dq2dy = 0.5*(dq2dyp+dq2dym)
                dq2dzp = idx*(q2[i+1][j][k]-q2[i][j][k])
                dq2dzm = idx*(q2[i][j][k]-q2[i-1][j][k])
                dq2dz = 0.5*(dq2dzp+dq2dzm)
                dq3dxp = idx*(q3[i][j][k+1]-q3[i][j][k])
                dq3dxm = idx*(q3[i][j][k]-q3[i][j][k-1])
                dq3dx = 0.5*(dq3dxp+dq3dxm)
                dq3dyp = idx*(q3[i][j+1][k]-q3[i][j][k])
                dq3dym = idx*(q3[i][j][k]-q3[i][j-1][k])
                dq3dy = 0.5*(dq3dyp+dq3dym)
                dq3dzp = idx*(q3[i+1][j][k]-q3[i][j][k])
                dq3dzm = idx*(q3[i][j][k]-q3[i-1][j][k])
                dq3dz = 0.5*(dq3dzp+dq3dzm)
                dq4dxp = idx*(q4[i][j][k+1]-q4[i][j][k])
                dq4dxm = idx*(q4[i][j][k]-q4[i][j][k-1])
                dq4dx = 0.5*(dq4dxp+dq4dxm)
                dq4dyp = idx*(q4[i][j+1][k]-q4[i][j][k])
                dq4dym = idx*(q4[i][j][k]-q4[i][j-1][k])
                dq4dy = 0.5*(dq4dyp+dq4dym)
                dq4dzp = idx*(q4[i+1][j][k]-q4[i][j][k])
                dq4dzm = idx*(q4[i][j][k]-q4[i-1][j][k])
                dq4dz = 0.5*(dq4dzp+dq4dzm)
                mgq_xp = math.sqrt(dq1dxp**2 + dq2dxp**2 + dq3dxp**2 + dq4dxp**2)
                mgq_xm = math.sqrt(dq1dxm**2 + dq2dxm**2 + dq3dxm**2 + dq4dxm**2)
                mgq_yp = math.sqrt(dq1dyp**2 + dq2dyp**2 + dq3dyp**2 + dq4dyp**2)
                mgq_ym = math.sqrt(dq1dym**2 + dq2dym**2 + dq3dym**2 + dq4dym**2)
                mgq_zp = math.sqrt(dq1dzp**2 + dq2dzp**2 + dq3dzp**2 + dq4dzp**2)
                mgq_zm = math.sqrt(dq1dzm**2 + dq2dzm**2 + dq3dzm**2 + dq4dzm**2)
                mag_grad_q = 0.5*(mgq_xp+mgq_xm+mgq_yp+mgq_ym+mgq_zp+mgq_zm)
                if(mgq_xp < beta):
                    mgq_xp = beta
                if(mgq_xm < beta):
                    mgq_xm = beta
                if(mgq_yp < beta):
                    mgq_yp = beta
                if(mgq_ym < beta):
                    mgq_ym = beta
                if(mgq_zp < beta):
                    mgq_zp = beta
                if(mgq_zm < beta):
                    mgq_zm = beta

                #psi terms
                q22 = q1[i][j][k]**2 + q2[i][j][k]**2 - q3[i][j][k]**2 - q4[i][j][k]**2
                q33 = q1[i][j][k]**2 - q2[i][j][k]**2 + q3[i][j][k]**2 - q4[i][j][k]**2
                q44 = q1[i][j][k]**2 - q2[i][j][k]**2 - q3[i][j][k]**2 + q4[i][j][k]**2

                q22x = 2*q1[i][j][k]*dq1dx + 2*q2[i][j][k]*dq2dx - 2*q3[i][j][k]*dq3dx - 2*q4[i][j][k]*dq4dx
                q33x = 2*q1[i][j][k]*dq1dx - 2*q2[i][j][k]*dq2dx + 2*q3[i][j][k]*dq3dx - 2*q4[i][j][k]*dq4dx
                q44x = 2*q1[i][j][k]*dq1dx - 2*q2[i][j][k]*dq2dx - 2*q3[i][j][k]*dq3dx + 2*q4[i][j][k]*dq4dx
                q22y = 2*q1[i][j][k]*dq1dy + 2*q2[i][j][k]*dq2dy - 2*q3[i][j][k]*dq3dy - 2*q4[i][j][k]*dq4dy
                q33y = 2*q1[i][j][k]*dq1dy - 2*q2[i][j][k]*dq2dy + 2*q3[i][j][k]*dq3dy - 2*q4[i][j][k]*dq4dy
                q44y = 2*q1[i][j][k]*dq1dy - 2*q2[i][j][k]*dq2dy - 2*q3[i][j][k]*dq3dy + 2*q4[i][j][k]*dq4dy
                q22z = 2*q1[i][j][k]*dq1dz + 2*q2[i][j][k]*dq2dz - 2*q3[i][j][k]*dq3dz - 2*q4[i][j][k]*dq4dz
                q33z = 2*q1[i][j][k]*dq1dz - 2*q2[i][j][k]*dq2dz + 2*q3[i][j][k]*dq3dz - 2*q4[i][j][k]*dq4dz
                q44z = 2*q1[i][j][k]*dq1dz - 2*q2[i][j][k]*dq2dz - 2*q3[i][j][k]*dq3dz + 2*q4[i][j][k]*dq4dz

                q12 = 2*q1[i][j][k]*q2[i][j][k]
                q13 = 2*q1[i][j][k]*q3[i][j][k]
                q14 = 2*q1[i][j][k]*q4[i][j][k]
                q23 = 2*q2[i][j][k]*q3[i][j][k]
                q24 = 2*q2[i][j][k]*q4[i][j][k]
                q34 = 2*q3[i][j][k]*q4[i][j][k]

                q12x = 2*q1[i][j][k]*dq2dx + 2*q2[i][j][k]*dq1dx
                q13x = 2*q1[i][j][k]*dq3dx + 2*q3[i][j][k]*dq1dx
                q14x = 2*q1[i][j][k]*dq4dx + 2*q4[i][j][k]*dq1dx
                q23x = 2*q2[i][j][k]*dq3dx + 2*q3[i][j][k]*dq2dx
                q24x = 2*q2[i][j][k]*dq4dx + 2*q4[i][j][k]*dq2dx
                q34x = 2*q3[i][j][k]*dq4dx + 2*q4[i][j][k]*dq3dx
                q12y = 2*q1[i][j][k]*dq2dy + 2*q2[i][j][k]*dq1dy
                q13y = 2*q1[i][j][k]*dq3dy + 2*q3[i][j][k]*dq1dy
                q14y = 2*q1[i][j][k]*dq4dy + 2*q4[i][j][k]*dq1dy
                q23y = 2*q2[i][j][k]*dq3dy + 2*q3[i][j][k]*dq2dy
                q24y = 2*q2[i][j][k]*dq4dy + 2*q4[i][j][k]*dq2dy
                q34y = 2*q3[i][j][k]*dq4dy + 2*q4[i][j][k]*dq3dy
                q12z = 2*q1[i][j][k]*dq2dz + 2*q2[i][j][k]*dq1dz
                q13z = 2*q1[i][j][k]*dq3dz + 2*q3[i][j][k]*dq1dz
                q14z = 2*q1[i][j][k]*dq4dz + 2*q4[i][j][k]*dq1dz
                q23z = 2*q2[i][j][k]*dq3dz + 2*q3[i][j][k]*dq2dz
                q24z = 2*q2[i][j][k]*dq4dz + 2*q4[i][j][k]*dq2dz
                q34z = 2*q3[i][j][k]*dq4dz + 2*q4[i][j][k]*dq3dz

                x1 = 2*q1[i][j][k]*dphidx
                x2 = 2*q2[i][j][k]*dphidx
                x3 = 2*q3[i][j][k]*dphidx
                x4 = 2*q4[i][j][k]*dphidx
                y1 = 2*q1[i][j][k]*dphidy
                y2 = 2*q2[i][j][k]*dphidy
                y3 = 2*q3[i][j][k]*dphidy
                y4 = 2*q4[i][j][k]*dphidy
                z1 = 2*q1[i][j][k]*dphidz
                z2 = 2*q2[i][j][k]*dphidz
                z3 = 2*q3[i][j][k]*dphidz
                z4 = 2*q4[i][j][k]*dphidz

                psix = q22*dphidx + (q23-q14)*dphidy + (q24+q13)*dphidz
                psiy = (q23+q14)*dphidx + q33*dphidy + (q34-q12)*dphidz
                psiz = (q24-q13)*dphidx + (q34+q12)*dphidy + q44*dphidz

                psix2 = psix**2
                psix3 = psix2*psix
                psix4 = psix2**2
                psiy2 = psiy**2
                psiy3 = psiy2*psiy
                psiy4 = psiy2**2
                psiz2 = psiz**2
                psiz3 = psiz2*psiz
                psiz4 = psiz2**2

                eta = 1. - 3.*y_e + 4.*y_e*(psix4 + psiy4 + psiz4)/mag_grad_phi4

                dpsixdx = q22x*dphidx + q22*d2phidx2 + (q23x-q14x)*dphidy + (q23-q14)*d2phidxy + (q24x+q13x)*dphidz + (q24+q13)*d2phidxz
                dpsixdy = q22y*dphidx + q22*d2phidxy + (q23y-q14y)*dphidy + (q23-q14)*d2phidy2 + (q24y+q13y)*dphidz + (q24+q13)*d2phidyz
                dpsixdz = q22z*dphidx + q22*d2phidxz + (q23z-q14z)*dphidy + (q23-q14)*d2phidyz + (q24z+q13z)*dphidz + (q24+q13)*d2phidz2
                dpsiydx = (q23x+q14x)*dphidx + (q23+q14)*d2phidx2 + q33x*dphidy + q33*d2phidxy + (q34x-q12x)*dphidz + (q34-q12)*d2phidxz
                dpsiydy = (q23y+q14y)*dphidx + (q23+q14)*d2phidxy + q33y*dphidy + q33*d2phidy2 + (q34y-q12y)*dphidz + (q34-q12)*d2phidyz
                dpsiydz = (q23z+q14z)*dphidx + (q23+q14)*d2phidxz + q33z*dphidy + q33*d2phidyz + (q34z-q12z)*dphidz + (q34-q12)*d2phidz2
                dpsizdx = (q24x-q13x)*dphidx + (q24-q13)*d2phidx2 + (q34x+q12x)*dphidy + (q34+q12)*d2phidxy + q44x*dphidz + q44*d2phidxz
                dpsizdy = (q24y-q13y)*dphidx + (q24-q13)*d2phidxy + (q34y+q12y)*dphidy + (q34+q12)*d2phidy2 + q44y*dphidz + q44*d2phidyz
                dpsizdz = (q24z-q13z)*dphidx + (q24-q13)*d2phidxz + (q34z+q12z)*dphidy + (q34+q12)*d2phidyz + q44z*dphidz + q44*d2phidz2

                d_term_dx = dTdx*((2.*psix3*q22 + 2.*psiy3*(q23-q14) + 2.*psiz3*(q24+q13))/mag_grad_phi2 - dphidx*(psix4+psiy4+psiz4)/mag_grad_phi4)
                d_term_dx += T[i][j][k]*(6.*psix2*dpsixdx*q22 + 2.*psix3*q22x + 6.*psiy2*dpsiydx*(q23-q14) + 2.*psiy3*(q23x-q14x) + 6.*psiz2*dpsizdx*(q24+q13) + 2.*psiz3*(q24x+q13x))/mag_grad_phi2
                d_term_dx -= 2.*T[i][j][k]*(2.*psix3*q22 + 2.*psiy3*(q23-q14) + 2.*psiz3*(q24+q13))*(dphidx*d2phidx2 + dphidy*d2phidxy + dphidz*d2phidxz)/mag_grad_phi4
                d_term_dx -= T[i][j][k]*(d2phidx2*(psix4+psiy4+psiz4) + dphidx*(4.*psix3*dpsixdx + 4.*psiy3*dpsiydx + 4.*psiz3*dpsizdx))/mag_grad_phi4
                d_term_dx += 4.*T[i][j][k]*dphidx*(psix4+psiy4+psiz4)*(dphidx*d2phidx2 + dphidy*d2phidxy + dphidz*d2phidxz)/mag_grad_phi6
                d_term_dx *= (4.*y_e*ebar2)

                d_term_dy = dTdy*((2.*psix3*(q23+q14) + 2.*psiy3*q33 + 2.*psiz3*(q34-q12))/mag_grad_phi2 - dphidy*(psix4+psiy4+psiz4)/mag_grad_phi4)
                d_term_dy += T[i][j][k]*(6.*psix2*dpsixdy*(q23+q14) + 2.*psix3*(q23y+q14y) + 6.*psiy2*dpsiydy*q33 + 2.*psiy3*q33y + 6.*psiz2*dpsizdy*(q34-q12) + 2.*psiz3*(q34y-q12y))/mag_grad_phi2
                d_term_dy -= 2.*T[i][j][k]*(2.*psix3*(q23+q14) + 2.*psiy3*q33 + 2.*psiz3*(q34-q12))*(dphidx*d2phidxy + dphidy*d2phidy2 + dphidz*d2phidyz)/mag_grad_phi4
                d_term_dy -= T[i][j][k]*(d2phidy2*(psix4+psiy4+psiz4) + dphidy*(4.*psix3*dpsixdy + 4.*psiy3*dpsiydy + 4.*psiz3*dpsizdy))/mag_grad_phi4
                d_term_dy += 4.*T[i][j][k]*dphidy*(psix4+psiy4+psiz4)*(dphidx*d2phidxy + dphidy*d2phidy2 + dphidz*d2phidyz)/mag_grad_phi6
                d_term_dy *= (4.*y_e*ebar2)

                d_term_dz = dTdz*((2.*psix3*(q24-q13) + 2.*psiy3*(q34+q12) + 2.*psiz3*q44)/mag_grad_phi2 - dphidz*(psix4+psiy4+psiz4)/mag_grad_phi4)
                d_term_dz += T[i][j][k]*(6.*psix2*dpsixdz*(q24-q13) + 2.*psix3*(q24z-q13z) + 6.*psiy2*dpsiydz*(q34+q12) + 2.*psiy3*(q34z+q12z) + 6.*psiz2*dpsizdz*q44 + 2.*psiz3*q44z)/mag_grad_phi2
                d_term_dz -= 2.*T[i][j][k]*(2.*psix3*(q24-q13) + 2.*psiy3*(q34+q12) + 2.*psiz3*q44)*(dphidx*d2phidxz + dphidy*d2phidyz + dphidz*d2phidz2)/mag_grad_phi4
                d_term_dz -= T[i][j][k]*(d2phidz2*(psix4+psiy4+psiz4) + dphidz*(4.*psix3*dpsixdz + 4.*psiy3*dpsiydz + 4.*psiz3*dpsizdz))/mag_grad_phi4
                d_term_dz += 4.*T[i][j][k]*dphidz*(psix4+psiy4+psiz4)*(dphidx*d2phidxz + dphidy*d2phidyz + dphidz*d2phidz2)/mag_grad_phi6
                d_term_dz *= (4.*y_e*ebar2)

                #c_N stuff
                cW = 0.
                c_N = 1.
                M_phi = 0.
                for l in range(5, len(fields)):
                    c_i = fields[l][i][j][k]
                    cW += c_i*W[l-5]
                    c_N -= c_i
                    M_phi += c_i*M[l-5]
                cW += c_N*W[len(fields)-5]
                M_phi += c_N*M[len(fields)-5]

                #mobilities
                M_q = M_qmax + (1e-6-M_qmax)*h
                M_phi *= eta

                #dphidt
                dphidt = ebar2*(1.-3.*y_e)*(T[i][j][k]*lphi + dTdx*dphidx + dTdy*dphidy + dTdz*dphidz)
                dphidt += d_term_dx + d_term_dy + d_term_dz
                dphidt -= hprime*(G_S[i][j][k] - G_L[i][j][k])/v_m
                dphidt -= gprime*T[i][j][k]*cW
                dphidt -= 4.*H*T[i][j][k]*phi[i][j][k]*mag_grad_q
                dphidt *= M_phi

                #noise in phi
                noise_phi = math.sqrt(2.*8.314*T[i][j][k]*M_phi/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
                dphidt += noise_phi*noise_amp_phi

                #dcidt
                for l in range(5, len(fields)):
                    c_i_out = fields_out[l]
                    M_c = transfer[l-3]
                    dFdci = transfer[l-3+len(fields)-5]
                    c_i_out[i][j][k] = (divagradb_3D(M_c[i][j][k], M_c[i][j][k+1], M_c[i][j][k-1], M_c[i][j+1][k], M_c[i][j-1][k], 
                                               M_c[i+1][j][k], M_c[i-1][j][k], dFdci[i][j][k], dFdci[i][j][k+1], dFdci[i][j][k-1], 
                                               dFdci[i][j+1][k], dFdci[i][j-1][k], dFdci[i+1][j][k], dFdci[i-1][j][k], idx))
                    for m in range(5, len(fields)):
                        c_j = fields[m]
                        dFdcj = transfer[m-3+len(fields)-5]
                        c_i_out[i][j][k] -= divagradb_3D(M_c[i][j][k]*c_j[i][j][k], M_c[i][j][k+1]*c_j[i][j][k+1], 
                                                   M_c[i][j][k-1]*c_j[i][j][k-1], M_c[i][j+1][k]*c_j[i][j+1][k], 
                                                   M_c[i][j-1][k]*c_j[i][j-1][k], M_c[i+1][j][k]*c_j[i+1][j][k], 
                                                   M_c[i-1][j][k]*c_j[i-1][j][k], dFdcj[i][j][k], dFdcj[i][j][k+1], dFdcj[i][j][k-1], 
                                                   dFdcj[i][j+1][k], dFdcj[i][j-1][k], dFdcj[i+1][j][k], dFdcj[i-1][j][k], idx)

                #dqdt
                D_q = 2.*H*T[i][j][k]*(phi[i][j][k]**2)
                D_q_xp = 2.*H*T[i][j][k+1]*(phi[i][j][k+1]**2)
                D_q_xm = 2.*H*T[i][j][k-1]*(phi[i][j][k-1]**2)
                D_q_yp = 2.*H*T[i][j+1][k]*(phi[i][j+1][k]**2)
                D_q_ym = 2.*H*T[i][j-1][k]*(phi[i][j-1][k]**2)
                D_q_zp = 2.*H*T[i+1][j][k]*(phi[i+1][j][k]**2)
                D_q_zm = 2.*H*T[i-1][j][k]*(phi[i-1][j][k]**2)

                #also needs fixing for 3D
                f_ori_1 = f_ori_term_3D(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, D_q_zp, D_q_zm, mgq_xp, mgq_xm, mgq_yp, mgq_ym, mgq_zp, mgq_zm,
                                     q1[i][j][k], q1[i][j][k+1], q1[i][j][k-1], q1[i][j+1][k], q1[i][j-1][k], q1[i+1][j][k], 
                                     q1[i-1][j][k], idx)
                f_ori_2 = f_ori_term_3D(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, D_q_zp, D_q_zm, mgq_xp, mgq_xm, mgq_yp, mgq_ym, mgq_zp, mgq_zm,
                                     q2[i][j][k], q2[i][j][k+1], q2[i][j][k-1], q2[i][j+1][k], q2[i][j-1][k], q2[i+1][j][k], 
                                     q2[i-1][j][k], idx)
                f_ori_3 = f_ori_term_3D(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, D_q_zp, D_q_zm, mgq_xp, mgq_xm, mgq_yp, mgq_ym, mgq_zp, mgq_zm,
                                     q3[i][j][k], q3[i][j][k+1], q3[i][j][k-1], q3[i][j+1][k], q3[i][j-1][k], q3[i+1][j][k], 
                                     q3[i-1][j][k], idx)
                f_ori_4 = f_ori_term_3D(D_q, D_q_xp, D_q_xm, D_q_yp, D_q_ym, D_q_zp, D_q_zm, mgq_xp, mgq_xm, mgq_yp, mgq_ym, mgq_zp, mgq_zm,
                                     q4[i][j][k], q4[i][j][k+1], q4[i][j][k-1], q4[i][j+1][k], q4[i][j-1][k], q4[i+1][j][k], 
                                     q4[i-1][j][k], idx)

                #dfintdq1 = 16.*ebar2*T[i][j]*y_e/mag_grad_phi2 * (psix3*(q1[i][j]*dphidx - q4[i][j]*dphidy) + psiy3*(q4[i][j]*dphidx + q1[i][j]*dphidy))
                #dfintdq4 = 16.*ebar2*T[i][j]*y_e/mag_grad_phi2 * (psix3*(-q4[i][j]*dphidx - q1[i][j]*dphidy) + psiy3*(q1[i][j]*dphidx - q4[i][j]*dphidy))
                dfintdq1 = 0. #use these blocks to zero out twisting in quaternion fields to lower interfacial energy
                dfintdq2 = 0.
                dfintdq3 = 0.
                dfintdq4 = 0.

                lq1 = (q1[i][j+1][k]+q1[i][j-1][k]+q1[i+1][j][k]+q1[i-1][j][k]+q1[i][j][k+1]+q1[i][j][k-1]-6*q1[i][j][k])*idx*idx
                lq2 = (q2[i][j+1][k]+q2[i][j-1][k]+q2[i+1][j][k]+q2[i-1][j][k]+q2[i][j][k+1]+q2[i][j][k-1]-6*q2[i][j][k])*idx*idx
                lq3 = (q3[i][j+1][k]+q3[i][j-1][k]+q3[i+1][j][k]+q3[i-1][j][k]+q3[i][j][k+1]+q3[i][j][k-1]-6*q3[i][j][k])*idx*idx
                lq4 = (q4[i][j+1][k]+q4[i][j-1][k]+q4[i+1][j][k]+q4[i-1][j][k]+q4[i][j][k+1]+q4[i][j][k-1]-6*q4[i][j][k])*idx*idx

                q_noise_coeff = 0.0000000001
                noise_q1 = noise_amp_q*math.sqrt(q_noise_coeff*8.314*T[i][j][k]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
                noise_q2 = noise_amp_q*math.sqrt(q_noise_coeff*8.314*T[i][j][k]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
                noise_q3 = noise_amp_q*math.sqrt(q_noise_coeff*8.314*T[i][j][k]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
                noise_q4 = noise_amp_q*math.sqrt(q_noise_coeff*8.314*T[i][j][k]/v_m)*cuda.random.xoroshiro128p_normal_float32(rng_states, threadId)
                #noise_q1 = 0.
                #noise_q4 = 0.
                qt1 = f_ori_1+lq1*eqbar2-dfintdq1+noise_q1
                qt2 = f_ori_2+lq2*eqbar2-dfintdq2+noise_q2
                qt3 = f_ori_3+lq3*eqbar2-dfintdq3+noise_q3
                qt4 = f_ori_4+lq4*eqbar2-dfintdq4+noise_q4

                dq1dt = M_q*((1-q1[i][j][k]**2)*qt1 - q1[i][j][k]*q2[i][j][k]*qt2 - q1[i][j][k]*q3[i][j][k]*qt3 - q1[i][j][k]*q4[i][j][k]*qt4)
                dq2dt = M_q*((1-q2[i][j][k]**2)*qt2 - q2[i][j][k]*q1[i][j][k]*qt1 - q2[i][j][k]*q3[i][j][k]*qt3 - q2[i][j][k]*q4[i][j][k]*qt4)
                dq3dt = M_q*((1-q3[i][j][k]**2)*qt3 - q3[i][j][k]*q2[i][j][k]*qt2 - q3[i][j][k]*q1[i][j][k]*qt1 - q3[i][j][k]*q4[i][j][k]*qt4)
                dq4dt = M_q*((1-q4[i][j][k]**2)*qt4 - q4[i][j][k]*q2[i][j][k]*qt2 - q4[i][j][k]*q3[i][j][k]*qt3 - q4[i][j][k]*q1[i][j][k]*qt1)
                dphi = dt*dphidt
                #dphi = max(-0.01, dt*dphidt)
                #dphi = min(0.01, dphi)
                phi_out[i][j][k] = phi[i][j][k]+dphi
                if(phi_out[i][j][k] < 0.0001):
                    phi_out[i][j][k] = 0.0001
                if(phi_out[i][j][k] > 0.9999):
                    phi_out[i][j][k] = 0.9999
                q1_out[i][j][k] = q1[i][j][k] + dt*dq1dt
                q2_out[i][j][k] = q2[i][j][k] + dt*dq2dt
                q3_out[i][j][k] = q3[i][j][k] + dt*dq3dt
                q4_out[i][j][k] = q4[i][j][k] + dt*dq4dt
                renorm = math.sqrt((q1_out[i][j][k]**2+q2_out[i][j][k]**2+q3_out[i][j][k]**2+q4_out[i][j][k]**2))
                q1_out[i][j][k] = q1_out[i][j][k]/renorm
                q2_out[i][j][k] = q2_out[i][j][k]/renorm
                q3_out[i][j][k] = q3_out[i][j][k]/renorm
                q4_out[i][j][k] = q4_out[i][j][k]/renorm
                for l in range(5, len(fields)):
                    c_i = fields[l]
                    c_i_out = fields_out[l]
                    c_i_out[i][j][k] *= dt
                    #c_i_out[i][j][k] = max(-0.1, c_i_out[i][j][k])
                    #c_i_out[i][j][k] = min(0.1, c_i_out[i][j][k])
                    c_i_out[i][j][k] += c_i[i][j][k]
                    #c_i_out[i][j][k] = max(0, c_i_out[i][j][k])

def npvalue(var, string, tdb):
    """
    Returns a numpy float from the sympy expression gotten from pycalphad
    Reason: some numpy functions (i.e. sqrt) are incompatible with sympy floats!
    """
    return se.lambdify(var, [tdb.symbols[string]])(1000)
    
class NCGPU_new(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL" #must be this framework for this engine
        self.user_data["d_ratio"] = 4. #default value
        self.user_data["noise_phi"] = 1. #default value
        self.user_data["noise_c"] = 1. #default value
        self.user_data["noise_q"] = 1. #default value
        self.user_data["seed_composition"] = None #default value
        
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
            self.set_time_step_length(self.get_cell_spacing()**2/20./self.user_data["D_L"]/len(self.dimensions))
            self.user_data["beta"] = 1.5
        except Exception as e:
            print("Could not load every parameter required from the TDB file!")
            print(e)
            raise Exception()
            
    def init_fields(self):
        #TODO: fix melt_angle for 3D case!
        self._num_transfer_arrays = 2*len(self._tdb_components)
        self._tdb_ufunc_input_size = len(self._tdb_components)+1
        seed = 1
        if(self._parallel):
            seed = self._MPI_rank
        self.user_data["rng_states"] = create_xoroshiro128p_states(256*256, seed=seed)
        #init_xoroshiro128p_states(256*256, seed=3446621627)
        dim = self.dimensions
        gdim = self._global_dimensions
        phi = np.zeros(dim)
        q1 = np.zeros(dim)
        if(len(self.dimensions) == 3):
            q2 = np.zeros(dim)
            q3 = np.zeros(dim)
        q4 = np.zeros(dim)
        try:
            melt_angle = self.user_data["melt_angle"]
        except:
            print("self.user_data[\"melt_angle\"] not defined, defaulting to 0")
            melt_angle = 0.
        #angle is halved because that is how quaternions do
        q1 += np.cos(0.5*melt_angle)
        q4 += np.sin(0.5*melt_angle)
        self.add_field(phi, "phi", colormap=COLORMAP_PHASE)
        self.add_field(q1, "q1")
        if(len(self.dimensions) == 3):
            self.add_field(q2, "q2")
            self.add_field(q3, "q3")
        self.add_field(q4, "q4")
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
        try:
            sim_type = self.user_data["sim_type"]
        except:
            #if sim_type isn't defined, we are done initializing a blank simulation
            print("self.user_data[\"sim_type\"] not defined, defaulting to an empty simulation")
            sim_type = None
        if(sim_type == "seed"):
            try:
                seed_angle = self.user_data["seed_angle"]
            except:
                print("self.user_data[\"seed_angle\"] not defined, defaulting to pi/4")
                seed_angle = 1*np.pi/4
            try:
                seed_composition = self.user_data["seed_composition"]
                assert not(seed_composition is None)
            except:
                print("self.user_data[\"seed_composition\"] not specified, defaulting to order-parameter-only nucleation for seeds")
                seed_composition = None
            try:
                seed_size = self.user_data["seed_radius"]
            except:
                print("self.user_data[\"seed_radius\"] not specified, defaulting to a radius of 5 cells")
                seed_size = 5
            if(len(self.dimensions) == 3):
                c_index_list = []
                for i in range(len(self._tdb_components)-1):
                    c_index_list.append(5+i)
                make_seed(self, p=0, q=[1, 2, 3, 4], c=c_index_list, composition=seed_composition, x=gdim[2]/2, y=gdim[1]/2, z=gdim[0]/2, angle=seed_angle, seed_radius=seed_size)
            else:
                c_index_list = []
                for i in range(len(self._tdb_components)-1):
                    c_index_list.append(3+i)
                make_seed(self, p=0, q=[1, 2], c=c_index_list, composition=seed_composition, x=gdim[1]/2, y=gdim[0]/2, angle=seed_angle, seed_radius=seed_size)
            #initialize concentration array(s)

        elif(sim_type == "seeds"):
            #initialize phi, q1, q4
            try:
                number_of_seeds = self.user_data["number_of_seeds"]
            except:
                print("self.user_data[\"number_of_seeds\"] not defined, defaulting to about 1 seed per 100^"+str(len(gdim))+" cells")
                number_of_seeds = np.prod(gdim)//(100**len(gdim))
            try:
                seed_composition = self.user_data["seed_composition"]
                assert not(seed_composition is None)
            except:
                print("self.user_data[\"seed_composition\"] not specified, defaulting to order-parameter-only nucleation for seeds")
                seed_composition = None
            try:
                seed_size = self.user_data["seed_radius"]
            except:
                print("self.user_data[\"seed_radius\"] not specified, defaulting to a radius of 5 cells")
                seed_size = 5
            np.random.seed(3)
            for j in range(number_of_seeds):
                if(len(self.dimensions) == 3):
                    c_index_list = []
                    for i in range(len(self._tdb_components)-1):
                        c_index_list.append(5+i)
                    make_seed(self, p=0, q=[1, 2, 3, 4], c=c_index_list, composition=seed_composition, seed_radius=seed_size)
                else:
                    c_index_list = []
                    for i in range(len(self._tdb_components)-1):
                        c_index_list.append(3+i)
                    make_seed(self, p=0, q=[1, 2], c=c_index_list, composition=seed_composition, seed_radius=seed_size)
        
        
    
    def save_simulation(self):
        super().save_simulation()
        if not self._save_path:
            #if save path is not defined, do not save, just return
            print("self._save_path not defined, aborting save!")
            return
        else:
            save_loc = Path(self._save_path)
        save_loc.mkdir(parents=True, exist_ok=True)
        if "save_points" in self.user_data:
            save_points_array = self.spa_gpu.copy_to_host()
            np.save(str(save_loc) + "/savepointsarray_" + str(self.time_step_counter), save_points_array)
    
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
            self.user_data["thermo_finite_diff_incr"] = 0.0000001
        if "save_points" in self.user_data:
            save_points_array = np.zeros([len(self.user_data["save_points"][0]), len(self.fields)+1, self._autosave_rate])
            self.spa_gpu = cuda.to_device(save_points_array)
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
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])
        self.user_data["c_params_GPU"] = cuda.to_device(self.user_data["c_params"])
        
    def simulation_loop(self):
        cuda.synchronize()
        if(len(self.dimensions) == 1):
            NComponent_helper_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            cuda.synchronize()
            NComponent_noise_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            cuda.synchronize()
            NComponent_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            if "save_points" in self.user_data:
                cuda.synchronize()
                NComponent_sp_kernel[self._gpu_blocks_per_grid_1D, self._gpu_threads_per_block_1D](self._fields_gpu_device,
                                                                        self._temperature_gpu_device,
                                                                        self.spa_gpu, self.user_data["save_points"], 
                                                                        (self.time_step_counter-1)%self._autosave_rate)
        elif(len(self.dimensions) == 2):
            NComponent_helper_kernel_2D[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            cuda.synchronize()
            NComponent_noise_kernel_2D[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            cuda.synchronize()
            NComponent_kernel_2D[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            if "save_points" in self.user_data:
                cuda.synchronize()
                NComponent_sp_kernel_2D[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,
                                                                        self._temperature_gpu_device,
                                                                        self.spa_gpu, self.user_data["save_points"], 
                                                                        (self.time_step_counter-1)%self._autosave_rate)
        elif(len(self.dimensions) == 3):
            NComponent_helper_kernel_3D[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            cuda.synchronize()
            NComponent_noise_kernel_3D[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self.user_data["rng_states"], self._tdb_ufunc_gpu_device, 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            cuda.synchronize()
            NComponent_kernel_3D[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device, 
                                                                      self._temperature_gpu_device, self._fields_transfer_gpu_device, 
                                                                      self._fields_out_gpu_device, self.user_data["rng_states"], 
                                                                      self.user_data["params_GPU"], self.user_data["c_params_GPU"])
            if "save_points" in self.user_data:
                cuda.synchronize()
                NComponent_sp_kernel_3D[self._gpu_blocks_per_grid_3D, self._gpu_threads_per_block_3D](self._fields_gpu_device,
                                                                        self._temperature_gpu_device,
                                                                        self.spa_gpu, self.user_data["save_points"], 
                                                                        (self.time_step_counter-1)%self._autosave_rate)
        cuda.synchronize()