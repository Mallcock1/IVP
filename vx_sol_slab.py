# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:48:58 2019

@author: Matt

Plotting the solution to the IVP for incompressible asymmetric slab.
"""

import numpy as np
import scipy as sc
import scipy.integrate as ig
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
K := kx_0
x := kx
z := kz
S := ks
W := w / k

R1 := rho_1 / rho_0
R2 := rho_2/ rho_0
"""

#K = 1.
x0 = 1.
z0 = 1.
R1 = 1.5
R2 = 1.2
w = 1.

#Define the alfven speeds.
vA0 = 1.
vA1 = 0.4
vA2 = 0.5

# define eigen phase speeds
def a(k):
    return (1 + R1*R2)*np.tanh(2*k*x0) + R1 + R2
def b(k):
    return (-(2*vA0**2 + R1*R2*(vA1**2 + vA2**2))*np.tanh(2*k*x0)
            - (R1*(vA0**2 + vA1**2) + R2*(vA0**2 + vA2**2)))
def c(k):
    return ((vA0**4 + R1*R2*vA1**2*vA2**2)*np.tanh(2*k*x0)
            + vA0**2*(R1*vA1**2 + R2*vA2**2))

def W0p(k):
    return sc.sqrt((-b(k) + sc.sqrt(b(k)**2 - 4*a(k)*c(k))) / (2*a(k)))
def W0m(k):
    return sc.sqrt((-b(k) - sc.sqrt(b(k)**2 - 4*a(k)*c(k))) / (2*a(k)))


# Define complex integration
def comp_quad(func, a, b, **kwargs):
    """
    Calculates the integral of a complex function from a to b.
    Returns the integral and an estimate of the absolute error in the real and
    imaginary parts, respectively.
    """
    def real_func(x):
        return sc.real(func(x))

    def imag_func(x):
        return sc.imag(func(x))
    real_integral = ig.quad(real_func, a, b, **kwargs)
    imag_integral = ig.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:],
            imag_integral[1:])


# Define initial conditions
def psi0(x, k):
    return k


def dpsi0_dt(x, k):
    return 0.


#def f(x, k):
#    return w*psi0(x) + 1.j*dpsi0_dt(x)


# Define functions that make up the solution
def eps0(W):
    return vA0**2 - W**2


def eps1(W):
    return R1*(vA1**2 - W**2)


def eps2(W):
    return R2*(vA2**2 - W**2)


def eps0_prime(W):
    return -2*W


def eps1_prime(W):
    return -R1*2*W


def eps2_prime(W):
    return -R2*2*W


def D(W, k):
    e0 = eps0(W)
    e1 = eps1(W)
    e2 = eps2(W)
    return e0*(e1 + e2)*np.cosh(2*k*x0) + (e0**2 + e1*e2)*np.sinh(2*k*x0)


def D_prime(W, k):
    e0, e0p = eps0(W), eps0_prime(W)
    e1, e1p = eps1(W), eps1_prime(W)
    e2, e2p = eps2(W), eps2_prime(W)
    return ((e0p*(e1 + e2) + e0*(e1p + e2p))*np.cosh(2*k*x0)
            + (2*e0p*e0 + e1p*e2 + e1*e2p)*np.sinh(2*k*x0))


def I0p(f, k):
    return comp_quad(lambda s: (np.sinh(k*(s + x0))/np.sinh(2*k*x0)) * f(s), -x0, x0)[0]


def I0m(f, k):
    return comp_quad(lambda s: (np.sinh(k*(s - x0))/np.sinh(2*k*x0)) * f(s), -x0, x0)[0]


def I1(f, k):
    return comp_quad(lambda s: np.exp(k*(s + x0)) * f(s), -np.inf, -x0)[0]


def I2(f, k):
    return comp_quad(lambda s: np.exp(k*(x0 - s)) * f(s), x0, np.inf)[0]


def T1(f, W, k):
    return ((I0m(f, k) - I1(f, k))*(eps0(W)*np.cosh(2*k*x0) + eps2(W)*np.sinh(2*k*x0))
            - eps0(W)*(I0p(f, k) + I2(f, k)))


def T2(f, W, k):
    return (eps0(W)*(I0m(f, k) - I1(f, k))
            - (I0p(f, k) + I2(f, k))*(eps0(W)*np.cosh(2*k*x0) + eps1(W)*np.sinh(2*k*x0)))


def chi1p(f, k):
    return T1(f, W0p(k), k) / D_prime(W0p(k), k)


def chi1m(f, k):
    return T1(f, W0m(k), k) / D_prime(W0m(k), k)


def chi2p(f, k):
    return T2(f, W0p(k), k) / D_prime(W0p(k), k)


def chi2m(f, k):
    return T2(f, W0m(k), k) / D_prime(W0m(k), k)


def greens0(x, s, k):
    if x < s:
        g = (1/np.sinh(2*k*x0))*np.sinh(k*(s - x0))*np.sinh(k*(x + x0))
    elif x >= s:
        g = (1/np.sinh(2*k*x0))*np.sinh(k*(x - x0))*np.sinh(k*(s + x0))
    return g


def greens1(x, s, k):
    if x < s:
        g = np.exp(k*(x + x0))*np.sinh(k*(s + x0))
    elif x >= s:
        g = np.exp(k*(s + x0))*np.sinh(k*(x + x0))
    return g


def greens2(x, s, k):
    if x < s:
        g = -np.exp(k*(x0 - s))*np.sinh(k*(x - x0))
    elif x >= s:
        g = -np.exp(k*(x0 - x))*np.sinh(k*(s - x0))
    return g


def A1(x, k, t):
    def psi0_1arg(x):
        return psi0(x, k)
    def dpsi0_dt_1arg(x):
        return dpsi0_dt(x, k)
    
    return (1j*W0p(k)*chi1p(psi0_1arg, k)*np.cos(W0p(k)*t)
            - chi1p(dpsi0_dt_1arg, k)*np.sin(W0p(k)*t)
            + 1j*W0m(k)*chi1m(psi0_1arg, k)*np.cos(W0m(k)*t)
            - chi1m(dpsi0_dt_1arg, k)*np.sin(W0m(k)*t))


def A2(x, k, t):
    def psi0_1arg(x):
        return psi0(x, k)
    def dpsi0_dt_1arg(x):
        return dpsi0_dt(x, k)
    
    return (1j*W0p(k)*chi2p(psi0_1arg, k)*np.cos(W0p(k)*t)
            - chi2p(dpsi0_dt_1arg, k)*np.sin(W0p(k)*t)
            + 1j*W0m(k)*chi2m(psi0_1arg, k)*np.cos(W0m(k)*t)
            - chi2m(dpsi0_dt_1arg, k)*np.sin(W0m(k)*t))


# Define the transverse velocity solution
def vx_hat(x, k, t):
    if x < -x0:
        vx_hat_vals = (-2 * np.exp(k*(x0 + x))*A1(x, k, t) +
                       (1j / R1)*comp_quad(lambda s: greens1(x, s, k)*(psi0(s, k)*np.cos(k*vA1*t) + dpsi0_dt(s, k)*np.sin(k*vA1*t)/vA1), -np.inf, -x0)[0])
    if x >= -x0 and x < x0:
        vx_hat_vals = ((-2 / np.sinh(2*k*x0)) * (A1(x, k, t)*np.sinh(k*(x0 - x)) + A2(x, k, t)*np.sinh(k*(x0 + x))) +
                       1j*comp_quad(lambda s: greens0(x, s, k)*(psi0(s, k)*np.cos(k*vA0*t) + dpsi0_dt(s, k)*np.sin(k*vA0*t)/vA0), -x0, x0)[0])
    if x >= x0:
        vx_hat_vals = (-2 * np.exp(k*(x0 - x))*A2(x, k, t) +
                       (1j / R2)*comp_quad(lambda s: greens2(x, s, k)*(psi0(s, k)*np.cos(k*vA2*t) + dpsi0_dt(s, k)*np.sin(k*vA2*t)/vA2), x0, np.inf)[0])
    return vx_hat_vals


def fourier_trans_inv(f, x, z, t):
    """ Returns the inverse fourier transform of function f """
    fti_vals = 1/(2*np.pi) * comp_quad(lambda k: f(x, k, t)*np.exp(1j*k*z),
                                       -np.inf, np.inf)[0]
    return fti_vals
    
def vx(x, z, t):
    return fourier_trans_inv(vx_hat, x, z, t)


v = vx(0.5, 0.5, 0.5)

## Domain
#xmin = -x0 - 1
#xmax = -xmin
#zmin = -3*z0
#zmax = -zmin
#Nx = 10
#Nz = 10
#
#x = np.linspace(xmin, xmax, Nx)
#z = np.linspace(zmin, zmax, Nz)
#X, Z = np.meshgrid(z, x)
#
#tstart = 0
#tend = 5
#Nt = 6
#
#t = np.linspace(tstart, tend, Nt)
#
## initialise vx array
#vxvals = np.zeros((Nx, Nz, Nt))
#
## populate vx array
#for i in range(Nx):
#    for j in range(Nz):
#        for k in range(Nt):
#            vxvals[i, j, k] = vx(x[i], z[j], t[k])







#
#
## Coordinate values
#x_vals = np.linspace(-K - 1, K + 1, 30)
#z_vals = 
#t_vals = np.linspace(0, 25, 51)
#
## Initialise solution array
#vx_vals = np.empty((len(x_vals), len(t_vals)))
#
## Fill array
#for i, t in enumerate(t_vals):
#    for j, x in enumerate(x_vals):
#        vx_vals[j, i] = 1j*vx(x, z, t)
#
## Initialise figure
#fig1 = plt.figure()
#l, = plt.plot(x_vals, vx_vals[:, 0], 'r-')
#
#
## Fuction to update animation at each time step
#def update(i):
#    l.set_data(x_vals, vx_vals[:, i])
#    return l,
#
## Set axis limits
#plt.ylim(-2.5, 2.5)
#
## Animate!
#line_ani = animation.FuncAnimation(fig1, update, len(t_vals), interval=200,
#                                   blit=True)
#
## Save animation
##line_ani.save('sol_animation2.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
