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

K = 1.
x0 = 1.
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


def D(W):
    e0 = eps0(W)
    e1 = eps1(W)
    e2 = eps2(W)
    return e0*(e1 + e2)*np.cosh(2*K) + (e0**2 + e1*e2)*np.sinh(2*K)


def D_prime(W):
    e0, e0p = eps0(W), eps0_prime(W)
    e1, e1p = eps1(W), eps1_prime(W)
    e2, e2p = eps2(W), eps2_prime(W)
    return (e0p*(e1 + e2) + e0*(e1p + e2p))*np.cosh(2*K) + (2*e0p*e0 + e1p*e2 + e1*e2p)*np.sinh(2*K)


def I0p(f, W):
    return comp_quad(lambda S: (np.sinh(S + K)/np.sinh(2*K)) * f(S), -K, K)[0]


def I0m(f, W):
    return comp_quad(lambda S: (np.sinh(S - K)/np.sinh(2*K)) * f(S), -K, K)[0]


def I1(f, W):
    return comp_quad(lambda S: np.exp(S + K) * f(S), -np.infty, -K)[0]


def I2(f, W):
    return comp_quad(lambda S: np.exp(K - S) * f(S), K, np.infty)[0]


def T1(f, W):
    return (I0m(f, W) - I1(f, W))*(eps0(W)*np.cosh(2*K) + eps2(W)*np.sinh(2*K)) - eps0(W)*(I0p(f, W) + I2(f, W))


def T2(f, W):
    return eps0(W)*(I0m(f, W) - I1(f, W)) - (I0p(f, W) + I2(f, W))*(eps0(W)*np.cosh(2*K) + eps1(W)*np.sinh(2*K))


def chi1p(f):
    return T1(f, W0p) / D_prime(W0p)


def chi1m(f):
    return T1(f, W0m) / D_prime(W0m)


def chi2p(f):
    return T2(f, W0p) / D_prime(W0p)


def chi2m(f):
    return T2(f, W0m) / D_prime(W0m)


def greens0(x, S):
    if x < S:
        g = (1/np.sinh(2*K))*np.sinh(S - K)*np.sinh(x + K)
    elif x >= S:
        g = (1/np.sinh(2*K))*np.sinh(x - K)*np.sinh(S + K)
    return g


def greens1(x, S):
    if x < S:
        g = np.exp(x + K)*np.sinh(S + K)
    elif x >= S:
        g = np.exp(S + K)*np.sinh(x + K)
    return g


def greens2(x, S):
    if x < S:
        g = -np.exp(K - S)*np.sinh(x - K)
    elif x >= S:
        g = -np.exp(K - x)*np.sinh(S - K)
    return g


def A1(x, t):
    return (1j*W0p*chi1p(psi0)*np.cos(W0p*t) - chi1p(dpsi0_dt)*np.sin(W0p*t) +
            1j*W0m*chi1m(psi0)*np.cos(W0m*t) - chi1m(dpsi0_dt)*np.sin(W0m*t))


def A2(x, t):
    return (1j*W0p*chi2p(psi0)*np.cos(W0p*t) - chi2p(dpsi0_dt)*np.sin(W0p*t) +
            1j*W0m*chi2m(psi0)*np.cos(W0m*t) - chi2m(dpsi0_dt)*np.sin(W0m*t))


# Define the transverse velocity solution
def vx_hat(x, t):
    if x < -K:
        vx_hat_vals = (-2 * np.exp(K + x)*A1(x, t) +
                       (1j / R1)*comp_quad(lambda S: greens1(x, S)*(psi0(S)*np.cos(vA1*t) + dpsi0_dt(S)*np.sin(vA1*t)/vA1), -np.infty, -K)[0])
    if x >= -K and x < K:
        vx_hat_vals = ((-2 / np.sinh(2*K)) * (A1(x, t)*np.sinh(K - x) + A2(x, t)*np.sinh(K + x)) +
                       1j*comp_quad(lambda S: greens0(x, S)*(psi0(S)*np.cos(vA0*t) + dpsi0_dt(S)*np.sin(vA0*t)/vA0), -K, K)[0])
    if x >= K:
        vx_hat_vals = (-2 * np.exp(K - x)*A2(x, t) +
                       (1j / R2)*comp_quad(lambda S: greens2(x, S)*(psi0(S)*np.cos(vA2*t) + dpsi0_dt(S)*np.sin(vA2*t)/vA2), K, np.infty)[0])
    return vx_hat_vals


def vx(x, z, t):
    return vx_hat(x, t)*np.exp(1j*z)

# Evaluate the solution at height z
z = 0.

# Coordinate values
x_vals = np.linspace(-K - 1, K + 1, 30)
t_vals = np.linspace(0, 25, 51)

# Initialise solution array
vx_vals = np.empty((len(x_vals), len(t_vals)))

# Fill array
for i, t in enumerate(t_vals):
    for j, x in enumerate(x_vals):
        vx_vals[j, i] = 1j*vx(x, z, t)

# Initialise figure
fig1 = plt.figure()
l, = plt.plot(x_vals, vx_vals[:, 0], 'r-')


# Fuction to update animation at each time step
def update(i):
    l.set_data(x_vals, vx_vals[:, i])
    return l,

# Set axis limits
plt.ylim(-2.5, 2.5)

# Animate!
line_ani = animation.FuncAnimation(fig1, update, len(t_vals), interval=200,
                                   blit=True)

# Save animation
#line_ani.save('sol_animation2.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
