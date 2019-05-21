# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 13:28:56 2019

@author: Matt

Animation of transverse velocity solution for asymmetric slab IVP for initially
uniform vorticity .
"""

import numpy as np
import scipy as sc
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

# Define parameters
K = 1.
R1 = 1.5
R2 = 1.2
W = 1.

# Define the alfven speeds.
vA0 = 1.
vA1 = 0.4
vA2 = 0.5

# Define initial vorticity
Omega0 = 1.

# Define eigen phase speeds
a = (1 + R1*R2)*np.tanh(2*K) + R1 + R2
b = -(2*vA0**2 + R1*R2*(vA1**2 + vA2**2))*np.tanh(2*K) - (R1*(vA0**2 + vA1**2) + R2*(vA0**2 + vA2**2))
c = (vA0**4 + R1*R2*vA1**2*vA2**2)*np.tanh(2*K) + vA0**2*(R1*vA1**2 + R2*vA2**2)
W0p = sc.sqrt((-b + sc.sqrt(b**2 - 4*a*c)) / (2*a))
W0m = sc.sqrt((-b - sc.sqrt(b**2 - 4*a*c)) / (2*a))


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
    return -2*R1*W


def eps2_prime(W):
    return -2*R2*W


def D(W):
    e0 = eps0(W)
    e1 = eps1(W)
    e2 = eps2(W)
    return e0*(e1 + e2)*np.cosh(2*K) + (e0**2 + e1*e2)*np.sinh(2*K)


def D_prime(W):
    e0, e1, e2 = eps0(W), eps1(W), eps2(W)
    e0_p, e1_p, e2_p = eps0_prime(W), eps1_prime(W), eps2_prime(W)
    return (e0_p*(e1 + e2) + e0*(e1_p + e2_p))*np.cosh(2*K) + (2*e0*e0_p + e1_p*e2 + e1*e2_p)*np.sinh(2*K)


def T1(W):
    return -Omega0*((np.tanh(K) + R1)*(eps0(W)*np.cosh(2*K) + eps2(W)*np.sinh(2*K)) + eps0(W)*(np.tanh(K) + R2))


def T2(W):
    return -Omega0*((np.tanh(K) + R2)*(eps0(W)*np.cosh(2*K) + eps1(W)*np.sinh(2*K)) + eps0(W)*(np.tanh(K) + R1))


def Astar1(t):
    return W0p*np.cos(W0p*t)*T1(W0p)/D_prime(W0p) + W0m*np.cos(W0m*t)*T1(W0m)/D_prime(W0m)


def Astar2(t):
    return W0p*np.cos(W0p*t)*T2(W0p)/D_prime(W0p) + W0m*np.cos(W0m*t)*T2(W0m)/D_prime(W0m)


# Define the transverse velocity solution
def vx_hat(x, t):
    if x < -K:
        vx_hat_vals = -1j*(2*np.exp(x + K)*Astar1(t) + Omega0*(1 - np.exp(x + K))*np.cos(vA1*t))
    if x >= -K and x < K:
        vx_hat_vals = -1j*((2/np.sinh(2*K))*(Astar1(t)*np.sinh(K - x) + Astar2(t)*np.sinh(K + x))
                           + Omega0*(1 - np.cosh(x)/np.cosh(K))*np.cos(vA0*t))
    if x >= K:
        vx_hat_vals = -1j*(2*np.exp(K - x)*Astar2(t) + Omega0*(1 - np.exp(K - x))*np.cos(vA2*t))
    return vx_hat_vals


def vx(x, z, t):
    return vx_hat(x, t)*np.exp(1j*z)

# Evaluate the solution at height z
z = 0.

# Coordinate values
x_vals = np.linspace(-K - 1, K + 1, 100)
t_vals = np.linspace(0, 100, 201)

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
line_ani = animation.FuncAnimation(fig1, update, len(t_vals),
                                   interval=50, blit=True)

# Save animation
line_ani.save('sol_animation.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
