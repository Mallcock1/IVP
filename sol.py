# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:48:58 2019

@author: Matt
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

"""
K = kx_0
x = kx
S = ks
W = w / k

R1 = rho_1 / rho_0
R2 = rho_2/ rho_0
"""

K = 1.
R1 = 0.5
R2 = 0.8

#Define the alfven speeds.
vA0 = 1.
vA1 = 0.4
vA2 = 0.5

x_vals = np.linspace(-K - 1, K + 1, 1000)


# define eigen phase speeds
a = (1 + R1*R2)*np.tanh(2*K) + R1 + R2
b = -(2*vA0**2 + R1*R2*(vA1**2 + vA2**2))*np.tanh(2*K) - (R1*(vA0**2 + vA1**2) + R2*(vA0**2 + vA2**2))
c = vA0**4 + R1*R2*vA1**2*vA2**2*np.tanh(2*K) + vA0**2*(R1*vA1**2 + R2*vA2**2)


def eps0(W):
    return vA0**2 - W**2

def eps1(W):
    return R1*(vA1**2 - W**2)

def eps2(W):
    return R2*(vA2**2 - W**2)

def D(W):
    e0 = eps0(W)
    e1 = eps1(W)
    e2 = eps2(W)
    return e0*(e1 + e2)*np.cosh(2*K) + (e0**2 + e1*e2)*np.sinh(2*K)

def I0p(f, W):
    return sc.integrate.quad(lambda S: (np.sinh(S + K) / np.sinh(2*K)) * f(S, W),
                             -K, K)

def I0m(f, W):
    return sc.integrate.quad(lambda S: (np.sinh(S - K) / np.sinh(2*K)) * f(S, W),
                             -K, K)

def I1(f, W):
    return sc.integrate.quad(lambda S: np.exp(S + K) * f(S, W), -np.infty, -K)

def I2(f, W):
    return sc.integrate.quad(lambda S: np.exp(K - S) * f(S, W), K, np.infty)

def T1(f, W):
    return (I0m(f, W) - I1(f, W))*(eps0(W)*np.cosh(2*K) + esp2(W)*np.sinh(2*K)) - eps0(W)*(I0p(f, W) + I2(f, W))
    
def T2(f, W):
    return eps0(W)*(I0m(f, W) - I1(f, W)) - (I0p(f, W) + I2(f, W))*(eps0(W)*np.cosh(2*K) + esp1(W)*np.sinh(2*K))

def chi1p(f):
    return T1(f, W0p) / D(W0p)
    
def chi1m(f):
    return T1(f, W0m) / D(W0m)
    
def chi2p(f):
    return T2(f, W0p) / D(W0p)
    
def chi2m(f):
    return T2(f, W0m) / D(W0m)
    



    


sc.misc.derivative


    
plt.figure()
for s in np.linspace(-x0, x0):
    g_vals = np.zeros_like(x_vals)
    
    for i, x in enumerate(x_vals):
        g_vals[i] = green(x, s)
    plt.plot(x_vals, g_vals)