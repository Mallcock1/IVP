# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 16:04:42 2018

@author: Matt
"""

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


r0 = 1.
r1 = 1.5
def r2(e):
    return r1 + e
    
v0 = 1.
v1 = 1.5
def v2(e):
    return v1 + e #np.sign(e)*sc.sqrt(np.abs(e))


def a(e, K):
    return (r0**2 + r1*r2(e)) * sc.tanh(2*K) + r0*(r1 + r2(e))

def b(e, K):
    return -(2*r0**2*v0**2 + r1*r2(e)*(v1**2 + v2(e)**2))*sc.tanh(2*K) - r0*(r1*(v0**2 + v1**2) + r2(e)*(v0**2 + v2(e)**2))

def c(e, K):
    return (r0**2*v0**4 + r1*r2(e)*v1**2*v2(e)**2)*sc.tanh(2*K) + r0*v0**2*(r1*v1**2 + r2(e)*v2(e)**2)
    

def disc(e, K):
    return b(e, K)**2 - 4*a(e, K)*c(e, K)
    

vals = np.linspace(-10, 10, 1000001)

plt.plot(vals, disc(vals, 1))
#plt.plot([0,0], [1,1])