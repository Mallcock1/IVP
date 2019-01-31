# -*- coding: utf-8 -*-
"""
Created on Wed May 02 14:57:04 2018

@author: Matt
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

vAm = 1
vAp = 1.5
vs = np.sqrt(0.5*(vAm**2 + vAp**2))

# Incorrect function from R + R 81
def vx_rr81(x, t):
    vx_vals = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < 0:
            vx_vals[i] = - np.cos(vAm*t) * (1 - np.exp(xi))
        else:
            vx_vals[i] = - np.cos(vAp*t) * (np.exp(-xi) - 1)
    return vx_vals

# Corrected version of function by me.
def vx_ae18(x, t):
    vx_vals = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < 0:
            vx_vals[i] = - np.cos(vAm*t) * (1 - np.exp(xi)) - np.cos(vs*t) * np.exp(xi)
        else:
            vx_vals[i] = - np.cos(vAp*t) * (1 - np.exp(-xi)) - np.cos(vs*t) * np.exp(-xi)
    return vx_vals
    
    
x = np.arange(-5, 5, 0.001)
t = 0

#ax1.set_xlim(-5,5)
#ax1.set_ylim(-1.2,1.2)
plt.plot(x, vx_rr81(x, t))
plt.plot(x, vx_ae18(x, t))
plt.ylabel(r'$v_x$', fontsize=20)
#ax1.set_aspect('4')
#ax1.grid(True)

plt.tight_layout()

#plt.savefig('v_x', bbox_inches = 'tight')