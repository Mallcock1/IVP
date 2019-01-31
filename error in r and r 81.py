# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:48:52 2018

@author: Matt
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

epsm = 1.
epsp = 1.5

def vx1a(x):
    vx_vals = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < 0:
            vx_vals[i] = 1./epsm * (-1. + np.exp(xi))
        else:
            vx_vals[i] = 1./epsp * (1. - np.exp(-xi))
    return vx_vals
    
def vx1b(x):
    vx_vals = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < 0:
            vx_vals[i] = 1./epsm * (1. + np.exp(xi)*(epsm-epsp)/(epsp+epsm))
        else:
            vx_vals[i] = 1./epsp * (1. + np.exp(-xi)*(epsp-epsm)/(epsp+epsm))
    return vx_vals


def vx2a(x):
    vx_vals = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < 0:
            vx_vals[i] = -1./epsm * (1. - np.exp(xi)*(epsp)/(epsp+epsm))
        else:
            vx_vals[i] = -1./(epsp+epsm) * np.exp(-xi)
    return vx_vals
    
def vx2b(x):
    vx_vals = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < 0:
            vx_vals[i] = 1./epsm * (1. - np.exp(xi)*(epsp)/(epsp+epsm))
        else:
            vx_vals[i] = 1./(epsp+epsm) * np.exp(-xi)
    return vx_vals
    

def vx3(x, x0):
    vx_vals = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < 0:
            vx_vals[i] = 1./(epsm+epsp) * np.exp(-(x0-xi))
        else:
            if xi < x0:
                vx_vals[i] = (epsp - epsm)/(2*(epsp + epsm)*epsp)*np.exp(-(xi+x0)) + 1./(2*epsp) * np.exp(-(x0 - xi))
            else:
                vx_vals[i] = (epsp - epsm)/(2*(epsp + epsm)*epsp)*np.exp(-(xi+x0)) + 1./(2*epsp) * np.exp(-(xi - x0))
    return vx_vals
    
x = np.arange(-5, 5, 0.001)

gs = gridspec.GridSpec(1, 3)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])

ax1.set_xlim(-5,5)
ax1.set_ylim(-1.2,1.2)
ax1.plot(x, vx1a(x))
ax1.plot(x, vx1b(x))
ax1.set_ylabel(r'$\tilde{v}_x$', fontsize=20)
ax1.set_aspect('4')
ax1.grid(True)

ax2.set_xlim(-5,5)
ax2.set_ylim(-1.2,1.2)
ax2.plot(x, vx2a(x))
ax2.plot(x, vx2b(x))
ax2.set_xlabel(r'$x$', fontsize=20)
ax2.yaxis.set_ticklabels([])
ax2.set_aspect('4')
ax2.grid(True)

ax3.set_xlim(-5,5)
ax3.set_ylim(-1.2,1.2)
ax3.plot(x, vx3(x,1))
ax3.plot(x, vx3(x,1))
ax3.yaxis.set_ticklabels([])
ax3.set_aspect('4')
ax3.grid(True)

plt.tight_layout()

plt.savefig('v_x', bbox_inches = 'tight')