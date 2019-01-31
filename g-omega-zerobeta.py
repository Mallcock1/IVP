# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:36:41 2018

@author: Matt
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

vA0 = 1.
vA1 = 0.5
vA2 = 0.7

#vA0 = 1.
#vA1 = 2.
#vA2 = 3.

def lamb0(W):
    return sc.sqrt(1 - W**2/vA0**2)

def lamb1(W):
    return sc.sqrt(1 - W**2/vA1**2)

def lamb2(W):
    return sc.sqrt(1 - W**2/vA2**2)

def disp_rel_thin(W, K):
    return lamb1(W) + lamb2(W) + 2*K*(lamb0(W)**2 + lamb1(W)*lamb2(W))
    
def disp_rel(W, K):
    return lamb0(W)*(lamb1(W) + lamb2(W)) + (lamb0(W)**2 + lamb1(W)*lamb2(W))*sc.tanh(2*lamb0(W)*K)

Kmin = 0
Wmin = 0.8
Kmax = 10
Wmax = 1
gmax = 2
gmin = -2

gvals = np.zeros((100, 100))
Wvals = np.arange(Wmin,Wmax,(Wmax-Wmin)*0.01)
Kvals = np.arange(Kmin,Kmax,(Kmax-Kmin)*0.01)
X, Y = np.meshgrid(Wvals,Kvals)
K = Kmin
for i in range(0, 100):
    W = Wmin
    for j in range(0, 100):
        dr = disp_rel(W, K)
        if dr < gmax and dr > gmin:
            gvals[i,j] = dr
        else:
            if dr > gmax:
                gvals[i,j] = gmax
            else:
                gvals[i,j] = gmin
        W = W + (Wmax-Wmin)*0.01
        j = j+1
    i = i + 1
    K = K + (Kmax - Kmin)*0.01

ax.plot_wireframe(X,Y,np.zeros((100, 100)), rstride=10, cstride=10)
ax.plot_surface(X, Y, gvals)
ax.set_zlim(-2, 2)