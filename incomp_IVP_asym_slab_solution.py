# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:18:29 2018

@author: Matt


Plot of solution for IVP for initially uniform vorticity
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


Omega0 = 1.
r0 = 1.
r1 = 1.
r2 = 1.

vA0 = 1.
vA1 = 1.
vA2 = 1.

K = 1. #K = kx_0
# X = kx



# The dicrete spectrum modes
a = (r0**2 + r1*r2)*sc.tanh(2*K) + r0*(r1 + r2)
b = -(2*r0**2*vA0**2 + r1*r2*(vA1**2 + vA2**2)) * sc.tanh(2*K) - r0*(r1*(vA0**2 + vA1**2) + r2*(vA0**2 + vA2**2))
c = (r0**2*vA0**4 + r1*r2*vA1**2*vA2**2)*sc.tanh(2*K) + r0*vA0**2*(r1*vA1**2 + r2*vA2**2)

W0p = -b + sc.sqrt(b**2 - 4*a*c)
W0m = -b - sc.sqrt(b**2 - 4*a*c)

def eps0(W):
    return r0*(vA0**2 - W**2)
    
def eps1(W):
    return r1*(vA1**2 - W**2)
    
def eps2(W):
    return r2*(vA2**2 - W**2)
    
def eps0prime(W):
    return 2*r0*W
    
def eps1prime(W):
    return 2*r1*W
    
def eps2prime(W):
    return 2*r2*W

def Dprime(W):
    return ((eps0prime(W)*(eps1(W) + eps2(W)) + eps0(W)*(eps1prime(W) + eps2prime(W)))*sc.cosh(2*K) + 
           (2*eps0(W)*eps0prime(W) + eps1prime(W)*eps2(W) + eps1(W)*eps2prime(W))*sc.sinh(2*K))

def T1(W): 
    return -2*Omega0*((r0*sc.tanh(K) + r1)*(eps0(W)*sc.cosh(2*K) + eps2(W)*sc.sinh(2*K)) + eps0(W)*(r0*sc.tanh(K) + r2))

def T2(W): 
    return -2*Omega0*((r0*sc.tanh(K) + r2)*(eps0(W)*sc.cosh(2*K) + eps1(W)*sc.sinh(2*K)) + eps0(W)*(r0*sc.tanh(K) + r1))
    
    
def Astar1(t):
    return W0p*T1(W0p)*sc.cos(W0p*t)/Dprime(W0p) + W0m*T1(W0m)*sc.cos(W0m*t)/Dprime(W0m)
    
def Astar2(t):
    return W0p*T2(W0p)*sc.cos(W0p*t)/Dprime(W0p) + W0m*T2(W0m)*sc.cos(W0m*t)/Dprime(W0m)
    

def vx(X, t):
    truth = np.array(np.abs(X) <= K*np.ones(len(X)))
    indices = np.where(truth == True)
    vxfunction = np.zeros(len(X), dtype=complex)
    for i in indices:
        vxfunction[i] = 
    truth2 = np.array(X < -K*np.ones(len(X)))
    indices2 = np.where(truth2 == True)
    for i in indices2:
        vxfunction[i] = 
    truth3 = np.array(X > K*np.ones(len(X)))
    indices3 = np.where(truth3 == True)
    for i in indices3:
        vxfunction[i] = 
    return vxfunction

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()