# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:24:44 2020

@author: Matthew Allcock
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# set global font size
plt.rcParams.update({'font.size': 12})

# set size of intially perturbed region
z0 = 1

# set characteristic speeds
v0 = 2
vp = 1
vm = 0.5
vs = sc.sqrt((vm**2 + vp**2)/2)

# vx function
def vx(x, z, t):
    def tan1(z, x, t, z00, v):
        return sc.arctan((z + z00 + v*t)/np.abs(x))
    
    vxm = (v0/(4*np.pi*z0)*(- tan1(z, x, t, z0, vm) - tan1(z, x, t, z0, -vm) 
                            + tan1(z, x, t, -z0, vm) + tan1(z, x, t, -z0, -vm)
                            + tan1(z, x, t, z0, vs) + tan1(z, x, t, z0, -vs)
                            - tan1(z, x, t, -z0, vs) - tan1(z, x, t, -z0, -vs)
                            + np.pi*(np.heaviside(z + z0 + vm*t, 0.5)
                            + np.heaviside(z + z0 - vm*t, 0.5)
                            - np.heaviside(z - z0 + vm*t, 0.5)
                            - np.heaviside(z - z0 - vm*t, 0.5))))
    
    vxp = (v0/(4*np.pi*z0)*(- tan1(z, x, t, z0, vp) - tan1(z, x, t, z0, -vp) 
                            + tan1(z, x, t, -z0, vp) + tan1(z, x, t, -z0, -vp)
                            + tan1(z, x, t, z0, vs) + tan1(z, x, t, z0, -vs)
                            - tan1(z, x, t, -z0, vs) - tan1(z, x, t, -z0, -vs)
                            + np.pi*(np.heaviside(z + z0 + vp*t, 0.5)
                            + np.heaviside(z + z0 - vp*t, 0.5)
                            - np.heaviside(z - z0 + vp*t, 0.5)
                            - np.heaviside(z - z0 - vp*t, 0.5))))
    if x <= 0:
        vx = vxm
    else:
        vx = vxp
    return vx

# Domain
xmin = -1
xmax = -xmin
zmin = -3*z0
zmax = -zmin
Nx = 50
Nz = 100

x = np.linspace(xmin, xmax, Nx)
z = np.linspace(zmin, zmax, Nz)
X, Z = np.meshgrid(z, x)

tstart = 0
tend = 5
Nt = 6

t = np.linspace(tstart, tend, Nt)

# initialise vx array
vxvals = np.zeros((Nx, Nz, Nt))

# populate vx array
for i in range(Nx):
    for j in range(Nz):
        for k in range(Nt):
            vxvals[i, j, k] = vx(x[i], z[j], t[k])

# Levels for contour plot
levels = np.linspace(-v0*0.55, v0*0.55, 501)

label_font_size = 20

# plot contours
fig = plt.figure()
ax = plt.gca()
contour = plt.contourf(Z, X, vxvals[:,:,0], levels=levels, cmap='RdBu')
plt.xlabel(r'$x$', fontsize=label_font_size)
plt.ylabel(r'$z$', fontsize=label_font_size)
cbar = plt.colorbar()
cbar.set_label('Velocity', rotation=270, fontsize=label_font_size)
plt.tight_layout()

## Fuction to update animation at each time step
#def update(k):
#    ax.clear()
#    ax.contourf(Z, X, vxvals[:,:,k], levels=levels, cmap='RdBu')
#
## Animate!
#contour_ani = animation.FuncAnimation(fig, update, Nt, interval=20, blit=False)
#
## Save animation
#contour_ani.save('contour_animation.mp4', fps=24, extra_args=['-vcodec', 'libx264'])


#plt.figure


fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10,10))
contour = axs[0, 0].contourf(Z, X, vxvals[:,:,0], levels=levels, cmap='RdBu')
axs[0, 1].contourf(Z, X, vxvals[:,:,1], levels=levels, cmap='RdBu')
axs[1, 0].contourf(Z, X, vxvals[:,:,2], levels=levels, cmap='RdBu')
axs[1, 1].contourf(Z, X, vxvals[:,:,3], levels=levels, cmap='RdBu')
axs[2, 0].contourf(Z, X, vxvals[:,:,4], levels=levels, cmap='RdBu')
axs[2, 1].contourf(Z, X, vxvals[:,:,5], levels=levels, cmap='RdBu')

cbar = fig.colorbar(contour, ax=axs)
cbar.set_label('Velocity', rotation=270, fontsize=label_font_size, labelpad=10)

axs[2, 0].set_xlabel(r'$x$', fontsize=label_font_size)
axs[2, 1].set_xlabel(r'$x$', fontsize=label_font_size)
axs[0, 0].set_ylabel(r'$z$', fontsize=label_font_size)
axs[1, 0].set_ylabel(r'$z$', fontsize=label_font_size)
axs[2, 0].set_ylabel(r'$z$', fontsize=label_font_size)

plt.savefig('contour_subplots.png')
