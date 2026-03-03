# -*- coding: utf-8 -*-
"""
Exemple of plotting code

@author: Depaepe Louis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re

#%% Coupled

N=256*2
Lx=256
Ly=256

file = r"YourPath"
C=np.load(file+r'\Psi_C.npy')
X=np.load(file+r'\Psi_X.npy')
times=np.load(file+r'\times.npy')
Psi=C

#%% GP

N=256*2
Lx=256
Ly=256

file = r"YourPath"
Psi=np.load(file+r'\Psi.npy')
times=np.load(file+r'\times.npy')


#%% Animation
import matplotlib.patches as patches
fig, ax = plt.subplots(1,2)



Phase = ax[0].imshow(np.angle(Psi[0]),extent=[-Lx/2,Lx/2,-Ly/2,Ly/2],cmap='twilight',aspect='equal')
ax[0].set_title('Phase')
ax[0].set_xlabel('Position X')
ax[0].set_ylabel('Position Y')
Phase_colorb=fig.colorbar(Phase, ax=ax[0],shrink=0.5)
Phase.set_clim(vmin=-np.pi,vmax=np.pi)

wavefunction_plot = ax[1].imshow(np.abs(Psi[0])**2, cmap='gray',extent=[-Lx/2,Lx/2,-Ly/2,Ly/2])
ax[1].set_title(r'$|\Psi|^2$')
ax[1].set_xlabel('Position X')
fig.colorbar(wavefunction_plot, ax=ax[1], shrink=0.54)


coef_time=1
def update(frame):
    new_data=(Psi[coef_time*frame])
    wavefunction_plot.set_data(np.abs(new_data)**2)
    Phase.set_data(np.angle(new_data))
    time=np.round(times[frame*coef_time])
    fig.suptitle(f'Time: {time} $T$')
    
    wavefunction_plot.set_clim(0, np.max(abs(new_data)**2))
    fig.canvas.draw_idle()

    return wavefunction_plot,Phase,

ani = FuncAnimation(fig, update, frames=range(int(np.shape(Psi)[0]/(coef_time))), blit=True, repeat=0)
