# -*- coding: utf-8 -*-
"""
Launcher of the simulation

@author: Depaepe Louis
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r"YourPath")
import environment

#########################
#Simulation parameters
#########################

folder=r"YourPath_SaveFolder"


Lx=256                  # Size of the cavity in x_axis (px)
Ly=256                  # Size of the cavity in x_axis (px)
N=256*2                 # Space step of the grip 
psi0=np.zeros((N,N))    # Initial cavity: empty cavity
E_C = 0                 # Energy of the photon at k=0 (meV)
E_X = 0                 # Energy of the exciton (meV)
rabi_mev=3.3            # Rabi splitting 2 * hbar * Omega_R (meV)
sigma_noise=0.36        # Noise correlation length (adim)
amp_noise=0.00146       # Noise amplitude (adim)
sample=10               # Sample time (adim)
detuning_LP= 0.4        # Laser detuning from the lower polariton branch at k=0 (adim)
P= 1                    # Pump amplitude (adim)
tf=2000                 # Final time of the simulation (adim)
dt=1e-3                 # Time step of the simulation (adim)
gamma=0.02              # Losses rate (adim)



#########################
#Pump parameters
#########################

k0x=0.4                 # In-plane momentum in x-axis (adim)
k0y=0                   # In-plane momentum in x-axis (adim)
k=(k0x,k0y) 
sigma0x=25              # Uncut Gaussian pump spot along the x direction (adim)
sigma0y=20              # Uncut Gaussian pump spot along the y direction (adim)
sigma_distance=40       # Characteristic length scale of the smooth cutting function (adim)
distance = 15           # Half of the separation between the two lobes (adim) !Not strictly the distance between the pumps


#%% Start the simulation



delta=detuning_LP + (4/rabi_mev)*(.5*(E_C+E_X-np.sqrt((E_C-E_X)**2+rabi_mev**2)))
x=y=np.linspace(-Lx/2, Lx/2, N, endpoint=0)
X,Y=np.meshgrid(x,y)

psi= np.exp(-((X)**2/(2 * sigma0x ** 2)+Y**2/ (2 * sigma0y ** 2)))
psi-= np.exp(-(abs(X)**3/(2 * sigma_distance ** 2)+Y**2/ (2 * sigma0y ** 2)))
psi=np.array(psi, dtype=complex)

psi01=np.pad(psi[:,distance:int(np.shape(psi)[0]/2)], ((0,0),(0,int(np.shape(psi)[0]/2+distance))))
psi02=np.pad(psi[:,int(np.shape(psi)[0]/2):-distance], ((0,0),(int(np.shape(psi)[0]/2)+distance,0)))
pump_profile=psi01*np.exp(1j * k0x * X) + psi02*np.exp(1j * -k0x * X)
pump_profile/=np.max(np.abs(pump_profile))

env=environment.SIMULATION((Lx,Ly), 
folder,                             # Folder of save
delta,                              # Detuning
gamma,                              # Losses rate for photon
gamma,                              # Losses rate for exciton
0,                                  # Saturaton coefficient (only for a specific model)
psi0,                               # Initial photonic field 
psi0,                               # Initial excitonic field 
pump_profile,                       # Profile of the pump
pump_pow=P,                         # Pump amplitude
absorb_border=True,                 # Presence of absorbing boundaries
k=k,                                # In-plane momenta of the pump (for labelling) 
sigma_noise=float(sigma_noise),     # Noise correlation length
amp_noise=float(amp_noise),         # Noise amplitude
model='GP_coupled'                  # Model of simulation:
                                    # - GP_LP: Basic GP equation of lower polariton only
                                    # - GP_coupled: Coupled gGPE exciton-photon
                                    # - GP_coupled_sat: In progress
)

env.evolution(tf,sample=sample, dt=dt)


 