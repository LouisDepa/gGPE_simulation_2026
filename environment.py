# -*- coding: utf-8 -*-
"""
Environment of the coupled exciton-photon equations

@author: Depaepe Louis
"""

import numpy as np
import cupy as cp
import pyfftw
import multiprocessing
from alive_progress import alive_bar
from Dealiasing import PAD, UNPAD
import os
import cupyx.scipy.fftpack as fftpack
from cupyx.scipy.fftpack import fft2, ifft2
from cupyx.scipy.ndimage import gaussian_filter

try:    
    from . import operators
except:
    import operators


pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()



class SIMULATION:
    def __init__(self, 
            dim:tuple, 
            folder:str, 
            omega_laser:float, 
            gamma_C:float, 
            gamma_X:float,
            g_s: float,
            psi0_c: np.array,
            psi0_x:np.array,
            pump_profile:np.array, 
            pump_pow:float=0,
            absorb_border:bool=False, 
            k:tuple=(0,0), 
            sigma_noise=0, 
            amp_noise=0,
            potential=0,
            model:str=r'GP_coupled'):
        """
        To solve driven-dissipative Non-Linear Schrodînger equation dimensionless.

        Parameters
        ----------
        dim: tuple
            Dimensions of the space.
        folder : str
            The folder where you want to save data.
        delta : float
            Detuning Δ between polariton resonance and laser frequency.
        gamma : float
            Loss of the cavity γ.
        g_S : float
            Saturation coefficient as a ratio g_s/g_x
        psi0 : np.array
            Initial photon field.
        pump_profile : np.array
            The pump profile of the laser.
        pump_pow : float, optional
            The power of the pump P.
        disorder_scale : float, optional
            The variability of a normal distribution increases when additional disorder is introduced.
        absorb_border : bool, optional
            Add damping. The default is False.
        k : tuple, optional
            The wavenumbers of the pump. The default is (0,0).

        Returns
        -------
        None.

        """
        self.Lx = dim[0]
        self.Ly = dim[1]
        self.omega_L = omega_laser
        self.E_C = 0.5*0 #meV
        self.E_X =4.9*0 #meV
        self.rabi_mev=3.3 #meV
        self.X2=np.round(1/2*(1+(self.E_C-self.E_X)/np.sqrt((self.E_C-self.E_X)**2+self.rabi_mev**2)),2)
        self.detuning_LP =-( (4/self.rabi_mev)*(.5*(self.E_C+self.E_X-np.sqrt((self.E_C-self.E_X)**2+self.rabi_mev**2)))-self.omega_L)
        self.delta_X = self.E_X*4/self.rabi_mev-self.omega_L
        self.delta_C = self.E_C*4/self.rabi_mev-self.omega_L
        self.gamma_C = gamma_C
        self.gamma_X = gamma_X
        self.pump_pow=pump_pow
        self.inc_pump = cp.array(pump_profile * pump_pow,dtype=cp.complex128)
        self.psi0_c = cp.array(psi0_c,dtype=cp.complex128)
        self.psi0_x = cp.array(psi0_x,dtype=cp.complex128)
        self.k=k
        self.folder= folder   
        self.operators = operators
        
        N = self.psi0_c.shape[0]
        self.N = N
        
        self.x, self.dx = cp.linspace(-self.Lx/2, self.Lx/2, N, retstep=True)
        self.y, self.dy = cp.linspace(-self.Ly/2, self.Ly/2, N, retstep=True)

        self.kx =( 2 * cp.pi * cp.fft.fftfreq(2*self.N, d=self.dx))
        self.ky =( 2 * cp.pi * cp.fft.fftfreq(2*self.N, d=self.dy))

        self.X, self.Y = cp.meshgrid(self.x, self.y)
        self.KX, self.KY = cp.meshgrid(self.kx, self.ky)
        

        self.sigma_W=sigma_noise #0.36 adim = 0.5 µm
        self.W0=amp_noise #0.073 adim = 0.06meV
        cp.random.seed(100000)

        noise=cp.random.normal(0, 1, (self.N,self.N))
        W = cp.zeros((self.N,self.N))
        gaussian_filter(noise, sigma=self.sigma_W, output=W)
        noise = W / cp.std(W) * self.W0
        self.noise= noise
        self.potential=cp.array(potential)

        
        if absorb_border:
            SuperGaussianPowerAbs = 20
            self.gamma_C = self.gamma_C*np.exp((self.X**SuperGaussianPowerAbs+self.Y**SuperGaussianPowerAbs)/cp.mean(self.X**SuperGaussianPowerAbs+self.Y**SuperGaussianPowerAbs))#20 * (cp.abs(1 - cp.exp(-((cp.abs(self.X) / (0.5 * (N/(self.Lx/self.Ly)))) ** SuperGaussianPowerAbs))) + cp.abs(1 - cp.exp(-((cp.abs(self.Y) / (0.5 * N)) ** SuperGaussianPowerAbs)))) + self.gamma_C * cp.ones((N, N))
            #self.gamma_C += cp.random.normal(loc=0, scale=self.gamma_C*0.1, size=(N,N))
        
        if not model in ['GP_coupled', 'GP_coupled_sat', 'GP_LP']:
            print('The model is not define.')
        else: self.model=model

        me= 9.1e-31
        m_photon = me*2.4e-5 #Lausanne: 3.9e-5  #kg
        c=3e8 #m.s^-1
        hbar = 6.626e-34 / (2 * np.pi) # J.s
        rabi =3.3 / (2 * hbar)*1.6e-22#Lausanne: 4.7 / (2 * hbar)*1.6e-22 # s^-1
        g = .03 *1e-36 /(6.591004080793116e-13) # s-1.m²
        m = m_photon  #kg
        self.rabi = rabi
        self.hbar = hbar
        self.mas = m
        self.g = g 
        self.tau0 = 2 / rabi
        self.psi_alpha = cp.sqrt(rabi / (2 * g))
        self.space = cp.sqrt(hbar / (m*rabi))
        self.alpha_pump = rabi * cp.sqrt(1 / g) / 2
        self.g_s=g_s #g_s/g_x
  
        
    def build_fft_plans(self, A: cp.ndarray):
        """Builds the FFT plan objects for propagation, will transform the last two axes of the array
    
        Args:
            A (np.ndarray): Array to transform.
        Returns:
             FFT plans
        """
        try: 
            plan = np.load(f'plans_{self.N}.npy')
            pyfftw.import_wisdom(plan)
            plan_fft = fftpack.get_fft_plan(
                A,axes=(-2, -1),
                value_type="C2C",
            )
        except:
            plan_fft = fftpack.get_fft_plan(
                A,axes=(-2, -1),
                value_type="C2C",
            )
            plan=pyfftw.export_wisdom()
            np.save(f'plans_{self.N}',plan)
        return plan_fft
    


    def evolution(self, T_final: float, sample: float = 0.5, dt: float = 0.001):
        """
        This function saves the field in the cavity at each sample into a single file.
        
        Parameters
        ----------
        T_final : float
            Final time of the simulation.
        sample : float, optional
            Number of time between saves. The default is .5.
        dt : float, optional
            Time step in the resolution process. The default is .001.
    
        Returns
        -------
        None.
        """
        save_interval = int(T_final / sample) + 1
        
    

    
        if self.model != 'GP_LP':
            Psi_C = cp.empty((save_interval, self.N, self.N), dtype=cp.complex128)
            Psi_X = cp.empty((save_interval, self.N, self.N), dtype=cp.complex128)
            psi_X = self.psi0_x 
            psi_C = self.psi0_c 
            folder_name = os.path.join(self.folder,fr'{self.model}\X={self.X2}\dt={dt}_P={np.round(self.pump_pow, 5)}_k={self.k}_tf={T_final}_laser={np.round(self.detuning_LP, 5)}')
 
        else:
            Psi = cp.empty((save_interval, self.N, self.N), dtype=cp.complex128)
            psi = cp.array(self.psi0, dtype=cp.complex128)
            folder_name = os.path.join(self.folder,fr'{self.model}\dt={dt}_P={np.round(self.pump_pow, 5)}_k={self.k}_tf={T_final}_laser={np.round(self.detuning_LP, 5)}')
        

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    
        times = []
        T = 0
        KX_2 = self.KX ** 2
        KY_2 = self.KY ** 2
        a = 0

    

        #plan_fft = self.build_fft_plans(PAD(psi_X))
        
        if self.model =='GP_coupled_sat':
            
            with alive_bar(save_interval) as bar:
                while T < T_final:
                    
                    self.operators.pump(psi_C, self.inc_pump, dt, T)
                    self.operators.losses(psi_C, dt, self.gamma_C)
                    self.operators.losses(psi_X, dt, self.gamma_X)
                    self.operators.saturation_C(psi_C, psi_X, dt, self.g_s)
                    self.operators.saturation_X(psi_C, psi_X, dt, self.g_s)
                    self.operators.non_linearity(psi_X, dt)
                    self.operators.detuning(psi_C, self.delta_C, dt)
                    self.operators.detuning(psi_X, self.delta_X, dt)
                    self.operators.noise(psi_C, self.noise,dt)
                    self.operators.unitary_coupling(psi_C, psi_X, dt)

                
                    psi_C = PAD(psi_C)
                    psi_C = fft2(psi_C)
                    self.operators.propagate(psi_C, KX_2, KY_2, dt)
                    psi_C = ifft2(psi_C)
                    psi_C = UNPAD(psi_C)
                    
                    
                    if np.round(T,5) % sample == 0:
                        
                        Psi_C[a] = psi_C.copy()
                        Psi_X[a] = psi_X.copy()
                        times.append(T)
                        bar()
                        a += 1
                    
                    T += dt
                    
            cp.save(os.path.join(folder_name, 'Psi_C.npy'), Psi_C)
            cp.save(os.path.join(folder_name, 'Psi_X.npy'), Psi_X)
            np.save(os.path.join(folder_name, 'times.npy'), cp.asnumpy(cp.array(times)))
            print('Process done!')
        elif self.model =='GP_coupled':
            with alive_bar(save_interval) as bar:
                while T < T_final:
                    
                    self.operators.pump(psi_C, self.inc_pump,self.gamma_C, 0, dt, T)
                    self.operators.losses(psi_C, dt, self.gamma_C)
                    self.operators.losses(psi_X, dt, self.gamma_X)

                    self.operators.non_linearity(psi_X, 1,dt)
                    self.operators.detuning(psi_C, self.delta_C, dt)
                    self.operators.detuning(psi_X, self.delta_X, dt)
                     
                    self.operators.unitary_coupling(psi_C, psi_X, dt)
                    self.operators.noise(psi_C, self.noise+self.potential,0,dt)

                    psi_C = PAD(psi_C)
                    psi_C = fft2(psi_C)
                    self.operators.propagate(psi_C, KX_2, KY_2,0, dt)
                    psi_C = ifft2(psi_C)
                    psi_C = UNPAD(psi_C)
                    
 
                    if np.round(T,5) % sample == 0:
                        
                        Psi_C[a] = psi_C.copy()
                        Psi_X[a] = psi_X.copy()
                        times.append(T)
                        bar()
                        a += 1
                    
                    T += dt
            cp.save(os.path.join(folder_name, 'Psi_C.npy'), Psi_C)
            cp.save(os.path.join(folder_name, 'Psi_X.npy'), Psi_X)
            np.save(os.path.join(folder_name, 'times.npy'), cp.asnumpy(cp.array(times)))
            print('Process done!')
                    
        else:
            with alive_bar(save_interval) as bar:
                while T < T_final:
     
                    
                    self.operators.pump(psi, self.inc_pump, self.gamma_C,self.X2,dt, T)
                    self.operators.losses(psi, dt, self.gamma_C)

                    self.operators.non_linearity(psi,self.X2, dt)
                    self.operators.detuning(psi, -1*self.detuning_LP, dt)
                    self.operators.noise(psi, self.noise,self.X2, dt)

                                                        
                    psi = PAD(psi)
                    psi= fft2(psi)
                    self.operators.propagate(psi, KX_2, KY_2,self.X2, dt)

                    psi= ifft2(psi)
                    psi= UNPAD(psi)
                    
                    
                    if np.round(T,5) % sample == 0:
                        
                        Psi[a] = psi.copy()
                        times.append(T)
                        bar()
                        a += 1
                    
                    T += dt
        
        

            cp.save(os.path.join(folder_name, 'Psi.npy'), Psi)
            np.save(os.path.join(folder_name, 'times.npy'), cp.asnumpy(cp.array(times)))
            np.save(os.path.join(folder_name, 'noise.npy'), cp.asnumpy(cp.array(self.noise)))
            # with open(os.path.join(folder_name, 'self.pkl'), 'wb') as f:
            #     pickle.dump(self, f)
            #with open('chemin/vers/ton/fichier/self.pkl', 'rb') as f:
            #    obj = pickle.load(f)
            print('Process done!')
        
        