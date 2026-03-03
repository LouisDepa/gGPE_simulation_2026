# -*- coding: utf-8 -*-
"""
Operators used in environment.py

@author: Depaepe Louis
"""

import cupy as cp



@cp.fuse(kernel_name="pump")
def pump(
    psi_C: cp.ndarray, 
    inc_pump: cp.ndarray,
    gamma_C: float,
    X2:float,
    dt: float,
    t: float
) -> None:
    """
    Apply the pump to the photon field.
    Parameters
    ----------
    psi_C : cp.ndarray
        Photon cavity field.
    inc_pump : cp.ndarray
        Photon field pump (profil + amplitude).
    dt : float
        Time step.
    t : float
        Time.

    Returns
    -------
    None

    """

    psi_C+=cp.sqrt(gamma_C*(1-X2)/2)*inc_pump*dt*(1-cp.exp(-t/70))
    
    
@cp.fuse(kernel_name="losses")
def losses(
    psi: cp.ndarray, 
    dt: float, 
    gamma: cp.array
) -> None:
    """
    Apply losses for excitonic or photonic field.

    Parameters
    ----------
    psi : cp.ndarray
        Excitonic or photonic field.
    dt : float
        Time step.
    gamma : cp.array
        Losses.

    Returns
    -------
    None
    """
    psi *= cp.exp(-dt * .5 * gamma) 
    
@cp.fuse(kernel_name="saturation_C")
def saturation_C(psi_C: cp.ndarray, psi_X: cp.ndarray, dt: float, g_s: float) -> None:
    """
    Apply saturation/dissipative corrections to the excitonic component.

    Parameters
    ----------
    psi_C : cp.ndarray
        Photonic field.
    psi_X : cp.ndarray
        Excitonic field.
    dt : float
        Time step.
    g_s : float
        Saturation coefficient.
    """
    psi_X -= g_s * (cp.abs(psi_X)**2 / 2 * psi_X)


@cp.fuse(kernel_name="saturation_X")
def saturation_X(psi_C: cp.ndarray, psi_X: cp.ndarray, dt: float, g_s: float) -> None:
    """
    Apply cross-component saturation dynamics between photon and exciton fields.

    Parameters
    ----------
    psi_C : cp.ndarray
        Photonic field.
    psi_X : cp.ndarray
        Excitonic field.
    dt : float
        Time step.
    g_s : float
        Saturation coefficient.
    """
    psi_X -= g_s * (cp.abs(psi_X)**2 * psi_C - psi_X**2 * cp.conjugate(psi_C)) * dt


@cp.fuse(kernel_name="non_linearity")
def non_linearity(psi_X: cp.ndarray, X2: float, dt: float) -> None:
    """
    Apply the nonlinearity.

    Parameters
    ----------
    psi_X : cp.ndarray
        Excitonic field.
    X2 : float
        Initial excitonic fraction.
    dt : float
        Time step.
    """
    psi_X *= cp.exp(- 1j * cp.abs(X2 * psi_X)**2 * dt)


@cp.fuse(kernel_name="unitary_coupling")
def unitary_coupling(psi_C: cp.ndarray, psi_X: cp.ndarray, dt: float) -> None:
    """
    Evolve the linear Rabi coupling between photons and excitons.

    This function solves the coupled linear system:
    dC/dt = -i*Omega*X and dX/dt = -i*Omega*C

    Parameters
    ----------
    psi_C : cp.ndarray
        Photonic field.
    psi_X : cp.ndarray
        Excitonic field.
    dt : float
        Time step.
    """
    c = cp.cos(2 * dt)
    s = cp.sin(2 * dt)

    psi_C_old = psi_C.copy()
    psi_X_old = psi_X.copy()

    psi_C[:] = c * psi_C_old - 1j * s * psi_X_old
    psi_X[:] = c * psi_X_old - 1j * s * psi_C_old
    
@cp.fuse(kernel_name="coupling")
def coupling(
    psi_1: cp.ndarray,
    psi_2: cp.ndarray,
    dt: float,
) -> None:
    """
    Apply the coupling term.

    Parameters
    ----------
    psi_1 : cp.ndarray
        Field we want to coupled with field 2.
    psi_2 : cp.ndarray
        Field 2.
    dt : float
        Time step.

    Returns
    -------
    None
    """

    psi_1+= 2j * psi_2 * dt

@cp.fuse(kernel_name="detuning")
def detuning(
    psi: cp.ndarray,
    delta: float,
    dt: float
) -> None:
    """
    Apply the detuning term.

    Parameters
    ----------
    psi : cp.ndarray
        Field.
    delta : float
        Detuning of the excitonic or photonic branch.
    dt : float
        Time step.

    Returns
    -------
    None
    """
    
    psi*=cp.exp(- 1j * delta * dt)

    
@cp.fuse(kernel_name="propagate")
def propagate(
    psi_k: cp.ndarray,
    KX2: cp.array,
    KY2: cp.array,
    X2:float,
    dt: float
) -> None:
    """
    Apply the laplacian operator.

    Parameters
    ----------
    psi_k : cp.ndarray
        Photonic field in Fourier space.
    KX2 : cp.array
        Wave numbers along X axis at power 2.
    KY2 : cp.array
        Wave numbers along Y axis at power 2.
    dt : float
        Time step.

    Returns
    -------
    None
    """
    psi_k*=cp.exp(- 1j *(1-X2)* (KX2 + KY2) * dt)
    
    
    
@cp.fuse(kernel_name="noise")
def noise(psi: cp.ndarray, noise: cp.ndarray, X2:float, dt:float)-> None:
    """
    Correlated noise.

    Parameters
    ----------
    psi : cp.ndarray
        Input field to be modified by the noise.
    noise : cp.ndarray
        Noise array.
    X2 : float
        Initial excitonic fraction.
    dt : float
        Time step.
    Returns
    -------
    None
    """
    
    psi*=cp.exp(-1j*(1-X2)**.5*noise*dt)

