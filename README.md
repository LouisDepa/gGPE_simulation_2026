# gGPE_simulation

This repository contains a **GPU-accelerated simulation framework** for the **driven–dissipative coupled exciton–photon Gross–Pitaevskii equations** describing **two-dimensional polariton quantum fluids**. The time evolution is performed using a **split-step Fourier method** with **spectral propagation**, **dealiasing**, and **correlated noise**, enabling the study of turbulence in large-scale systems.

The implementation follows the numerical model and simulation strategy used in:

**L. Depaepe et al. (2026)**  
*Emergence of Turbulence in a Counterflow Geometry of 2D Polariton Quantum Fluids*

and is designed to reproduce the dynamical regimes explored in that work.

---

## Main characteristics

- **Coupled exciton–photon Gross–Pitaevskii equations**
  - Photon field ψ₍C₎ and exciton field ψ₍X₎
  - Rabi coupling, detuning, decay, interactions, and coherent pumping
- **Split-step Fourier propagation**
  - Real-space nonlinear, loss, pump, noise, and coupling operators
  - Momentum-space kinetic propagation
  - Spectral dealiasing via zero-padding
- **GPU acceleration**
  - All fields and operators handled with **CuPy**
  - FFTs performed using `cupyx.scipy.fftpack`
- **Optional CPU FFT planning**
  - FFTW wisdom support via `pyFFTW` for hybrid workflows
- **Driven–dissipative dynamics**
  - Coherent pump profiles
  - Cavity and exciton losses
  - Absorbing boundaries (super-Gaussian damping)
- **Noise and disorder**
  - Gaussian-correlated noise generation
  - Optional external potential
- **Multiple models**
  - `GP_coupled`: full exciton–photon model  --> used in the reference
  - `GP_coupled_sat`: exciton saturation effects (In progress) 
  - `GP_LP`: effective lower-polariton equation
- **Data output**
  - Time-resolved fields saved as `.npy`
  - Automatic folder structure based on physical parameters

---

## Structure

- **`environment.py`**  
  Core of the simulation framework.  
  Define the main time-evolution loop implementing the split-step Fourier scheme.

- **`operators.py`**  
  Collection of all elementary operators acting on the fields.  
  Implements pumping, losses, detuning, nonlinear interactions, saturation effects, unitary exciton–photon coupling, noise, and kinetic propagation.  
  All operators are written using **CuPy** for GPU execution.

- **`launcher.py`**  
  Entry point for running simulations.  
  Defines tunable physical and numerical parameters (pump power, detuning, losses, grid size, time step, model choice, etc.) and initializes the simulation environment before launching time evolution.

- **`plot.py`**  
  Post-processing and visualization utilities.  
  Provides example plots and animations of the field dynamics (e.g. density and phase evolution as a function of time) from the saved simulation data.

---

## Typical applications

- Counterflow polariton geometries  
- Vortex nucleation and dynamics  
- Turbulence emergence in 2D quantum fluids  
- Parameter-space exploration for experiments  

---

## Requirements

- Python ≥ 3.9  
- CUDA-enabled GPU  
- `cupy`
- `numpy`
- `cupyx`
- `pyfftw`
- `alive-progress`

---

## Scope and intent

This code is intended for **research, reproducibility, and large-scale numerical studies** of polariton quantum fluids. It prioritizes physical fidelity, numerical stability, and performance over general-purpose usability.

----

## Reference

If you use this code, please cite:

> L. Depaepe *et al.*, *Emergence of Turbulence in a Counterflow Geometry of 2D Polariton Quantum Fluids* (2026).
