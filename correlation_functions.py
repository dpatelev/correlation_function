# goal - code to calculate 1D correlation functions for given potentials

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# CM-DVR functions
# generate grids - https://github.com/ScottHabershon/PS/blob/main/src/grids.py

def get_exact_grid(x_min: float, x_max: float, exact_grid_size: int) -> npt.NDArray:
    """
    Generates linearly space grid to be used in the exact solver.

    :param x_min: minimum value of the grid
    :type x_min: float
    :param x_max: maximium value of the grid
    :type x_max: float
    :param exact_grid_size: size of the grid used in the exact solver
    :type exact_grid_size: int
    :return: xgrid_exact
    :rtype: npt.NDArray
    """
    xgrid_exact = np.linspace(x_min, x_max, exact_grid_size)
    dx = xgrid_exact[1] - xgrid_exact[0]
    return xgrid_exact, dx

# generate 1D potentials
# 0.5kx^2 - harmonic oscillator

def harmonic_oscillator(k: float, grid: npt.NDArray):
    """
    Sets up a harmonic oscillator potential on a provided grid.

    Args:
        k: force/spring constant

    Returns:
        v[:]: 1-D potential energy, calculated at grid points.
    """
    v = 0.5 * k * (grid ** 2)
    return v

# solve Schrödinger eqn for potentials - DVR
# CM DVR taken from https://github.com/ScottHabershon/PS/blob/main/src/exact_solver.py

def colbert_miller_DVR(ngrid, x, m, v):
    """
    Performs Colbert-Miller DVR solution of 1-D Schrodinger equation.

    Args:
        ngrid: Number of grid points
        x[:]: Positions of grid points.
        m: mass of particle
        v[:]: 1-D potential energy, calculated at grid points.

    Returns:
        c[:,0]: The ground-state wavefunction.
    """

    #  Atomic units:
    hbar = 1.0

    # Set grid spacing.
    dx = x[1] - x[0]

    #  Set up potential energy matrix.
    V = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        V[i, i] = v[i]

    #  Set up kinetic energy matrix.
    T = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            if i == j:
                T[i, j] = ((hbar ** 2) * np.pi ** 2) / (6 * m * dx ** 2)
            else:
                T[i, j] = ((hbar ** 2) * (-1.0) ** (i - j)) / (m * dx ** 2 * (i - j) ** 2)

    # Create the Hamiltonian matrix:
    H = T + V

    #  Solve the eigenvalue problem using the linalg.eigh
    E, c = np.linalg.eigh(H)
    E = E.astype('float128',copy=False) # not needed for mac M1 systems

    #  Normalize each eigenfunction using simple quadrature.
    for i in range(ngrid):
        csum = np.trapz(np.conj(c[:,i]) * c[:,i], x)
        c[:, i] = c[:, i] / np.sqrt(csum)
        E[i] = np.real(E[i])

    return c, E, H

# calculate Kubo TCF from CM-DVR results
# TODO - FIX - this is incorrect, it should give different values for different timesteps
def Kubo_TCF(beta, grid_size, grid, E, c, dx, times):
    for i in range(0, grid_size):
        Z = 1 / np.exp(-beta * E[i])
        C_1 = np.exp(-beta * E[i])
        for j in range(0, grid_size):
            C_2 = np.exp(-1j*(E[i]- E[j])*times) # hbar = 1 - assume atomic units
            Aij = np.trapz(np.conj(c[:,i]) * grid[i] * c[:,j], dx = dx)
            Bji = np.trapz(np.conj(c[:,j]) * grid[j] * c[:,i], dx = dx)
            if i != j: # i == j -> 1-exp(0) = 0, AND div by 0 occurs
                C_3 = (1 - np.exp(-beta * (E[j]-E[i]))) / (E[j]-E[i])
    C = (1 / beta*Z) * C_1 * C_2 * Aij * Bji * C_3
    return C

# Classical MD functions
# energy and force for harmonic oscillator

def harmonic_energy_force(x,k):
    energy = 0.5*k*x**2
    force = -k*x # -ve derivative of energy

    return energy, force

# velocity verlet functions

def update_position(x,v,F,dt,m):
    x_new = x + v*dt + 0.5*m*F*dt*dt
    return x_new

def update_velocity(v,F_new,F_old,dt,m):
    v_new = v + 0.5*m*(F_old+F_new)*dt
    return v_new

# TODO - change function so it only does one timestep
def velocity_verlet(potential, max_time, dt, initial_position, initial_velocity, m, save_frequency=1):
    x = initial_position
    v = initial_velocity # p = mv
    t = 0
    m = m
    step_number = 0
    positions = []
    velocities = []
    total_energies = []
    save_times = []
    
    while(t<max_time):
        potential_energy, force = potential(x,k)
        if step_number%save_frequency == 0:
            e_total = 0.5*v*v + potential_energy

            positions.append(x)
            velocities.append(v)
            total_energies.append(e_total)
            save_times.append(t)
        
        x = update_position(x,v,force,dt,m)
        potential_energy2, force2 = potential(x,k)
        v = update_velocity(v,force2,force,dt,m)
                
        t = t+dt
        step_number = step_number + 1
    
    return save_times, positions, velocities, total_energies

# calculating position auto correlation function for 1 trajectory
def position_auto_correlation_function():
    correlation_function = []

    for t in range(len(times)):
        cf_t = positions[0] * positions[t]
        correlation_function.append(cf_t)
    return correlation_function

# TODO - loop over a number of traj for harmonic oscillator

# TODO - for each traj generate intial conditions and run
# TODO - MD timestep - function for ONLY ONE TIMESTEP, and lopp for required timesteps
    # TODO - generate initial conditions from constant temperature ensemble
    # Run initial MD trajectory with a thermostat (Andersen) attached
        # Sample initial velocity/momenta of particle
        # Calculate initial forces on particle
        # Equilibrate for set number of timesteps - Andersen thermostat
        # Resample initial velocities/momenta

    # Run MD trajectory and get position
    # Accumulate correlation function
    # Average correlation function - divide by no of trajectories

# TODO - calculate overall ensemble average - average over lots of diff trajectories 

# initialise variables
k = 1
m = 1

initial_energy = 2
initial_position = 0.1
initial_velocity = 1

dt = 0.1
max_time = 20

# After equilibration period, run constant-energy simulation to accumulate the correlation function, which is repeated for lots of initial conditions
# loop for nt timesteps after VV func changed
times, positions, velocities, total_energies = velocity_verlet(harmonic_energy_force, max_time, dt, initial_position, initial_velocity, m)

# loop over time values and calculate the correlation function
C_t = position_auto_correlation_function() # C_t is array containing correlation function
print(C_t)

plt.plot(times,C_t)
plt.plot(times,positions)
plt.plot(times,total_energies)
plt.savefig('file.png')

# run CM DVR
x_min = -5
x_max = 5
grid_size = 51

grid, dx = get_exact_grid(x_min,x_max,grid_size)
v = harmonic_oscillator(k,grid)

# returns c - ground state wavefunction - ith column of c contains wavefunction phi(i), E - eigenvalues, H - hamiltonian, of the system
c, E, H = colbert_miller_DVR(grid_size, grid, m, v)

t = np.arange(0,max_time, 0.1) # same times used in classical MD
beta = 1
C = Kubo_TCF(beta, grid_size, grid, E, c, dx, t)