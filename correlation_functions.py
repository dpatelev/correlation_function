# goal - code to calculate 1D correlation functions for given potentials

import numpy as np
import numpy.typing as npt

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

# Performing classical MD
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

    for t in range(len(positions)):
        cf_t = positions[0] * positions[t]
        correlation_function.append(cf_t)
    return correlation_function

# initialise variables
k = 3
m = 2

omega = np.sqrt(k/m)
tau = 2*np.pi/omega
dt = tau/100.

initial_energy = 1
initial_position = 2
initial_velocity = 0

max_time = 10

# run MD

times, positions, velocities, total_energies = velocity_verlet(harmonic_energy_force, max_time, dt, initial_position, initial_velocity, m)

Cxx = position_auto_correlation_function() # Cxx is array containing correlation function
print("TCF using classical MD = ", sum(Cxx), "with max_time:", max_time) # sum of correlation function array. Ran MD at max_time = 1000000, calc. TCF = 24.260024382957504

# TODO - eventually, loop over a number of traj for harmonic oscillator
# TODO - for each traj generate intial conditions and run
    # TODO - generate initial conditions from constant temperature ensemble.
    # Run initial MD trajectory with a thermostat (Anderson) attached. After equilibration period, run constant-energy simulation to accumulate the correlation function, which is repeated for lots of initial conditions
    # will get position and momentum as a function of time.
    # loop over set time values and calculate the correlation function at these times - MD code already does this, using save_frequency and step_number.
# TODO -calculate overall ensemble average - average over lots of diff trajectories


# run CM DVR
x_min = -5
x_max = 5
grid_size = 51

grid, dx = get_exact_grid(x_min,x_max,grid_size)
v = harmonic_oscillator(k,grid)

# returns c - ground state wavefunction - ith column of c contains wavefunction phi(i), E - eigenvalues, H - hamiltonian, of the system
c, E, H = colbert_miller_DVR(grid_size, grid, m, v)

# use the CM-DVR results to calculate the exact position auto correlation function. The operator is purely the position of the particle!

def Kubo_TCF(beta, grid_size, grid, T, E, c, dx, t):
    for i in range(0, grid_size):
        Z = 1 / np.exp(-beta * E[i])
        C_1 = np.exp(-beta * E[i])
        for j in range(0, grid_size):
            C_2 = np.exp(-1j*(E[i]- E[j])*t) # hbar = 1 - assume atomic units
            Aij = np.trapz(np.conj(c[:,i]) * grid[i] * c[:,j], dx = dx)
            Bji = np.trapz(np.conj(c[:,j]) * grid[j] * c[:,j], dx = dx)
            if i != j: # i == j -> 1-exp(0) = 0, AND div by 0 occurs
                C_3 = (1 - np.exp(-beta * (E[j]-E[i]))) / (E[j]-E[i])
        C = (1 / beta*Z) * C_1 * C_2 * Aij * Bji * C_3
    return C

t = 0 # loop over series of t from 0 to a maximum - times at which we want to calculate the TCF.
T = 298.15 # assumed RT
k_b = 8.61733262*np.exp(-5) # get from scipy constants
beta = 1/(k_b*T)

C = Kubo_TCF(beta, grid_size, grid, T, E, c, dx, t)
print("Kubo TCF = ", C) # Kubo TCF for these conditions == 25.121253139611380905