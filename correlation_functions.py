# goal - code to calculate 1D correlation functions for given potentials

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

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
    return xgrid_exact

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
        m: m of particle
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

    #  Normalize each eigenfunction using simple quadrature.
    for i in range(ngrid):  # ONLY FOR FIRST EIGENFUNCTION !!!!!
        # csum = 0.0
        # for j in range(ngrid):
        #    csum += c[j, i] * c[j, i] * dx
        # print("csum1 = ",csum)
        # c[:, i] = c[:, i] / np.sqrt(csum)
        csum = np.trapz(np.conj(c[:,i]) * c[:,i], x)
        #print("csum = ",csum,dx)
        c[:, i] = c[:, i] / np.sqrt(csum)
        E[i] = np.real(E[i])

    return c, E, H

# energy and force for harmonic oscillator

def energy_force(x,k):
    energy = 0.5*k*x**2
    force = -k*x

    return energy, force

# velocity verlet

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

# actually run MD

# initial conditions

initial_energy = 1
initial_position = 2
initial_velocity = 0
max_time = 10

times, positions, velocities, total_energies = velocity_verlet(energy_force, max_time, dt, initial_position, initial_velocity, m)

Cxx = position_auto_correlation_function()

# #plot position and auto correlation function
plt.figure()
plt.plot(times,positions,marker='o',linestyle='')
plt.plot(times,Cxx, marker='*',linestyle='')
plt.xlabel('time')
plt.savefig('file')

# TODO - eventually, loop over a number of traj
# TODO - for each traj generate intial conditions and run
    # will get position and momentum as a function of time.
    # TODO - how to generate initial conditions
# TODO -calculate overall ensemble average - average over lots of diff trajectories


# run CM DVR
x_min = -5
x_max = 5
grid_size = 51

grid = get_exact_grid(x_min,x_max,grid_size)
v = harmonic_oscillator(k,grid)

# returns c - ground state wavefunction, E - eigenstates(?), H - hamiltonian, of the system
c, E, H = colbert_miller_DVR(grid_size,grid, m, v)

# TODO - use the CM-DVR results to calculate the exact position auto correlation function. The operator is purely the position of the particle!