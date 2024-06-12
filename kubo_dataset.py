# Goal - to generate a dataset of exact Kubo TCF's for a variety of randomly generated BOUND potentials, to be used as training data to compare against classical MD results.

import sys
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
    :return: dx
    :rtype: float
    """
    xgrid_exact = np.linspace(x_min, x_max, exact_grid_size)
    dx = xgrid_exact[1] - xgrid_exact[0]
    return xgrid_exact, dx

# generate 1D potentials

def potential(grid: npt.NDArray):
    """
    Sets up a potential on a provided grid.

    Args:
        grid[:]: provided grid.

    Returns:
        v[:]: 1-D potential energy, calculated at grid points.
    """
    v = 0.25 * grid**4 # Manolopoulos (2004)
    # v = 0.5*(grid**2) + 0.1*(grid**3) + 0.01*(grid**4) # Manolopoulos (2004)
    return v

# solve Schrödinger eqn for potentials
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
        E: The eigenvalues of the system.
        H: The Hamiltonian of the system.
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
    if not sys.platform.startswith('darwin'): # not required for mac OS, # TODO 17.06 / 18.06 - check if the float requirement is really needed on Ubuntu
        E = E.astype('float128',copy=False)

    #  Normalize each eigenfunction using simple quadrature.
    for i in range(ngrid):
        csum = np.trapz(np.conj(c[:,i]) * c[:,i], x)
        c[:, i] = c[:, i] / np.sqrt(csum)
        E[i] = np.real(E[i])

    return c, E, H

# calculate Kubo TCF from CM-DVR results

def Kubo_TCF(grid, E, c, dx, times, range_E, beta=1):
    """
    Calculates the Kubo transformed correlation function on the given grid, with the given eigenvalues and wavefunctions.

    Args:
    grid: linearly spaced grid
    E: array of eigenvalues of the system
    c[;0]: the groundstate wavefunction of the system
    dx: grid spacing
    times: array of timesteps
    range_E: range of eigenvalues to calculate the TCF with
    beta: inverse temperature (1/k_b*T)

    Returns:
    C: The Kubo TCF calculated at each timestep in times

    """
    # calculate and accumulate the partition function for the entire system
    Z = 0.0
    for i in range(0, len(E)):
        Z += np.exp(-beta * E[i])

    nt = len(times)
    C = np.zeros(nt, dtype = 'complex') # create empty array of length times for TCF

    # calculate and accumulate individual TCF terms over the eigenvalues i,j
    for i in range(range_E):
       for j in range(range_E):
            t1 = np.exp(-beta*E[i])
            t2 = np.exp(-1j * (E[i]- E[j]) * times) # hbar = 1 - assume atomic units
            Aij = np.trapz(np.conj(c[:,i]) * grid[:] * c[:,j], dx = dx) # integrate over ALL grid points!
            Bji = np.trapz(np.conj(c[:,j]) * grid[:] * c[:,i], dx = dx) # grid[:] == the operator
            if i != j: 
                t3 = (1 - np.exp(-beta * (E[j]-E[i]))) / (E[j]-E[i])
            else: # limit of (1 - exp(-beta * x)) / x = beta
                t3 = beta
            C +=  (t1 * t2 * Aij * Bji * t3)
    C /= (beta * Z)
    C = C.real 
    return C

# initialise variables
m = 1

dt = 0.1
max_time = 20
t = np.arange(0,max_time, dt)

# run CM DVR
x_min = -10
x_max = 10
grid_size = 101

grid, dx = get_exact_grid(x_min,x_max,grid_size)
v = potential(grid)

# returns c - ground state wavefunction - ith column of c contains wavefunction phi(i), E - eigenvalues, H - hamiltonian, of the system
c, E, H = colbert_miller_DVR(grid_size, grid, m, v)

# set range of eigenvalues to use when calculating TCF - can use all (set to len(E))!
range_E = 10
C = Kubo_TCF(grid, E, c, dx, t, range_E, beta=1)

plt.plot(t, C)
plt.savefig('Kubo.png')

# TODO - randomly generate a set of BOUND potentials and calculate the TCF for all of them
    # how to bound them?
    # how to randomly generate - what parameters are required for random generation? General format for potential?

# save Kubo TCF data to file - also potential data to file
data = np.column_stack((t,C))
np.savetxt('output.dat', data, fmt=('%5.2f', '%5.10f'), header='t\tC')