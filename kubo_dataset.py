# Goal - to generate a dataset of exact Kubo TCF's for a variety of randomly generated BOUND potentials, to be used as training data to compare against classical MD results.

import sys, os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Tuple

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

# generate 1D potentials - taken from https://github.com/ScottHabershon/PS/blob/main/src/potentials.py

def potential_2004(grid: npt.NDArray):
    """
    Sets up potentials from Manolopolous (2004) on a provided grid and saves them to a file.

    Args:
        grid[:]: provided grid.
    """

    directory = 'potential/2004/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    v1 = 0.25 * grid**4 # Manolopoulos (2004)
    np.savetxt(f'{directory}potential_1.dat', np.column_stack((grid, v1)), fmt=('%5.10f'), header='x\tV = 0.25x**4')

    v2 = 0.5*(grid**2) + 0.1*(grid**3) + 0.01*(grid**4) # Manolopoulos (2004)
    np.savetxt(f'{directory}potential_2.dat', np.column_stack((grid, v2)), fmt=('%5.10f'), header='x\tV = 0.5x**2 + 0.1x**3 + 0.01x**4')
    return v1, v2

def polynomial(nord: int, xgrid: npt.NDArray, **kwargs) -> Tuple[npt.NDArray, npt.NDArray, int, float]:
    """
    Generates polynomials of a randomly generated order \in [2, `nord`+1], along the grid defined by `xgrid`.
    This function takes keyword argument `coeffs` - if one wishes to generate a series of precomputed polynomials,
    or `coeff_min`, `coeff_max`, `v_min`, `v_max` - for randomly generating new polynomials.

    :param nord: maximum order of on which the order of the polynomial is generated from
    :type nord: int
    :param xgrid: grid of x points
    :type xgrid: npt.NDArray
    :param kwargs: keyword args
    `coeffs` = list of precomputed polynomial coeffs for each problem,
    `coeff_min` = min value of coeffs if randomly generating polynomials,
    `coeff_max` = max value of coeffs,
    `v_min` = min value of potential in random generate of polynomial,
    `v_max` = max value of potential,
    either `coeffs` is specified alone for precomputed polynomials,
    or `coeff_min`, `coeff_max`, `v_min`, and `v_max` are specified together to generate a new random polynomial.
    :return: v_grid or v_exact, coeffs, order, v_min
    """
    coeff_min = kwargs.get('coeff_min')
    coeff_max = kwargs.get('coeff_max')
    v_min = kwargs.get('v_min')
    v_max = kwargs.get('v_max')
    v_exact, coeffs, order, v_min = generate_polynomial(nord, xgrid, coeff_min, coeff_max, v_min, v_max)
    print("poly generated")
    return v_exact, coeffs, order, v_min

def generate_polynomial(nord: int, xgrid: npt.NDArray, coeff_min: float, coeff_max: float, v_min: float, v_max: float) -> Tuple[npt.NDArray, npt.NDArray, int, float]:
    """
    Generates a random bound polynomial.

    :param nord: maximum order of on which the order of the polynomial is generated from
    :type nord: int
    :param xgrid: grid of x points
    :type xgrid: npt.NDArray
    :param coeff_min: min value of coeffs to be randomly generated
    :type coeff_min: float
    :param coeff_max: max value of coeffs to be randomly generated
    :type coeff_max: float
    :param v_min: minimum value of potential to be generated
    :type v_min: float
    :param v_max: maximum value of potential
    :type v_max: float
    :return: Vtarg, coeff, order, vm
    :rtype: Tuple[npt.NDArray, npt.NDArray, int, float]
    """

    ngrid = len(xgrid)
    Vtarg = np.zeros(ngrid)
    order = np.random.randint(low=2, high=nord + 1)
    print("generating poly.. of order ", order)
    bound = False
    while bound == False:
        coeff = np.random.uniform(low=coeff_min, high=coeff_max, size=order)
        Vtarg[:] = 0.000
        for j in range(0, order):
            Vtarg[:] += coeff[j] * xgrid ** (j + 1)

        # Get min and shift so that minimum is always zero...
        vm = min(Vtarg[:])
        Vtarg[:] -= vm
        fl = False
        for j in range(1, ngrid - 1):
            if Vtarg[j] > Vtarg[0] or Vtarg[j] > Vtarg[ngrid - 1]:
                fl = True
        if fl == False:
            if Vtarg[0] > v_min and Vtarg[0] < v_max:
                if Vtarg[ngrid - 1] > v_min and Vtarg[ngrid-1] < v_max:
                    bound = True
                    print("bound!")
                    return Vtarg, coeff, order, vm

def random_potentials(number_of_potentials, nord, grid, coeff_min, coeff_max, v_min, v_max):
    # Generates potentials of maximum order of nord of the form
    # V(x) = sum_{k=1}^{nord} a_{k} x^{k}
    # a_k coefficients are selected randomly from the range coeff_min to coeff_max
    grid = grid
    nord = nord
    coeff_min = coeff_min
    coeff_max = coeff_max
    v_min = v_min
    v_max = v_max

    directory = 'output/potential/dat/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(number_of_potentials):
        v, coeffs, order, vmin = polynomial(nord, grid, coeff_min=coeff_min, coeff_max=coeff_max, v_min=v_min, v_max=v_max)
        np.savetxt(f'{directory}potential_{i}.dat', np.column_stack((grid, v)), fmt=('%5.10f'), header='x\tV')
        np.savetxt(f'{directory}potential_{i}_data.dat', np.hstack((coeffs, order)), fmt=('%5.10f'), header='coeffs & order')

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

def calculate_TCF(npot, range_E, beta, grid, test=False):

    if test == True: # generate potentials from 2004 paper and calculate TCF
        print("Using potentials calculated with potential_2004")
        directory = 'potential/2004/'
        grid, v1 = np.loadtxt(f'{directory}potential_1.dat', unpack=True)
        # v1 =  1/4 x^4
        plt.plot(grid, v1)
        plt.savefig(f'{directory}potential_1.png')
        plt.close()
        c, E, H = colbert_miller_DVR(grid_size, grid, m, v1)
        range_E = range_E
        C = Kubo_TCF(grid, E, c, dx, t, range_E, beta=beta)
        data = np.column_stack((t,C))
        np.savetxt(f'{directory}Kubo_1.dat', data, fmt=('%5.2f', '%5.10f'), header='t\tC')
        plt.plot(t, C)
        plt.savefig(f'{directory}Kubo_1.png')
        plt.close()

        grid, v2 = np.loadtxt(f'{directory}potential_2.dat', unpack=True)
        # v2 =  1/2x^2 + 0.1x^3 + 0.01c^4
        plt.plot(grid, v2)
        plt.savefig(f'{directory}potential_2.png')
        plt.close()
        c, E, H = colbert_miller_DVR(grid_size, grid, m, v2)
        range_E = range_E
        C = Kubo_TCF(grid, E, c, dx, t, range_E, beta=beta)
        data = np.column_stack((t,C))
        np.savetxt(f'{directory}Kubo_2.dat', data, fmt=('%5.2f', '%5.10f'), header='t\tC')
        plt.plot(t, C)
        plt.savefig(f'{directory}Kubo_2.png')
        plt.close()
    else:
        random_potentials(npot, nord, grid, coeff_min, coeff_max, v_min, v_max)

        dat = 'output/potential/dat/'
        kubo_dat = 'output/Kubo/dat/'
        pot_png = 'output/potential/png/'
        kubo_png = 'output/Kubo/png/'

        directory = [dat, kubo_dat, kubo_png, pot_png]
        for d in directory:
            if not os.path.exists(d):
                os.makedirs(d)

        for i in range(npot):
            print(f'Loading potential_{i}...')
            grid, v = np.loadtxt(f'{dat}potential_{i}.dat', unpack=True)
            
            plt.plot(grid, v)
            plt.xlabel("Position")
            plt.ylabel("V(x)")
            plt.savefig(f'{pot_png}potential_{i}.png')
            plt.close()

            # returns c - ground state wavefunction - ith column of c contains wavefunction phi(i), E - eigenvalues, H - hamiltonian, of the system
            c, E, H = colbert_miller_DVR(grid_size, grid, m, v)

            # set range of eigenvalues to use when calculating TCF - can use all (set to len(E))!
            range_E = range_E
            C = Kubo_TCF(grid, E, c, dx, t, range_E, beta=beta)
            # save Kubo TCF data to file - also potential data to file
            data = np.column_stack((t,C))
            np.savetxt(f'{kubo_dat}Kubo_{i}.dat', data, fmt=('%5.2f', '%5.10f'), header='t\tC')
            plt.plot(t, C)
            plt.xlabel("Time")
            plt.ylabel("Kubo TCF")
            plt.savefig(f'{kubo_png}Kubo_{i}.png')
            plt.close()
            print(f'Saved to {kubo_png}Kubo_{i}.png')

# generate grid
x_min = -10
x_max = 10
grid_size = 101

grid, dx = get_exact_grid(x_min,x_max,grid_size)

# initialise variables
m = 1

# set timesteps
dt = 0.1
max_time = 20
t = np.arange(0,max_time, dt)

# generate random potentials
npot = 10
nord = 6
coeff_min = -5
coeff_max = 5
v_min = 5
v_max = 100000

# if test= True - Manolopolous (2004), False == random potentials
calculate_TCF(npot,range_E=10, beta=1, grid=grid,test=True)


