# goal - code to calculate 1D correlation functions for given potentials, using classical MD
# TODO - what is the bare minimum (i.e. no plots) for program synthesis?

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import yaml

# Classical MD functions
# Regen potential & obtain energy & force expressions
def potential(filepath):
    """
    Regenerates polynomial expression from the data file and returns lambda expressions for the potential energy and force.

    Args:
        filepath: path to file

    Returns:
        lam_e: lambda expression for potential energy
        lam_f: lambda expression for force
    """
    # regenerates expression from potential data file
    # returns python lambda expressions for energy (V(x)) and force (F(x)) to be used in individual calculations (see potential_energy_force for actual use in MD)
    coeffs_array = np.loadtxt(filepath)

    order = int(coeffs_array[-1])
    coeffs = coeffs_array[:-1]
    v = ''
    expr = ''
    for i in range(len(coeffs)):
        for j in range(0,order):
            if i == j:
                expr += v.join(f'{coeffs[i]}*x**{j+1} + ')
    expr = expr[:-3]
    V = sympify(expr)

    x = symbols('x')
    F = -diff(V,x)

    lam_e = lambdify(x, V, modules=['numpy'])
    lam_f = lambdify(x, F, modules=['numpy'])

    return lam_e, lam_f

def potential_energy_force(x, lam_e, lam_f):
    """
    Returns the potential energy and force of a particle based on its position x.

    Args:
        x: position
        lam_e: lambda expression for potential energy
        lam_f: lambda expression for force

    Returns:
        energy: potential energy on particle at position x
        force: force on particle at position x

    """
    # returns expressions of energy and force based on potential()
    energy = lam_e(x)
    force = lam_f(x)
    return energy, force

# sample velocity
def sample(beta, m):
    """
    Randomly picks a velocity from the normal distribution with width parameter sigma.

    Args:
        beta: inverse temperature
        m: mass of particle
    
    Returns:
    v: velocity of particle
    """
    sigma = np.sqrt(1/(beta*m))
    v = np.random.normal(0,sigma)
    return v

# velocity verlet functions
# functions to update position and velocity of particle for 1 timestep
def upd_pos(x,v,m,dt,f):
    """
    Updates position of particle according to the velocity Verlet algorithm.
    
    Args:
        x: position
        v: velocity
        m: mass
        dt: timestep
        f: force
    
    Returns:
        x_new: new position of particle
    """    
    x_new = x + v*dt + 0.5 * m * f * dt**2
    return x_new

def upd_vel(v,m,dt,f,f_new):
    """
    Updates velocity of particle according to the velocity Verlet algorithm.
    
    Args:
        v: velocity
        m: mass
        dt: timestep
        f: force
        f_new: new force on particle
    
    Returns:
        v_new: new velocity of particle
    """
    v_new = v + 0.5* m * (f + f_new) * dt
    return v_new

# velocity verlet for 1 timestep
def velocity_verlet_1(x_init, v_init, m, dt, lam_e, lam_f):
    """
    Returns the position & velocity on a particle, and the total energy after 1 velocity Verlet timestep.

    Args:
        x_init: initial position
        v_init: initial velocity
        m: mass
        dt: timestep
        lam_e: lambda expression for potential energy
        lam_f: lambda expression for force

    Returns:
        x: position
        v: velocity
        total_energy: total energy of system
    """
    # initial position and velocity
    x = x_init
    v = v_init
    # initial forces on the particle
    lam_e = lam_e
    lam_f = lam_f
    energy, force = potential_energy_force(x, lam_e, lam_f)
    total_energy = energy + 0.5 * m * v * v
    # update position
    x = upd_pos(x,v,m,dt,force)
    # new forces from new position
    energy_new, force_new = potential_energy_force(x, lam_e, lam_f)
    # update velocity
    v = upd_vel(v,m,dt,force, force_new)

    return x,v, total_energy

# main MD function
# equilibration time and dynamics time - 2 different max times
def velocity_verlet(beta, x_init, m, eq_time, max_time,dt,tau, lam_e, lam_f):
    """
    Runs a full, single velocity Verlet MD trajectory with equilibration and dynamics.

    Args:
        beta: inverse T
        x_init: initial position
        m: mass
        eq_time: equilibration time
        max_time: full length of simulation
        dt: timestep
        tau: threshold for Anderson thermostat
        lam_e: lambda expression for potential energy
        lam_f: lambda expression for force

    Returns:
        times: full time seroes
        dy_times: time series used in dynamics
        positions: positions of particle
        velocities: velocities of particle
        energies: total energies of the system
    """

    # lambda energy and forces
    lam_e = lam_e
    lam_f = lam_f
    beta = beta
    m = m
    num = int(max_time/dt)
    # timesteps
    times = np.linspace(0,max_time,num)
    dy_times = np.linspace(eq_time,max_time,int((max_time-eq_time)/dt))
    positions = []
    velocities = []
    energies = []

    # initial position and velocity
    x = x_init
    v = sample(beta,m)

    # loop over timesteps
    t = 0
    for t in times:
        if t<eq_time: # equilibration
            x,v, e_tot = velocity_verlet_1(x,v,m,dt, lam_e, lam_f)
            velocities.append(v)
            energies.append(e_tot)
            # thermostat on
            i = np.random.rand()
            if i <= tau:
                v = sample(beta,m)
            t = t + dt
        elif t>=eq_time: # dynamics - accumulate positions
            x,v, e_tot = velocity_verlet_1(x,v,m,dt, lam_e, lam_f)
            positions.append(x)
            velocities.append(v)
            energies.append(e_tot)
            t = t+dt

    return times, dy_times, positions, velocities, energies

# calculating position auto correlation function for 1 trajectory
def position_auto_correlation_function(dy_times, positions):
    """
    Calculates the position auto correlation function for a single trajectory.
    Args:
        dy_times: time series used in dynamics
        positions: positions of particle
    Returns:
        correlation_function: array of position auto correlation function
    """
    correlation_function = []

    for t in range(len(dy_times)):
        cf_t = positions[0] * positions[t]
        correlation_function.append(cf_t)

    return correlation_function

def ensemble_TCF(num_traj,beta, x_init, mass, eq_time, max_time, dt, tau, lam_e, lam_f):
    """
    Carries out num_traj MD trajectories and calculates the average position auto correlation function.

    Args:
        num_traj: number of trajectories
        beta: inverse T
        x_init: initial position
        mass: mass
        eq_time: equilibration time
        max_time: full length of simulation
        dt: timestep
        tau: threshold for Anderson thermostat
        lam_e: lambda expression for potential energy
        lam_f: lambda expression for force
    
    Returns:
        Ct_all: averaged position auto correlation function
        dy_times: time series used in dynamics
    """
    num_traj = num_traj
    beta = beta
    x_init = x_init
    mass = mass
    max_time = max_time
    eq_time = eq_time
    dt = dt
    tau = tau

    t = np.linspace(eq_time,max_time,int((max_time-eq_time)/dt))
    lam_e = lam_e
    lam_f = lam_f

    Ct_all = np.zeros(len(t))
    for i in range(num_traj):
        times, dy_times, positions, velocities, energies = velocity_verlet(beta, x_init, mass, eq_time, max_time, dt, tau, lam_e, lam_f)
        C_t = position_auto_correlation_function(dy_times, positions)
        for j in range(len(Ct_all)):
            Ct_all[j] += C_t[j]
    for i in range(len(Ct_all)):
        Ct_all[i] = Ct_all[i] / num_traj
    
    # setting dy_times to start from 0 for plotting purposes
    dy_times = dy_times - eq_time
    return Ct_all, dy_times

def main():
    """
    Runs MD simulations and calculates TCF for all potential files.
    """
    with open('traj_input.yaml', 'r') as file:
        data = yaml.safe_load(file)
    items = list(data.items())

    num_traj = items[0][1]
    beta = items[1][1]
    x_init = items[2][1]
    mass = items[3][1]
    max_time = items[4][1]
    dt = items[5][1]
    tau = items[6][1]
    directory = items[7][1]
    eq_time = max_time / 2
    dir = f'{directory}/potential/dat/'
    fileCounter = len(glob.glob1(dir, '*_data.dat'))
    print(f'Running {num_traj} trajectories for {fileCounter} potential(s)')
    for i in range(fileCounter):
        filepath = f'{dir}potential_{i}_data.dat'
        lam_e, lam_f = potential(filepath)
        print(f'Running MD trajectories for potential {i}......')
        Ct_all, dy_times = ensemble_TCF(num_traj,beta, x_init, mass, eq_time, max_time, dt, tau, lam_e, lam_f)
        print(f'Saving calculated Kubo TCF {i} to file')

        dir_png = f'{directory}/MD/png/'
        dir_dat = f'{directory}/MD/dat/'
        dirs = [dir_png, dir_dat]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        plt.plot(dy_times, Ct_all)
        plt.savefig(f'{dir_png}calc_Kubo_{i}.png')
        plt.close()

        np.savetxt(f'{dir_dat}calc_Kubo_{i}.dat', np.column_stack((dy_times,Ct_all)), fmt=('%5.2f', '%5.10f'),header='t\tC(t)')
    print('Complete!')

if __name__ == "__main__":
    main()
