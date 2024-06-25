# goal - code to calculate 1D correlation functions for given potentials, using classical MD

import numpy as np
import matplotlib.pyplot as plt
from sympy import *

# Classical MD functions
# Regen potential & obtain energy & force expressions
def potential():
    # regenerates expression from potential data file
    # returns python lambda expressions for energy (V(x)) and force (F(x)) to be used in individual calculations (see potential_energy_force for actual use in MD)
    coeffs_array = np.loadtxt('output/potential/dat/potential_0_data.dat')

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

    # plotting code to check
    # plt.plot(grid,energy)
    # plt.plot(grid, force)
    # plt.savefig('potential_energy_force.png')

    return lam_e, lam_f

def potential_energy_force(x):
    # returns expressions of energy and force based on potential()
    # lam_e, lam_f = potential()
    # energy = lam_e(x)
    # force = lam_f(x)

    # V(x) = 0.25x**4
    energy = 0.25 * x **4
    force = -x**3
    return energy, force

# sample velocity
def sample(beta, m):
    sigma = np.sqrt(1/(beta*m))
    v = np.random.normal(0,sigma)
    return v

# velocity verlet functions
# functions to update position and velocity of particle for 1 timestep
def upd_pos(x,v,m,dt,f):
    x_new = x + v*dt + 0.5 * m * f * dt**2
    return x_new

def upd_vel(v,m,dt,f,f_new):
    v_new = v + 0.5* m * (f + f_new) * dt
    return v_new

# velocity verlet for 1 timestep
def velocity_verlet_1(x_init, v_init, m, dt):
    # initial position and velocity
    x = x_init
    v = v_init
    # initial forces on the particle
    energy, force = potential_energy_force(x)
    total_energy = energy + 0.5 * m * v * v
    # update position
    x = upd_pos(x,v,m,dt,force)
    # new forces from new position
    energy_new, force_new = potential_energy_force(x)
    # update velocity
    v = upd_vel(v,m,dt,force, force_new)

    return x,v, total_energy

# main velocity verlet function - boolean for thermostat (True for equilibration, False for dynamics)
# equilibration time and dynamics time - 2 different max times
def velocity_verlet(beta, x_init, m, eq_time, max_time,dt,tau):
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
        if t<eq_time:
            x,v, e_tot = velocity_verlet_1(x,v,m,dt)
            velocities.append(v)
            energies.append(e_tot)
            # thermostat on
            i = np.random.rand()
            if i <= tau:
                v = sample(beta,m)
            t = t + dt
        elif t>=eq_time: # dynamics - accumulate positions
            x,v, e_tot = velocity_verlet_1(x,v,m,dt)
            positions.append(x)
            velocities.append(v)
            energies.append(e_tot)
            t = t+dt


    return times, dy_times, positions, velocities, energies

# calculating position auto correlation function for 1 trajectory
def position_auto_correlation_function():
    correlation_function = []

    for t in range(len(dy_times)):
        cf_t = positions[0] * positions[t]
        correlation_function.append(cf_t)

    return correlation_function

# TODO - loop over a number of trajectories
# DONE - for each traj generate initial conditions and run
    # DONE - generate initial conditions from constant temperature ensemble
    # DONE - Run initial MD trajectory with a thermostat (Anderson) attached
        # DONE - Sample initial velocity of particle
        # DONE - Equilibrate for set number of timesteps - Andersen thermostat
        # DONE Resample initial velocities


    # Run MD trajectory and get position
    # Accumulate correlation function
    # Average correlation function - divide by no of trajectories

# TODO - calculate overall ensemble average - average over lots of diff trajectories

# equilibrate with Anderson thermostat attached
beta = 1
x_init = 0
mass = 1
max_time = 80
eq_time = max_time / 2
dt = 0.01
tau = 0.005

times, dy_times, positions, velocities, energies = velocity_verlet(beta, x_init, mass, eq_time, max_time, dt, tau)

# plotting
plt.plot(times, energies, label="Energy")
plt.legend()
plt.savefig('energy.png')
plt.close()

# accumulate the correlation function
C_t = position_auto_correlation_function() # C_t is array containing correlation function

plt.plot(dy_times, C_t, label="TCF")
plt.legend()
plt.savefig('TCF.png')
plt.close()