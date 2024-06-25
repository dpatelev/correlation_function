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
    lam_e, lam_f = potential()
    energy = lam_e(x)
    force = lam_f(x)

    # returns expressions of energy and force based on potential()
    return energy, force

# velocity verlet functions

def update_position(x,v,F,dt,m):
    x_new = x + v*dt + 0.5*m*F*dt*dt
    return x_new

def update_velocity(v,F_new,F_old,dt,m):
    v_new = v + 0.5*m*(F_old+F_new)*dt
    return v_new

# TODO - change function so it only does one timestep
def velocity_verlet(potential_e_f, max_time, dt, initial_position, initial_velocity, m, save_frequency=1):
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
        potential_energy, force = potential_e_f(x)
        if step_number%save_frequency == 0:
            e_total = 0.5*v*v + potential_energy

            positions.append(x)
            velocities.append(v)
            total_energies.append(e_total)
            save_times.append(t)
        
        x = update_position(x,v,force,dt,m)
        potential_energy2, force2 = potential_e_f(x)
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

initial_energy = 1
initial_position = 0.5
initial_velocity = 1

dt = 0.1
max_time = 20

# After equilibration period, run constant-energy simulation to accumulate the correlation function, which is repeated for lots of initial conditions
# loop for nt timesteps after VV func changed
times, positions, velocities, total_energies = velocity_verlet(potential_energy_force, max_time, dt, initial_position, initial_velocity, m)

# loop over time values and calculate the correlation function
C_t = position_auto_correlation_function() # C_t is array containing correlation function

plt.plot(times,C_t)
plt.savefig(f'calc_Kubo_0.png')