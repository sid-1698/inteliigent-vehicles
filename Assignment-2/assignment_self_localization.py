#!/usr/bin/python3
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pf_init_around_state import pf_init_around_state
from pf_init_freespace import pf_init_freespace
from pf_update_step import pf_update_step
from map_measurement_loglik import map_measurement_loglik
from copy_sensor import copy_sensor
from selfloc_scenario import selfloc_scenario
from pf_predict_step import pf_predict_step
from visualizations import plot_setup_selfloc, plot_sensor_rays,\
                            plot_vehicle_state, plot_measurements,\
                            plot_current_measurements, plot_gauss2d
# --------------------------------------------------------
# Intelligent Vehicles Lab Assignment
# --------------------------------------------------------
# Julian Kooij, Delft University of Technology

## Exercise 2.1: load and visualize the scenario
#-- define occupancy map, vehicle measurements --
np.random.seed(1)
scenario = selfloc_scenario()

# number of timesteps
T = scenario['T']

# time difference between timesteps (in seconds)
dt = scenario['dt']

map_ = scenario['map'] # occupancy map description of static environment
sensors = scenario['sensors'] # the sensor struct for all T timesteps
measurements = scenario['measurements'] # the measurement struct for all T timesteps
vehicles = scenario['vehicles'] # true position, pf CANNOT use this information
control_inputs = scenario['control_inputs'] # 2xT matrix, each column is contol input at time t

# initial true vehicle position
veh_init = vehicles[0]

# -- visualize --
plt.figure()
plot_setup_selfloc(map_)
ax = plt.gca()

for t in range(T):
    ##
    veh = vehicles[t]
    sensor = sensors[t]
    meas = measurements[t]

    # plot sensor rays
    lines_sensor = plot_sensor_rays(ax, sensor)
    
    # plot the vehicle state
    lines_veh = plot_vehicle_state(ax, veh)

    # plot measurements
    lines_meas = plot_measurements(t, sensor, measurements, c='r', marker='*')

    plt.pause(.01)
    lines_veh.remove()
    lines_meas.remove()
    for lines in [lines_sensor]:
        if lines is not None: [x.remove() for x in lines]

input("Press Enter to continue.")
## Exercise 2.2: Non-linear Motion model
# You will need to complete the code in
#     pf_predict_step

# initialize particle on the initial vehicle position
particle = np.array([veh_init['x'], veh_init['y'], veh_init['theta']])

# duplicate N times
particles = np.tile(particle.reshape(3,1), (1,1000))
assert particles.shape == (3, 1000)

# we will assume for now that the control input remains the same
# during the following 20 predict steps.
velocity = 1. # 1 m/s % <--  you can change this, of course
steering_angle = 0. # (in radians) 0 = go straight % <-- you can change this, of course
control_input = np.array([velocity, steering_angle])

# draw setup
plt.figure()
ax = plt.gca()
plot_setup_selfloc(map_)
plot_vehicle_state(ax, veh_init)

for step in range(20):
    # perform the particle predict step, using a fixed control input
    particles = pf_predict_step(particles, control_input, dt)

    # update plot
    lines_particles = ax.scatter(particles[0,:], particles[1,:], marker='.', c='m')
    ax.set_title(f'predicting {step+1} time steps ahead')
    
    plt.pause(.05)
    lines_particles.remove()

input("Press Enter to continue")
## Exercise 2.3 & 2.4: Measurement likelihood
#
# For Exercise 2.4, you will need to complete the code in
#     map_measurement_loglik

# let's take the first time step as reference
sensor = sensors[0]
vehicle = vehicles[0]
control_input = control_inputs[0]
measurement = measurements[0]

# -- select a test location --
# In a particle filter, the particles represent 'candidate' positions, or
# a set of 'hypotheses' about the true state.
# Here we take a look at 7 hypotheses, without considering any vehicle
# dynamics.
test_location_id = 0 # <-- *CHANGE THIS*, try out locations 0 to 6

test_particles = [
        [0, 10, 2*np.pi * 0/8], # first test location
        [0, 10, 2*np.pi * 2/8], # second test location
        [0, 16, 2*np.pi * 4/8], # third test location
        [0, 33, 2*np.pi * 4/8], # etc ...
        [10, 22, 2*np.pi * 2/8], 
        [-12, 22, 2*np.pi * 12/8], 
        [-10, 22, 2*np.pi * 13/8] 
]
particle = test_particles[test_location_id]

# The following lines create a hypothetical 'ideal' or 'expected' 
#   measurement for the selected particle.
#   This represents what we *expect* to measure if the vehicle is actually
#   at the particle position.
particle_sensor = copy_sensor(sensor, particle[0], particle[1], particle[2], 0.) # copy our sensor, but put it on the particle's state
particle_meas = particle_sensor.new_meas() # create a new virtual measurement for the sensor
particle_meas = particle_sensor.observe_point(particle_meas,\
                    list(map_['obstacles']), 1.) # measure the obstacles in the map

log_weight = map_measurement_loglik(particle, map_, measurement, sensor)
if log_weight == -9999:
    warnings.warning('You did not implement map_measurement_loglik correctly yet!',
                        UserWarning)
print(f'expected measurement at particle x_t\n'
        f'log weight = {log_weight:.4}, i.e. '
        f'weight = {np.exp(log_weight):.4}\n')

# -- visualize --
plt.subplot(1,2,1)
plot_setup_selfloc(map_)
ax = plt.gca()

# true vehicle state and measurements
plot_vehicle_state(ax, vehicle)
plot_sensor_rays(ax, sensor)
plot_current_measurements(ax, sensor, measurement)

# particle's vehicle state and measurements
plot_vehicle_state(ax, {'x': particle[0], 'y': particle[1], 
                    'theta': particle[2], 'kappa': 0})
plot_sensor_rays(ax, particle_sensor)
plot_current_measurements(ax, particle_sensor, particle_meas)

# show the 'ideal' sensor measurement
plt.subplot(2,2,2)
ax = plt.gca()
ax.plot(sensor.angles, measurement.dists, c='r')
ax.set_ylim([0, 30])
ax.set_ylabel('distance (m)')
ax.set_title('actual measurement z_t')

# also show the 'expected' measurement at the particle's position,orientation
plt.subplot(2,2,4)
ax = plt.gca()
ax.plot(particle_meas.dists, c='b')
ax.set_ylim([0, 30])
ax.set_xlabel('sensor ray')
ax.set_ylabel('distance (m)')
ax.set_title('expected measurement at particle x_t')

plt.pause(.1)
input("Press Enter to continue")
## Exercise 2.5, 2.6, 2.7 & 2.8: Particle Filtering
# For Exercise 2.5, you will need to complete the code in
#     pf_update_step

N = 100 # num particles % <-- change this
INITIAL_POSITION_KNOWN = True # <-- ** Exercise 2.7 **

# compute number of particles to reinitialize each timestep
frac_reinit = 0.0 # <-- ** Exercise 2.8 ** set fraction here
N_reinit = np.ceil(N * frac_reinit) # number of particles to reinitialize
N_reinit = int(N_reinit)

# setup plot
plot_setup_selfloc(map_)

# -- initialize particle filter --
if INITIAL_POSITION_KNOWN:
    # initial position known
    print('** informed initialization **\n')
    particles = pf_init_around_state(N, veh_init['x'], veh_init['y'],\
                                        veh_init['theta'])
else:
    # init random on freespace
    print('** random initialization **\n')
    particles = pf_init_freespace(N, map_)

assert particles.shape == (3,N)
# -- run particle filtering --
pf = {
    'particles' : [],
    'mean' : [],
    'cov' : []
    }
ax = plt.gca()
for t in range(T):
    meas = measurements[t]
    veh = vehicles[t]
    control_input = control_inputs[:,t]

    ## Predict step: predict motion and add noise
    particles = pf_predict_step(particles, control_input, dt)
    assert particles.shape == (3,N)

    # Exercise 2.8: randomly reinitialize N_reinit particles
    #   Tip: use pf_init_freespace here
#########################
## YOUR_CODE_GOES_HERE ##
#########################

    # show particles
    lines_particles_1 = ax.scatter(particles[0,:], particles[1,:], c='m', marker='.')

    ## Update step: evaluate particle likelihood and resample
    particles = pf_update_step(particles, meas, map_, sensor)
    
    ## particle statistics
    assert particles.shape == (3,N)
    m_pos = np.mean(particles[:2,:], axis=1)
    S_pos = np.cov(particles[:2,:])

    ## store
    pf['particles'].append(particles)
    pf['mean'].append(m_pos)
    pf['cov'].append(S_pos)

    ## plot for time t    
    
    # show particles
    lines_particles_2 = ax.scatter(particles[0,:], particles[1,:], c='g', marker='.')
    
    # show particle statistics with 2D Gaussian distribution
    lines_gauss = plot_gauss2d(m_pos, S_pos, c='b')

    # show actual the vehicle
    lines_veh = plot_vehicle_state(ax, veh)
    
    plt.pause(.01)
    [x.remove() for x in lines_gauss if x is not None]
    lines_veh.remove()
    lines_particles_1.remove()
    lines_particles_2.remove()

input('Press Enter to continue')

## animate stored results
plot_setup_selfloc(map_)
ax = plt.gca()

for t in range(T):
    meas = measurements[t]
    veh = vehicles[t]
    particles = pf['particles'][t]
    m_pos = pf['mean'][t]
    S_pos = pf['cov'][t]
    
    ## update the plot for time t    

    # show particles 
    lines_particles = ax.scatter(particles[0,:], particles[1,:], marker='.', c='g')
    
    # show particle statistics with 2D Gaussian distribution
    lines_gauss = plot_gauss2d(m_pos, S_pos, c='b')

    # show actual the vehicle
    lines_veh = plot_vehicle_state(ax, veh)
    
    plt.pause(.05)
    [x.remove() for x in lines_gauss if x is not None]
    lines_veh.remove()
    lines_particles.remove()

input('Press Enter to continue')

