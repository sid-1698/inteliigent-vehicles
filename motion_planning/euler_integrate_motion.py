import math
import numpy as np
from state_types import *
import copy

# Euler integration of motion model
def euler_integrate_motion(init_state, dt, path_length, steering_profile):

    # Assuming a constant velocity profile:
    # - estimate the duration to travel the given path length
    # - estimate traveled distance at each step
    velocity = init_state.v 
    T = path_length / velocity
    nsteps = math.ceil(T / dt)

    ts = (np.arange(nsteps) + 1) * dt
    traveled_dists = velocity * ts

    t = 0  # t is time in seconds since start
    
    # set initial state (copy necessary because we do not want to edit the initial state)
    state = copy.copy(init_state)

    # start simulation
    states = np.array([])
    for step in range(nsteps):
        t = t + dt
        dt = min(dt, T-t)  # last time step might use smaller dt

        # assuming fixed velocity profile, so 0 acceleration
        u_acc = 0

        # -- Exercise 3.2: Implementing Euler euler integration on the dynamical model -- #
        # Here we 
        #   - compute the steering control u_steer, and then
        #   - simulate the next state using the motion model
        
        # Compute the steering control u_steer
        #  by calling the steering profile using the fraction
        #  of path traveled s / s_f.
        #  Note that the current traveled distance s is given
        #  by traveled_dists(step), and path_length gives the
        #  total length s_f.
#########################
## YOUR_CODE_GOES_HERE ##
#########################
        s_frac = traveled_dists[step]/path_length
        u_steer = steering_profile.steering_profile_function_spline(s_frac)
        # Now we simply update the vehicle state using the given
        #   motion_model, the control inputs u_acc and u_steer,
        #   and the time difference dt.
#########################
## YOUR_CODE_GOES_HERE ##
#########################
        # -----------------------------------------------------------------------------------#
        state = vehicle_state(state.x, state.y, state.theta, state.kappa, state.v, state.a)
        # print(state.x, state.y, state.theta, state.kappa, state.v, state.a)
        state.motion_model(u_acc, u_steer, dt)
        # print("-----------------------------------------------------------------")
        # print(state.x, state.y, state.theta, state.kappa, state.v, state.a)
        # print("------------------------------------------------------------------")
        # print("------------------------------------------------------------------")
        # append a copy of the result
        states = np.append(states, copy.copy(state))

    # Return the array of states
    return states
