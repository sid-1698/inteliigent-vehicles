import numpy as np
from setup_trajectory_scenario import *
from state_types import *
from euler_integrate_motion import *
from curv_param_state_predict import *
from plot_setup_trajectory_planning import *
from make_steering_profile import *
from animate_steering_profile_plot import steering_animation_figure
from optimize_control_parameters import *
from trajplanning_obstacle_scenario import *
from compute_trajectory_cost import *


# -- Practical instructions -- #
# DISABLING FIGURES: we often define figures via trajectory_figure, you can set disabled to true to prevent
# animations/plots from running

# BLOCKING AT FIGURES: Use blocking=True in the draw() functions to stop execution at the figure

# SAVING FIGURES: See the function save() in trajectory_figure

# DEBUGGING SECTIONS: We recommend using an IDE such as Pycharm so that you can check sections of the code without
# rerunning by using breakpoints
# -- ---------------------- -- #

# -- Define road segment -- #
# define the road curvature radius (meter)
rradius = 60  # as rradius goes to inf, the road becomes more straight
road = road(rradius)  # Initialize the road

print(' -- road segment properties --')
print('  segment length: %d meter' % road.rlength)
print('curvature radius: %d meter' % road.rradius)
print('      lane width: %d meter' % road.rwidth)

# -- Plot the created road -- #
# Draw the road
setup_figure = trajectory_figure(disabled=True, idx=1)
setup_figure.plot_setup_trajectory_planning(road)

# Draw the vehicle's initial state
init_state = vehicle_state(0., 0., 0.0*np.pi, 1./rradius, 5., 0.)
setup_figure.plot_vehicle_state(init_state)
setup_figure.draw(pause=1.0, blocking=True)

# -- Exercise 3.1: Define a spline steering profile -- #
# You will need to complete the code in
#     make_steering_profile.py

k0 = -1.
k1 = .2
k2 = -.3  # some values for clear illustration

# Create a steering profile
steer_profile = steering_profile(k0, k1, k2, True)

# Plot the profile, and animate the effect on the wheels
steering_anim = steering_animation_figure(disabled=True)
steering_anim.animate_steering_profile_plot(steer_profile)

# -- Define goal position and angle along the road -- #
# The goal vehicle state is defined with respect the road curve.
lat_offset = 0  # lateral offset (in meters) how many meters from centerline ?
long_frac = 1   # longitudinal offset (fraction between 0 and 1)
goal_state = get_goal_state(road, long_frac, lat_offset, init_state)

# Visualize the problem
# plot initial state
initial_goal_figure = trajectory_figure(disabled=True, idx=3)
initial_goal_figure.plot_setup_trajectory_planning(road)
initial_goal_figure.plot_vehicle_state(init_state, color='blue', label='initial state')

# plot goal state
initial_goal_figure.plot_vehicle_state(goal_state,  color='green', label='goal state')
initial_goal_figure.draw(pause=1.0, blocking=True)

# -- Exercise 3.2 & 3.3: Euler integration of the motion model -- #
# Let's take a look at how the steering control parameters affect the path.
#
# You will need to complete the code in
#     euler_integrate_motion.py   (Exercise 3.2)
#     curv_param_state_predict.py (Exercise 3.3)

# set initial parameter estimates (will later be redefined through optimization)
k0 = init_state.kappa
k1 = 1/63         # ** change this **
k2 = -1/32      # ** change this **
path_length = 40    # ** change this **

# simultation time steps in seconds, i.e. 10 Hz
dt = 1e-1

# Todo(Exercise 3.2): Take a look at the continuous-time dynamical model of the vehicle in state_types.py

# Change this once you understand how the k parameters affect the vehicle path.
# Todo (Exercise 3.3): Change this parameter to True to test curv_param_state_predict(.)
USE_CURV_PARAM_STATE_PREDICT = False

if not USE_CURV_PARAM_STATE_PREDICT:
    # -- Perform Euler integration -- #

    # define steering profile
    steering_profile = steering_profile(k0, k1, k2)

    # simulate dynamics: return the vehicle state at each step
    states = euler_integrate_motion(init_state, dt, path_length, steering_profile)

else:
    # -- Perform the same as above, but in a single function so that we can run an optimization -- #
    # Put the control parameters in a single vector
    params = [k1, k2, path_length]

    # Retrieve states as a function of the initial state and the parameters
    states = curv_param_state_predict(init_state, dt, params)

# Animate the resulting trajectory
initial_vehicle_trajectory_figure = trajectory_figure(disabled=True, idx=4)
initial_vehicle_trajectory_figure.animate_vehicle_trajectory(road, states)

# -- Exercise 3.4: Optimize control parameters -- #
# Now the goal is to find those control parameters that make the vehicle
# reach the goal position and orientation.
#
# You will need to complete the code in
#     optimize_control_parameters.py

# initial parameters
k1 = init_state.kappa
k2 = init_state.kappa
path_length = road.rlength
init_params = np.array([k1, k2, path_length])

# Run the non-linear optimization algorithm
params = optimize_control_parameters(init_state, goal_state, dt, init_params)

# Recompute optimal state sequence for optimized control parameters
states = curv_param_state_predict(init_state, dt, params)

# Animate the optimized trajectory
optimized_vehicle_trajectory_figure = trajectory_figure(disabled=True, idx=5)
optimized_vehicle_trajectory_figure.animate_vehicle_trajectory(road, states)

params_center_line = params

# -- Create candidate trajectories with different lateral offsets -- #
# Using the trajectory optimization, we can here create multiple
# candidate trajectories with different lateral offsets in a simple
# for-loop.

num_candidates = 7  # number of candidate trajectories, each with different lateral offset

lat_offsets = np.linspace(-5, 5, num_candidates)  # how many meters from centerline ?
init_params = params_center_line  # initialize from parameters found in first optimization round

candidates = np.array([])
for lat_offset in lat_offsets:

    # define new goal for given lateral offset
    goal_state_lat = get_goal_state(road, 1, lat_offset, init_state)

    # run optimization
    params = optimize_control_parameters(init_state, goal_state_lat, dt, init_params)

    # recompute optimal state sequence
    states = curv_param_state_predict(init_state, dt, params)

    # Define a new candidate trajectory (see state_types.py)
    new_candidate = candidate(lat_offset, params, states)

    # append the results to the list of candidates
    candidates = np.append(candidates, new_candidate)


# plot the candidate tracks
candidate_figure = trajectory_figure(disabled=False, idx=6)

# Plot the scene
candidate_figure.plot_setup_trajectory_planning(road)
candidate_figure.plot_vehicle_state(init_state)

# Plot the candidate trajectories
for c in candidates:
    states = c.states
    candidate_figure.plot_trajectory(states, color='gray')

candidate_figure.draw(pause=1.0, blocking=False)

# -- Exercise 3.5: Compute trajectory cost with respect to other obstacles -- #
# We now consider various scenarios with another moving vehicle.
# Each scenario defines a variation with a different vehicle moving
#  on (or off!) the road, potentially crossing the path of our own
#  vehicle.
#
# You will need to complete the code in
#     compute_trajectory_cost.py

# select scenario 1 to 6
# for idx in range(1,7):
scenario_idx = 5# ** change this **

# Construct obstacle trajectories based on the picked scenario
obstacle_states = trajplanning_obstacle_scenario(road, scenario_idx)

# evaluate candidate tracks
for cidx in range(candidates.size):
    c = candidates[cidx]

    # compute the cost of the candidate by comparing the proposed
    # trajectory to the predicted path of the obstacle,
    # and by considering the lateral offset of the path
    [total_cost, cost_per_timestep] = compute_trajectory_cost(obstacle_states, c.states, c.lat_offset)

    print('Cost for candidate #%d: %.2f' % (cidx, total_cost))

    # add the computed cost to the stored candidate information
    candidates[cidx].cost_per_timestep = cost_per_timestep
    candidates[cidx].total_cost = total_cost

    # candidate_cost = list(c.total_cost for c in candidates)

    # best_cost = min(candidate_cost)
    # best_cidx = np.where(candidate_cost == best_cost)[0][0]
    # s_candidate = candidates[best_cidx]
    # states = s_candidate.states
    # print('-- selected candidate %d : total_cost = %.3f --' % (best_cidx, s_candidate.total_cost))

# plot candidate trajectories color by cost
candidate_cost_figure = trajectory_figure(disabled=False, idx=7)
candidate_cost_figure.plot_setup_trajectory_planning(road)
candidate_cost_figure.plot_vehicle_state(init_state)

# Retrieve the cost of each candidate
candidate_cost = list(c.total_cost for c in candidates)

# Compute the range of costs for coloring
cost_range_min = min(candidate_cost)
cost_range_max = max(candidate_cost)
cost_range = cost_range_max - cost_range_min

for c in candidates:
    total_cost = c.total_cost
    states = c.states

    # use the cost to visualize good (green) and bad (red) trajectories
    alpha = float((total_cost - cost_range_min) / (cost_range + 1E-5))
    candidate_figure.plot_trajectory(states, color=(alpha, 1.-alpha, 0.))

candidate_figure.draw(pause=1.0, blocking=False)

# -- inspect a single candidate -- #
# determine optimal candidate and its index
best_cost = min(candidate_cost)
best_cidx = np.where(candidate_cost == best_cost)[0]

if best_cidx.size > 0:
    cidx = best_cidx[0]
else:
    cidx = 0
    print('Warning: No best candidate trajectory found (or all trajectories are of equal cost)!')

#  cidx = 1  # DEBUG: you could manually select any of the trajectories here

s_candidate = candidates[cidx]
states = s_candidate.states
print('-- selected candidate %d : total_cost = %.3f --' % (cidx, s_candidate.total_cost))

# -- animate the resulting scenario -- #
maneuvre_figure = trajectory_figure(disabled=False, idx=8)

T = min(states.size, obstacle_states.size)
for t in range(T):
    # Plot the scene
    maneuvre_figure.plot_setup_trajectory_planning(road)
    for c in candidates:
        total_cost = c.total_cost
        candidate_states = c.states

        # use the to visualize good (green) and bad (red) trajectories
        alpha = float((total_cost - cost_range_min) / (cost_range + 1E-5))  # per element
        candidate_figure.plot_trajectory(candidate_states, color=(alpha, 1. - alpha, 0.), linewidth=0.5)

    # Show the cost over time (comment to skip)
    maneuvre_figure.plot_trajectory_cost(states, s_candidate.cost_per_timestep)

    obstacle = obstacle_states[t]
    state = states[t]

    # update ego-vehicle and obstacle
    maneuvre_figure.plot_vehicle_state(state)
    maneuvre_figure.plot_vehicle_state(obstacle, color='blue', label='obstacle')

    maneuvre_figure.draw()
    maneuvre_figure.clear()
