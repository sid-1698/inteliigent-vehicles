import numpy as np
import math

from parking_scenario import parking_scenario
from determine_occupied_cells import *
from search_shortest_path_astar import *
from plot_setup_groundplane_2d import *
from plot_setup_vehicle_configuration_space_3d import *
from remove_unreachable_cells import *
from vertices_reachable_in_n_steps import *

[cspace, reachable] = parking_scenario()

# -- Practical instructions -- #
# DISABLING FIGURES: we often define figures via plot_setup_groundplane_2d or plot_setup_vehicle_configuraiton_space_3d
# you can set disabled to true to prevent animations/plots from running

# BLOCKING AT FIGURES: Use blocking=True in the draw() functions to stop execution at the figure

# SAVING FIGURES: See the function save() in plot_setup_groundplane_2d or plot_setup_vehicle_configuration_space_3d

# DEBUGGING SECTIONS: We recommend using an IDE such as Pycharm so that you can check sections of the code without
# rerunning by using breakpoints
# -- ---------------------- -- #

V = cspace.num_cells # total number of cells
D = cspace.num_dims # number of dimensions (D=4)
states = cspace.cell_centers # 4D state at each of the cell centers

# -- define environment --
# for simplicity, we 'draw' the occupied environment in a small grid
occupancy_2d = np.zeros([32, 32]) # 32 x 32 is the same gridsize used in the sp struct
occupancy_2d[19:26, 26:32] = 1
occupancy_2d[19:26, 0:6] = 1

# compute which cells in the discrete space are occupied, given the static obstacles in the world
state_occupancy = determine_occupied_cells(cspace, occupancy_2d)

# print some information on the size in the discrete configuration space
print('-- discrete configuration space --')
print('dimension 1 (x pos) has ' + str(cspace.dim_size[0]) + ' cells')
print('dimension 2 (y pos) has ' + str(cspace.dim_size[1]) + ' cells')
print('dimension 3 (theta) has ' + str(cspace.dim_size[2]) + ' cells')
print('dimension 4 (omega) has ' + str(cspace.dim_size[3]) + ' cells')
print('In total, the state space has ' + str(V) + ' cells')

# -- Exercise 2.1: Visualize the vehicle in the discrete configuration space -- #
# We can define the vehicle state by its grid cell in the 4D configuration space,
# and map it to a unique single vertex idx.

# define a grid cell in the discrete state spaceweb
idx_4d = np.array([1, 1, 1, 1]).transpose()  # ** change this **

# convert given 4D grid indices to single cell id
idx = cspace.map_nd_to_1d(idx_4d)
print('the cell at grid location [%d, %d, %d, %d]' % (idx_4d[0], idx_4d[1], idx_4d[2], idx_4d[3]))
print('    has id %d' % idx)

# Draw the 2D workspace
init_config_space = figure_2d(disabled=False, idx=1)
init_config_space.plot_setup_groundplane_2d(cspace, occupancy_2d)
init_config_space.plot_vehicle_state(vehicle_struct(states, idx-1))
init_config_space.draw(pause=1.0, blocking=False)

# Draw the 3D configuration space
init_3d_config_space = figure_3d(disabled=False, idx=2)
init_3d_config_space.plot_vehicle_configuration_space(cspace, state_occupancy)
# init_3d_config_space.plot_vehicle_state_configuration_space(cspace, idx, marker='d', label='Start', color='blue', ms=10)

# Draw reachable vertices from the selected idx
nsteps = 6
[to_idxs, from_idxs] = vertices_reachable_in_n_steps(idx, reachable, nsteps)
init_3d_config_space.plot_edges_configuration_space(cspace, from_idxs, to_idxs, color='red', label='Reachable in ' + str(nsteps) + ' steps')
init_3d_config_space.draw(pause=1.0, blocking=False)

# -- Exercise 2.2 & 2.3: plan the path using A-star in the configuration space -- #
# Select the planning problem here. All problems have the vehicle start
# at the same origin, but define different goal positions in the environment.
planning_problem_idx = 1  # ** change this **

# define the vehicle start position in the created discrete space
start = cspace.map_nd_to_1d(np.array([7, 1, 1, 1]))

# Look up the goal state based on planning_problem_idx
if planning_problem_idx is 1: goal = cspace.input_to_1d(np.array([8, 4.5, 270/180*np.pi, 0]))
elif planning_problem_idx is 2: goal = cspace.input_to_1d(np.array([10, 1, 180/180*np.pi, 0]))
elif planning_problem_idx is 3: goal = cspace.input_to_1d(np.array([10, 1, 0, 0]))
elif planning_problem_idx is 4: goal = cspace.input_to_1d(np.array([10, 9, 180/180*np.pi, 0]))
elif planning_problem_idx is 5: goal = cspace.input_to_1d(np.array([7, 5, 180/180*np.pi, 0]))

# Ensure indicies are integers
goal = int(goal)
start = int(start)

print("Planning the path")


# A heuristic / cost function
def vertex_dist(v, idx):
    return np.linalg.norm(np.squeeze(states[:, v]) - states[:, np.atleast_1d(idx)].transpose(), axis=1)

print(len(reachable), reachable[0])
reachable = remove_unreachable_cells(reachable, state_occupancy)
# perform graph search for shortest path
print("-----------------------")
print(len(reachable), reachable[0])
[path, info] = search_shortest_path_astar(V, start, goal, reachable, vertex_dist, vertex_dist)

# Done, print some info
print('\n')
print('start            : ', info["start"])
print('goal             : ', info["goal"])
print('length of path   : ', info["path_length"] / 1000.)
print('costs            : ', info["costs"])
print('backpoint        : ', info["backpoint"])
print('iterations       : ', info["iterations"])
print('vertices in path : ', len(path))

animated_path_fig = figure_2d(disabled=False, idx=3)

# Animate the path
for v in path:
    # Plot the current state
    animated_path_fig.plot_setup_groundplane_2d(cspace, occupancy_2d)
    animated_path_fig.plot_path(states, path)

    # Start and goal
    animated_path_fig.plot_vehicle_state(vehicle_struct(states, start), color='blue')
    animated_path_fig.plot_vehicle_state(vehicle_struct(states, goal), color='green')

    # The total path
    animated_path_fig.plot_vehicle_state(vehicle_struct(states, int(v)))

    # Draw and clear
    animated_path_fig.draw()

    if v != path[len(path) - 1]:
        animated_path_fig.clear()


# Create a 3D figure showing the path in configuration space
path_3d_fig = figure_3d(disabled=True, idx=4)

# Plot the occupancy
path_3d_fig.plot_vehicle_configuration_space(cspace, state_occupancy)

# Plot start and goal as markers
# path_3d_fig.plot_vehicle_state_configuration_space(cspace, start, marker='d', markerfacecolor='blue', markeredgecolor='black',
#                                           markersize=10, label='Start')
# path_3d_fig.plot_vehicle_state_configuration_space(cspace, goal, marker='d', markerfacecolor='green', markeredgecolor='black',
#                                           markersize=10, label='Goal')

# Plot the path
path_3d_fig.plot_vehicle_state_configuration_space(cspace, path, linestyle='--', color='red', markeredgecolor='black',
                                          markersize=15, label='Path', linewidth=2, zorder=10)

# Draw
path_3d_fig.draw(pause=1.0, blocking=True)


input("Press Enter to continue")




