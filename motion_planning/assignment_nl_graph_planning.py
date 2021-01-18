import scipy.io
import numpy as np
from plot_network_graph import plot_network_graph
from search_shortest_path import search_shortest_path
from search_shortest_path_astar import search_shortest_path_astar
import matplotlib.pyplot as plt

# load and visualize the route planning problem
# load Dutch road network, data obtained from
#   http://www.cc.gatech.edu/dimacs10/archive/streets.shtml
#
# NOTE: this can take a while...
mat = scipy.io.loadmat('netherlands-osm-road-network.mat')

V         = np.squeeze(mat["V"])  # number of vertices
E         = mat["E"]  # number of edges
XY        = mat["XY"] # V x 2 matrix with the 2D locations of all V vertices
G         = mat["G"]  # V x V sparse matrix defining the network graph structure:
                      # G_i,j == 1 if vertices i en j are connected, 0 otherwise
reachable = np.squeeze(mat["reachable"] - 1) # alternative network connectivity representation:
                              # reachable{i} contains list of all vertex
                              # indices that can be reached from vertex i

# Display network graph
# NOTE: this can take a while...
plot_network_graph(G, XY)

## Select the path planning problem
planning_problem_idx = 1 # <-- change this

if planning_problem_idx == 1:
	start = 0
	goal = 100399 # shortest path: 42.912 km
if planning_problem_idx == 2:
	start = 0
	goal = 1390 # shortest path: 157.681 km
if planning_problem_idx == 3:
	start = int(214e4-1)
	goal = 1600 # shortest path: ?
if planning_problem_idx == 4:
	start = int(32e4-1)
	goal = 1599
if planning_problem_idx == 5:
	start = int(205e4-1)
	goal = int(4e3-1)
if planning_problem_idx == 6:
	start = int(4e3-1)
	goal = int(205e4-1)

plt.plot(XY[start,0], XY[start,1],color='green',linewidth='25',marker='+', markersize=22)
plt.plot(XY[goal,0], XY[goal,1],color='blue',linewidth='25',marker='+', markersize=22)

## Exercise 1.1: graph-based path planning
# You will need to complete the code in
#    search_shortest_path.m

# Euclidean distance function between vertices
def node_dist(v, idx):
    return np.linalg.norm(XY[v,:]-XY[idx,:], axis=2)

# perform graph search for shortest path
path, info = search_shortest_path(V, start, goal, reachable, node_dist)

# done, print some info
print('\n')
print('start            : ', info["start"])
print('goal             : ', info["goal"])
print('length of path   : ', info["path_length"]/1000.)
print('costs            : ', info["costs"])
print('backpoint        : ', info["backpoint"])
print('iterations       : ', info["iterations"])
print('vertices in path : ', len(path))

# XY coordinates of the path (indices)
XY_path = XY[path]

# Plot optimal path
visited_vertices = XY[np.where(info["backpoint"])]
plt.scatter(visited_vertices[:,0], visited_vertices[:,1], color='cyan', marker='.')
plt.plot(XY_path[:,0], XY_path[:,1], color='red', marker='.', markersize=4)
plt.show()

## Exercise 1.2: graph-based path planning with A-star
# You will need to complete the code in
#    search_shortest_path_astar.py

# Euclidean distance function between vertices
# Use the Euclidean distance as the hueristic function for A-star
heur_func = node_dist

# perform graph search for shortest path with A-star
path, info = search_shortest_path_astar(V, start, goal, reachable, node_dist, heur_func)

# done, print some info
print('\n')
print('start            : ', info["start"])
print('goal             : ', info["goal"])
print('length of path   : ', info["path_length"]/1000.)
print('costs            : ', info["costs"])
print('backpoint        : ', info["backpoint"])
print('iterations       : ', info["iterations"])
print('vertices in path : ', len(path))

# XY coordinates of the path (indices)
XY_path = XY[path]

# Plot optimal path
plot_network_graph(G, XY)
visited_vertices = XY[np.where(info["backpoint"])]
plt.scatter(visited_vertices[:,0], visited_vertices[:,1], color='cyan', marker='.')
plt.plot(XY_path[:,0], XY_path[:,1], color='red', marker='.', markersize=4)
plt.show()

