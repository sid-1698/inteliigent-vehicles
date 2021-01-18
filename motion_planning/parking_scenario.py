import numpy as np
from math import pi
from make_discrete_space import make_discrete_space

def parking_scenario():

	## define discrete state-space
	# The used state-space is 4D:
	#
	#       [ pos_x ]  % dim 1: vehicle x position
	#       [ pos_y ]  % dim 2: vehicle y position
	#   x = [ theta ]  % dim 3: vehicle orientation
	#       [ omega ]  % dim 4: angle of wheels
	#

	# define how many cells we use per dimension
	spatial_size = 32
	theta_size = 32
	dim_size = np.asarray([spatial_size, spatial_size, theta_size, 3])

	# define the minimum and maximum value of continuous space
	dim_min = np.asarray([ 0,  0, 0*pi-pi/theta_size, -pi/6])
	dim_max = np.asarray([10, 10, 2*pi-pi/theta_size, +pi/6])

	# which of the dimensions is/are on a circular domain?
	circular_dims = 3

	# create a struct with utility function for this discrete space,
	# namely to determine the number of cells, map from the continuous space
	# to discrete cell indices, and back, etc.
	sp = make_discrete_space(dim_min, dim_max, dim_size, circular_dims)

	states = np.transpose(sp.cell_centers) # cell centers
	V = sp.num_cells # total number of cells
	D = sp.num_dims # number of dimensions (D=4)

	## Determine which grid cells can be reached from which other grid cells

	# Note: at the end, we also add edges (b,a) for each edge (a,b). This means
	# that we only have to consider 'moving forward' and 'steering right' to
	# automatically also add 'moving backward' and 'steering left'.

	moves = []
	vels = [2 * sp.dim_width[0]]
	dkappas = [1 * sp.dim_width[3]]

	# -- compute per cell the effect of a move ---
	# Note that some moves may result in a position outside the grid world.
	# In this case the moved to cell indices will be 'NaN' which we remove
	# in the next step.

	# driving forward/backward: velocity control (uses vehicle motion model)
	for vel in vels:
		new_states = np.copy(states)
		new_states[:,0:2] = new_states[:,0:2] + np.transpose(np.vstack((np.sin(states[:,2]), np.cos(states[:,2]))) * vel)
		new_states[:,2] = new_states[:,2] + new_states[:,3] * vel

		moves.append(sp.input_to_1d_NaN(np.transpose(new_states)))

	# steering control (no velocity)
	for dk in dkappas:
		new_states = np.copy(states)
		new_states[:,3] = new_states[:,3] + dk

		moves.append(sp.input_to_1d_NaN(np.transpose(new_states)))

	moves = np.asarray(moves)
	moves = moves - 1

	# -- create a list of reachable cells, removing any NaN cell indices --
	reachable = []
	for idx in range(V):
		nidx = moves[:,idx]
		nidx = nidx[~np.isnan(nidx)]
		nidx = nidx[nidx!=idx]
		reachable.append(nidx.astype('int'))

	## if node A is reachable from B, then B is also reachable from A!
	for idx in range(V):
		nidxs = reachable[idx]
		for nidx in nidxs:
			if not np.any(reachable[int(nidx)] == idx):
				reachable[int(nidx)] = np.append(reachable[int(nidx)], idx)

	return sp, reachable
