import numpy as np

def make_discrete_space(dim_min, dim_max, dim_size, circular_dims):
	# ensure column vectors
	dim_min = dim_min.flatten('F')
	dim_max = dim_max.flatten('F')
	dim_size = dim_size.flatten('F')

	D = dim_size.size
	assert D == dim_min.size
	assert D == dim_max.size

	# determine which dimensions are bounded, and which are circular
	mask_dim_modulo = np.zeros(D, dtype=bool)
	mask_dim_modulo[np.argwhere(circular_dims)] = True

	# create cell indices
	dim_width = (dim_max - dim_min) / dim_size

	# create cell-array storing per dimension the cell centers
	def calculate_dim_centers(d):
		return np.linspace(dim_min[d] + 0.5*dim_width[d], dim_max[d] - 0.5*dim_width[d], dim_size[d])

	dim_centers = []
	for i in range(D):
		dim_centers.append(calculate_dim_centers(i))
	dim_centers = np.array(dim_centers, dtype='object')

	# precompute stepsize for N-d to 1-d conversion
	dim_steps = np.cumprod( np.append(1, dim_size[:-1]) )
	num_cells = np.prod(dim_size)

	grid = np.meshgrid(*dim_centers)
	cell_centers = []
	for dim in grid:
		cell_centers.append(dim.ravel(order='F'))
	cell_centers = np.array(cell_centers)
	cell_centers[[0, 1]] = cell_centers[[1, 0]] # Swap rows for matlab compatibility

	# create result dict
	sp = Sp(dim_size, dim_width, dim_centers, cell_centers, num_cells, D, dim_min, dim_max, dim_steps)

	return sp

class Sp():

	def __init__(self, dim_size, dim_width, dim_centers, cell_centers, num_cells, num_dims, dim_min, dim_max, dim_steps):
		self.dim_size = dim_size
		self.dim_width = dim_width
		self.dim_centers = dim_centers
		self.cell_centers = cell_centers
		self.num_cells = num_cells
		self.num_dims = num_dims
		self.dim_min = dim_min
		self.dim_max = dim_max
		self.dim_steps = dim_steps

	# -- helper functions ---
	def input_to_1d(self, inp):
		idxn = self.input_to_nd(inp)
		idx1 = self.map_nd_to_1d(idxn)
		return idx1

	def input_to_1d_NaN(self, inp):
		idxn = self.input_to_nd_NaN(inp)
		idx1 = self.map_nd_to_1d(idxn)
		return idx1

	def input_to_nd(self,data_in):
		# first, convert all dimensions to [0, 1] range
		data_idx = data_in
		data_idx = np.transpose(np.transpose(data_idx) - self.dim_min) # subtract minimum value
		data_idx = np.transpose(np.transpose(data_idx) / (self.dim_max - self.dim_min)) # divide by minmax range
		data_idx = np.clip(data_idx, 0, 1) # clamp in case min/max bounds are within data range

		# now change [0, 1] range to [1, ... dim_size] range
		data_idx = np.transpose(np.transpose(data_idx) * self.dim_size)
		data_idx = np.ceil(data_idx)
		data_idx[data_idx == 0] = 1 # index was 0 possible if data_in <= table_min
		return data_idx

	def input_to_nd_NaN(self, data_in):
		# first, convert all dimensions to [0, 1] range
		data_idx = data_in
		data_idx = np.transpose(np.transpose(data_idx) - self.dim_min) # subtract minimum value
		data_idx = np.transpose(np.transpose(data_idx) / (self.dim_max - self.dim_min)) # divide by minmax range
		data_idx[data_idx < 0] = np.nan
		data_idx[data_idx > 1] = np.nan

		# now change [0, 1] range to [1, ... dim_size] range
		data_idx = np.transpose(np.transpose(data_idx) * self.dim_size)
		data_idx = np.ceil(data_idx)
		data_idx[data_idx == 0] = 1 # index was 0 possible if data_in <= table_min
		return data_idx



	# N-d to 1-d index conversion
	def map_nd_to_1d(self, idxn):
		idx1 = np.sum(np.transpose(np.transpose(idxn-1) * self.dim_steps), axis=0)+1
		return idx1

