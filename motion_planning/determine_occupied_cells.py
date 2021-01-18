import numpy as np
import scipy.signal

try:
    import skimage.transform
except ImportError:
    raise ImportError("Please install the scikit-image package, using\npython -m pip install scikit-image")


def determine_occupied_cells(cspace, occupancy_2d):
    # define vehicle extent
    veh_long = 4.9
    veh_lat = 2.7
    veh_size = np.round(np.array([veh_lat / cspace.dim_width[0], veh_long / cspace.dim_width[1]])) - 1
    vert_occ = np.ones(veh_size.astype(int))

    padsize_x = np.floor(min(veh_size) + (max(veh_size) - veh_size[0]) / 2).astype(int)
    padsize_y = np.floor(min(veh_size) + (max(veh_size) - veh_size[1]) / 2).astype(int)
    vert_occ = np.pad(vert_occ, ((padsize_x, padsize_x),(padsize_y, padsize_y)), 'constant', constant_values=0)

    # Determine legal states
    theta_size = cspace.dim_size[2] # number of cells in third dim
    state_occupancy = np.tile(occupancy_2d[..., None], (1, 1, theta_size))
    vehicle_occ = np.empty([vert_occ.shape[0], vert_occ.shape[1], theta_size])
    vehicle_occ[:] = np.nan

    for i in range(theta_size):
        # rotate by vehicle occupancy map by state angle
        theta = cspace.dim_centers[2][i] / np.pi * 180.
        theta = round(theta * 10) / 10. # improve precision
        vehicle_occ[:,:, i] = skimage.transform.rotate(vert_occ.astype(float), -theta, clip=True) > 0

        # Remove illegal spatial positions at this angle
        state_occupancy[:,:, i] = scipy.signal.convolve2d(state_occupancy[:,:, i], vehicle_occ[:,:, i], 'same')
        state_occupancy[:,:, i] = state_occupancy[:,:, i] > 0

    state_occupancy = np.tile(state_occupancy[..., None], (1, 1, 1, cspace.dim_size[3]))

    return state_occupancy