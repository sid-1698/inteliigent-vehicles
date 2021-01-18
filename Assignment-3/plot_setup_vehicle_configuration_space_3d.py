import numpy as np
import scipy.ndimage as ndimage
from mpl_toolkits.mplot3d import Axes3D
from plot_setup_groundplane_2d import check_enabled

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



class figure_3d:

    def __init__(self, idx=1, disabled=False):
        self.disabled = disabled

        if not self.disabled:
            self.fig = plt.figure(idx, figsize=(5, 5))
            self.ax = plt.axes(projection='3d')

            plt.ion()
            plt.show()

    # Visualize configuration space in 3D
    @check_enabled
    def plot_vehicle_configuration_space(self, cspace, state_occupancy):

        xs = cspace.dim_centers[0]
        ys = cspace.dim_centers[1]
        zs = cspace.dim_centers[2]
        minmax_x = minmax(xs) + np.array([-1, 1])*cspace.dim_width[0]/2.
        minmax_y = minmax(ys) + np.array([-1, 1])*cspace.dim_width[1]/2.
        minmax_theta = minmax(zs) + np.array([-1, 1])*cspace.dim_width[2]/2.

        volume = state_occupancy[:, :, :, 1]

        # Smoothen the surfaces
        data = ndimage.uniform_filter(volume, 3)

        # Then pick from the interpolated surface
        contour = np.where(data > 0.5)

        self.ax.scatter(cspace.dim_centers[0][contour[0]], cspace.dim_centers[1][contour[1]],
                   cspace.dim_centers[2][contour[2]], c=contour[2] * (cspace.dim_width[2]),
                        cmap='viridis', label='Collision Region', edgecolors='black')

        # Limits and labels
        plt.xlim(minmax_x[0], minmax_x[1])
        plt.ylim(minmax_y[0], minmax_y[1])
        self.ax.set_zlim(minmax_theta[0], minmax_theta[1])
        plt.xlabel('x (meter)')
        plt.ylabel('y (meter)')
        self.ax.set_zlabel('Orientation (radians)')

        # Set the camera angle
        self.ax.view_init(elev=35., azim=170)
        self.ax.dst = 4.0

    # @check_enabled
    # def draw(self):
    #     self.ax.legend(loc='upper right')
    #     plt.draw()
    #     plt.pause(10000)
    #     self.fig.show()
    @check_enabled
    def draw(self, pause=0.000001, blocking=False):
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(pause)

        if blocking:
            plt.pause(10000)

    @check_enabled
    def save(self, filename):
        plt.savefig(filename + '.pdf')

    @check_enabled
    def clear(self):
        plt.clf()

    @check_enabled
    def close(self):
        plt.clf()
        plt.close(self.fig)

    @check_enabled
    def plot_vehicle_state_configuration_space(self, cspace, idxs, **kwargs):
        # Show position
        coords = cspace.cell_centers[0:3, idxs]
        self.ax.plot3D(coords[0], coords[1], coords[2], **kwargs)

    @check_enabled
    def plot_edges_configuration_space(self, cspace, from_idxs, to_idxs, **kwargs):
        assert from_idxs.shape == to_idxs.shape

        from_xyz = cspace.cell_centers[0:3, from_idxs.astype(int)].transpose()
        to_xyz = cspace.cell_centers[0:3, to_idxs.astype(int)].transpose()

        [N, D] = from_xyz.shape

        pos = np.empty((N*3, 3), dtype=from_xyz.dtype)

        # Interlace vectors
        nan_array = np.ones([N, D]) * np.nan
        pos[0::3, :] = from_xyz
        pos[1::3, :] = to_xyz
        pos[2::3, :] = nan_array

        self.ax.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], **kwargs)


def minmax(vec):
    return np.array([min(vec), max(vec)])





