import numpy as np
import matplotlib.pyplot as plt

# Useful function for not showing figures
def check_enabled(func):
    def wrapper(self, *args, **kwargs):
        if not self.disabled:
            return func(self, *args, **kwargs)

    return wrapper

class vehicle_struct:

    def __init__(self, states, idx):
        self.x = states[0, idx]
        self.y = states[1, idx]
        self.theta = states[2, idx]
        self.kappa = states[3, idx]
        self.v = 0
        self.a = 0


class figure_2d:

    def __init__(self, idx=1, disabled=False):

        self.disabled = disabled

        if not self.disabled:
            self.fig = plt.figure(idx, figsize=(5, 5))
            self.ax = self.fig.gca()
            self.ax.set_aspect('equal', adjustable='box')

            plt.ion()
            plt.show()

    @check_enabled
    def box_in_frame(self, cx, cy, w, h, R, T, **kwargs):
        # car outset
        points = np.array([[1, -1, -1, 1, 1],[-1, -1, 1, 1, -1]]).astype(float)

        points[0, :] = points[0, :] * (float(w) / 2.) + cx
        points[1, :] = points[1, :] * (float(h) / 2.) + cy
        self.plot_in_frame(points, R, T, **kwargs)

    @check_enabled
    def plot_in_frame(self, points, R, T, **kwargs):

        # Apply transformation
        points = R.dot(points)

        plt.plot(points[0, :] + T[0], points[1, :] + T[1], **kwargs)

    def minmax(self, vec):
        return np.array([min(vec), max(vec)])

    @check_enabled
    def plot_setup_groundplane_2d(self, cspace, occupancy_2d):

        minmax_x = self.minmax(cspace.dim_centers[0]) + np.array([-1, 1])*cspace.dim_width[0]/2
        minmax_y = self.minmax(cspace.dim_centers[1]) + np.array([-1, 1])*cspace.dim_width[1]/2

        plt.xlim(minmax_x[0] - 2, minmax_x[1] + 1)
        plt.ylim(minmax_y[0] - 2, minmax_y[1] + 1)

        plt.xlabel('x (meter)')
        plt.ylabel('y (meter)')

        # plot the outline of the 2D groundplane space (i.e. where car center can be located
        plt.plot(minmax_x[np.array([0, 0, 1, 1, 0])], minmax_y[np.array([0, 1, 1, 0, 0])], 'k--')

        self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_xticks(np.arange(minmax_x[0], minmax_x[1]+1, 2))
        self.ax.set_yticks(np.arange(minmax_y[0], minmax_y[1]+1, 2))
        plt.grid()

        # Plot occupied cells

        [ox, oy] = np.array(np.where(occupancy_2d == 1))
        ox = cspace.dim_centers[0][ox] # grid cell to spatial coordinates
        oy = cspace.dim_centers[1][oy]
        plt.scatter(ox, oy, 50, 'k', label='Collision Area')

    @check_enabled
    def save(self, filename):
        plt.savefig(filename + '.pdf')

    @check_enabled
    def draw(self, pause=0.000001, blocking=False):
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(pause)

        if blocking:
            plt.pause(10000)

    @check_enabled
    def close(self):
        plt.clf()
        plt.close(self.fig)

    @check_enabled
    def clear(self):
        self.fig.clear()

    @check_enabled
    def plot_path(self, states, path):
        plt.plot(states[0, path], states[1, path],
                 markersize=5, label='Path', marker='o', linestyle='solid', color='gray')

    @check_enabled
    def plot_vehicle_state(self, s, **kwargs):

        # rotation, translation
        ct = np.cos(s.theta)
        st = np.sin(s.theta)
        R = np.array([[ct, st],
                      [-st, ct]])
        T = np.array([s.x, s.y])

        slong = 4.5 # longitudinal size(meter)
        slat = 2. # lateral size(meter)

        if 'color' in kwargs:
            self.box_in_frame(0., 0., slat, slong, R, T, **kwargs)  # car outset
            self.box_in_frame(0., slong * .05, slat * .8, slong * .2, R, T, **kwargs)  # front windshield
            self.box_in_frame(0., slong * -.25, slat * .8, slong * .15, R, T, **kwargs)  # back window
        else:
            self.box_in_frame(0., 0., slat, slong, R, T, color='black', label='Vehicle', **kwargs)  # car outset
            self.box_in_frame(0., slong * .05, slat * .8, slong * .2, R, T, color='black', **kwargs)  # front windshield
            self.box_in_frame(0., slong * -.25, slat * .8, slong * .15, R, T, color='black', **kwargs)  # back window

        # wheel angle
        kappa_mult = 1
        kct = np.cos(s.kappa * kappa_mult)
        kst = np.sin(s.kappa * kappa_mult)
        kR = np.array([[kct, kst],
                    [-kst, kct]])

        points = np.array([[0., 0.],
                            np.array([-.2, .2]) * slong])

        points_left = kR.dot(points) + np.array([[-.35 * slat, .3 * slong], [-.35 * slat, .3 * slong]]).transpose()
        points_right = kR.dot(points) + np.array([[.35 * slat, .3 * slong], [.35 * slat, .3 * slong]]).transpose()

        if 'color' in kwargs:
            self.plot_in_frame(points_left, R, T, linewidth=2, **kwargs)
            self.plot_in_frame(points_right, R, T, linewidth=2, **kwargs)
        else:
            self.plot_in_frame(points_left, R, T, color='red', linewidth=2, **kwargs)
            self.plot_in_frame(points_right, R, T, color='red', linewidth=2, **kwargs)
