import numpy as np
import matplotlib.pyplot as plt
from state_types import *

# Useful function for not showing figures
def check_enabled(func):
    def wrapper(self, *args, **kwargs):
        if not self.disabled:
            return func(self, *args, **kwargs)

    return wrapper

class steering_animation_figure:

    def __init__(self, disabled=False):

        self.disabled = disabled

        if not disabled:
            self.fig = plt.figure(2, figsize=(10, 5))
            self.ax1 = plt.subplot(1, 2, 1)
            self.ax2 = plt.subplot(1, 2, 2)
            self.ax2.set_aspect('equal', adjustable='box')

            plt.ion()
            plt.show()

    @check_enabled
    def save(self, filename):
        plt.savefig(filename + '.pdf')

    @check_enabled
    def draw(self):
        #plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.001)

    @check_enabled
    def close(self):
        plt.clf()
        plt.close(self.fig)

    @check_enabled
    def clear(self):
        self.ax1.clear()
        self.ax2.clear()


    # Plot the steering profile, given the three control points, and animate
    # the effect on the vehicle.
    @check_enabled
    def animate_steering_profile_plot(self, steering_profile):
        # longitudinal fraction (i.e. how far are we along road)
        long_fracs = np.linspace(0, 1, 25)

        # steering angle control inputs
        steering_angles = steering_profile.steering_profile_function(long_fracs)

        for j in range(long_fracs.size):
            self.ax1.plot(long_fracs, steering_angles, 'k-', label='spline')

            self.ax1.plot(long_fracs[j], steering_angles[j], 'k*')

            self.ax1.plot(0, steering_profile.k0, 'r.', markersize=20, label='k0')
            self.ax1.plot(.5, steering_profile.k1, 'g.', markersize=20, label='k1')
            self.ax1.plot(1, steering_profile.k2, 'b.', markersize=20, label='k2')
            self.ax1.legend()
            self.ax1.grid()
            self.ax1.set_ylim(-1.1*np.pi, 1.1*np.pi)

            self.ax1.set_xlabel('fraction of travelled distance')
            self.ax1.set_ylabel('steering angle')
            self.ax1.set_yticks(np.array([-1, -.5, 0, .5, 1])*np.pi)

            self.plot_vehicle_state(self.vehicle_state_from_angle(steering_angles[j]))
            self.ax2.set_xlim(-2, 2)
            self.ax2.set_ylim(-2, 2)
            self.ax2.grid()

            self.draw()

            if j != long_fracs.size - 1:
                self.clear()

    @check_enabled
    def vehicle_state_from_angle(self, steer_angle):
        return vehicle_state(0., 0., 0.*np.pi, steer_angle, 5., 0.)

    @check_enabled
    def box_in_frame(self, cx, cy, w, h, R, T, **kwargs):
        # car outset
        points = np.array([[1, -1, -1, 1, 1], [-1, -1, 1, 1, -1]]).astype(float)

        points[0, :] = points[0, :] * (float(w) / 2.) + cx
        points[1, :] = points[1, :] * (float(h) / 2.) + cy
        self.plot_in_frame(points, R, T, **kwargs)

    @check_enabled
    def plot_in_frame(self, points, R, T, **kwargs):
        # Apply transformation
        points = R.dot(points)

        plt.plot(points[0, :] + T[0], points[1, :] + T[1], **kwargs)

    @check_enabled
    def plot_vehicle_state(self, s, **kwargs):
        # rotation, translation
        ct = np.cos(s.theta)
        st = np.sin(s.theta)
        R = np.array([[ct, st],
                      [-st, ct]])
        T = np.array([s.x, s.y])

        slong = 4.5  # longitudinal size(meter)
        slat = 2.  # lateral size(meter)

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
