from setup_trajectory_scenario import *
import copy

class vehicle_state:

    def __init__(self, x, y, theta, kappa, v, a):
        self.x = x
        self.y = y
        self.theta = theta
        self.kappa = kappa
        self.v = v
        self.a = a

    # -- define the vehicle dynamics --
    # See Algorithm 3 [Ferguson, 2008]
    # state: [x, y, theta, kappa, v, a]
    #   x     : vehicle positional x
    #   y     : vehicle positional y
    #   theta : vehicle orientation
    #   kappa : curvature
    #   v     : velocity
    #   a     : acceleration
    # control:
    #   u_acc   : acceleration command
    #   u_steer : steering angle command
    def motion_model(self, u_acc, u_steer, dt):
        self.x += self.v * np.sin(self.theta) * dt
        self.y += self.v * np.cos(self.theta) * dt
        self.theta += self.v * self.kappa * dt
        self.kappa = u_steer
        self.v += self.a * dt
        self.a = u_acc


def get_goal_state(road, long_frac, lat_offset, init_state):
    target_x, target_y, target_theta = make_road_xy(road.rradius, road.rlength * long_frac, lat_offset)

    new_state = copy.copy(init_state)
    new_state.x = target_x
    new_state.y = target_y
    new_state.theta = target_theta
    return new_state


# A class representing a candidate trajectory
class candidate:
    def __init__(self, lat_offset, params, states):
        self.lat_offset = lat_offset
        self.params = params
        self.states = states
