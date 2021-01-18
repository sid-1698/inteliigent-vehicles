from setup_trajectory_scenario import *
import numpy as np
import math
from state_types import vehicle_state

def trajplanning_obstacle_scenario(road, scenario_idx):

    rradius = road.rradius
    rlong = road.rlong

    if scenario_idx is 1:
            name = 'normal vehicle on other lane'
            [ox, oy, otheta] = make_road_xy(rradius, rlong[::-1], -4)

    elif scenario_idx is 2:
            name = 'normal vehicle moving in front'
            [ox, oy, otheta] = make_road_xy(rradius, rlong + 17., 0.)
            otheta = np.pi+otheta

    elif scenario_idx is 3:
            name = 'moving in front (too slow)'
            [ox, oy, otheta] = make_road_xy(rradius, rlong / 2. + 17., 0.)
            otheta = np.pi+otheta

    elif scenario_idx is 4:
            name = 'approaching too close to center'
            [ox, oy, otheta] = make_road_xy(rradius, rlong[::-1], -2.)

    elif scenario_idx is 5:
            name = 'madman 1 crossing'
            [ox, oy, otheta] = make_road_xy(-rradius, rlong[::-1], -4.)
            ox = ox + 9 * np.sign(rradius)

    elif scenario_idx is 6:
            name = 'madman 2 crossing'
            [ox, oy, otheta] = make_road_xy(-rradius, rlong[::-1], -4.)
            ox = ox + 14. * np.sign(rradius)

    else:
        raise Exception('trajplanning_obstacle_scenario: incorrect obstacle option')

    print('[scenario %d / 6] %s' % (scenario_idx, name))

    obstacle_states = np.array([])
    for t in range(ox.size):
        obstacle_states = np.append(obstacle_states, vehicle_state(ox[t], oy[t], otheta[t] + np.pi, 0., 5., 0.))

    return obstacle_states
