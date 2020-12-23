import numpy as np
from define_occupancy_map import define_occupancy_map
from Sensor import Sensor

# a slightly more complicated vehicle model
def motion_model(s, u, dt):
    return {
            'x':     s['x'] + s['v'] * np.sin(s['theta']) * dt, 
            'y':     s['y'] + s['v'] * np.cos(s['theta']) * dt,
            'theta': s['theta'] + s['v'] * s['kappa'] * dt,
            'kappa': u['kappa'],
            'v':     s['v'] + s['a'] * dt,  # <-- NOTE: different from Algorithm 3
            'a':     u['a']
            }

def selfloc_scenario():
    map_ = define_occupancy_map()

    # vehicle state
    veh_init = {
                'x' : 0.,
                'y' : 10.,
                'theta' : 0.,
                'kappa' : 0.,
                'v' : 1.,
                'a' : 0.
                }

    # control input over T timesteps
    T = 90
    dt = .5
    u_as = np.zeros(T)
    u_ks = np.zeros(T)
    u_ks[19:25] = -np.pi/6
    u_ks[41:47] = -np.pi/6
    u_ks[64:70] = +np.pi/6

    # simulate vehicle motion and observations
    veh = veh_init
    control_inputs = np.zeros((2,T)) - 9999.  # NaN(2, T)
    vehicles, measurements, sensors = [], [], []
    for t in range(T):
        veh = motion_model(veh, {'a':u_as[t], 'kappa':u_ks[t]}, dt)

        # create range sensor on vehicle
        M = 16*2
        sensor = Sensor(0, 2*np.pi, M, 25, 5e-1, 1, veh)
        meas = sensor.new_meas()
        meas = sensor.observe_point(meas=meas, pos=list(map_['obstacles']), radius=1.)

        vehicles.append(veh)
        sensors.append(sensor)
        measurements.append(meas)
        control_inputs[:,t] = np.array([veh['v'], veh['kappa']])

    # put everything in a convenient struct
    scenario = {
                'T' : T,
                'dt' : dt,
                'map' : map_,
                'sensors' : sensors,
                'measurements' : measurements,
                'vehicles' : vehicles,
                'control_inputs' : control_inputs
                }

    return scenario
