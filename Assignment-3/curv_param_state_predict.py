import math
import numpy as np
from state_types import *
import make_steering_profile
from euler_integrate_motion import *

def curv_param_state_predict(init_state, dt, params):

    k0 = init_state.kappa

    # -- Exercise 3.3: Complete the mapping from control parameters to predicted states -- #
    # Create the steering profile function here,
    #  and then perform Euler integration.
    #  Use make_steering_profile and euler_integrate_motion.
    # Todo: Compute your states vector here
    states = np.array([init_state])  # Dummy
    steer_profile = make_steering_profile.steering_profile(k0, params[0], params[1])
    states = euler_integrate_motion(init_state, dt, params[2], steer_profile)
#########################
## YOUR_CODE_GOES_HERE ##
#########################
    # --------------------------------------------------------------------------------------- #

    # Return the resulting states
    return states
