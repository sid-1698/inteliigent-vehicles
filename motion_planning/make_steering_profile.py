import random
import scipy.interpolate
import numpy as np
import numpy.random

class steering_profile:

    # This dummy function shows a dummy example of the steering profile.
    # It just returns same random output independent of the input values
    def steering_profile_function_dummy(self, sfrac):
        sfrac = np.atleast_1d(sfrac)

        return np.squeeze(np.random.random_sample((sfrac.size)) - .5)   # Not a good steering function

    # -- Exercise 3.1: Define a steering function as cubic spline -- #
    # Define the steering profile
    # Input:
    #   - k0, k1, k2 : the spline control inputs at a fraction of 0, .5 and 1
    #                  (i.e. 0#, 50# and 100#) of the traveled distance only
    #                  the road segment
    # Output:
    #   - steering_profile : a *function* which maps sfrac to steering angles
    #         if given a 1xN vector sfrac with N fractions between 0 and 1,
    #         steering_profile(sfrac) should return a 1xN vector with
    #         corresponding steering angles.
    #
    # Note that the profile function should satisfy the following properties:
    #     steering_profile(0) == k0
    #     steering_profile(.5) == k1
    #     steering_profile(1) == k2
    #     steering_profile(0.25) # a value between k0 and k1
    #     steering_profile([0 .5]) == [k0 k1]
    #
    # Tip: use scipy's 'scipy.interpolate.CubicSpline' function
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    def steering_profile_function_spline(self, sfrac):
        steering_profile = self.steering_profile_function_dummy(sfrac) # dummy function until spline is implemented
        x = [0,0.5,1]
        y = [self.k0, self.k1, self.k2]
        steering_profile = scipy.interpolate.CubicSpline(x,y)

        return steering_profile(sfrac)
        # Dummy to prevent errors
        # Return an array of angle

    # -------------------------------------------------------------- #

    # The initialization of the steering profile
    def __init__(self, k0, k1, k2, validate_steering_function=False):

        # Save parameters
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2

        # Currently assigns a dummy function
        self.steering_profile_function = self.steering_profile_function_spline

        if validate_steering_function:
            # Check that the function returns a steering angle for
            self.debug_test_equal(self.steering_profile_function, 0, self.k0)
            self.debug_test_equal(self.steering_profile_function, .5, self.k1)
            self.debug_test_equal(self.steering_profile_function, 1, self.k2)
            self.debug_test_equal(self.steering_profile_function, np.array([1, 0, .5]),
                                  np.array([self.k2, self.k0, self.k1])) # should also work out of order


    # Test that two vectors x and y are almost equal
    def debug_test_equal(self, steering_profile_function, sfrac, true_out):
        true_out = np.atleast_1d(true_out)
        out = steering_profile_function(sfrac)
        out = np.atleast_1d(out)
        if not all(np.atleast_1d(len(out) == len(true_out))):
            print('Warning: steering_profile: different number of actual outputs and expected outputs')
            return

        if not all(np.atleast_1d(len(out) == len(true_out))):
            print('Warning: steering_profile: actual output and expected output have different sizes')
            return

        if not all(np.atleast_1d(abs(out - true_out) < 1e-10)):
            print('Warning: steering_profile: actual output and expected output differ too much')
            return

        # All okay
