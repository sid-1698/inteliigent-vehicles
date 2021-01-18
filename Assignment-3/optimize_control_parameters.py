import scipy.optimize
from curv_param_state_predict import *


def optimize_control_parameters(init_state, goal_state, dt, init_params):

    # Run non-linear least-squares optimization to find parameter p in
    #    p = argmin_{p} |C(p)|^2
    # using gradient descent.
    # We will use the scipy optimization package
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    # Solves the LSQ problem
    # args - are the arguments fed into the cost function defined below
    # loss - linear loss is the standard LSQ cost function
    # vebose - determines the level of output in the console (use 1 or 0 for less output)
    # max_nfev - limits the number of function evaluations
    params = scipy.optimize.least_squares(cost, init_params, args=(init_state, goal_state, dt),
                                          loss='linear',
                                          verbose=2,
                                          max_nfev=100)

    # Return the optimized paramers
    return params.x


# this is the error function
def cost(params, init_state, goal_state, dt):

    # -- Exercise 3.4: Compute the errors (residuals) to optimize -- #
    # This function should return a vector of errors to minimize,
    # given the control parameters.
    #
    # NOTE that variables `init_state` `goal_state` and `dt` are accessible from within this function
    #
    # The predicted vehicle states can be obtained with
    #     curv_param_state_predict.
    #
    # Compute the error of x, y and theta between the
    # final predicted state and the goal state.

    # Todo: Implement your error value here
    err = 0  # Dummy
    XT = curv_param_state_predict(init_state, dt, params)[-1]
    err = np.array([XT.x - goal_state.x, XT.y - goal_state.y, XT.theta - goal_state.theta])

#########################
## YOUR_CODE_GOES_HERE ##
#########################
    # ----------------------------------------------------------------- #
    return err
