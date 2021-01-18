import math
import numpy as np

def calculate_distance(A:list, B:list) -> float:
    x = A[0]-B[0]
    y = A[1]-B[1]
    dist = np.sqrt(x**2+y**2)
    return dist

def calculate_orientation(A,B):
        return np.abs(A-B)

# Evaluate a candidate trajectory by comparing each timestep
def compute_trajectory_cost(obstacle_states, states, lat_offset):
    
    alpha=60
    beta=1
    # number of timesteps
    nsteps = min(states.size, obstacle_states.size)

    costs_per_timestep = np.ones([nsteps]) * np.nan
    min_dist = 3
    gamma=200
    # -- Exercise 3.5: Compute the cocsts of candidate trajectories -- #
    # compute cost for each time step
    for step in range(nsteps):

        # Here we can compute costs_per_timestep(step), e.g. based on the
        # distance between our vehicle state, states(step),
        # and the obstacle state, obstacle_states(step).
        # What do you think would be a good cost function?

        # Todo: implement your own cost here
        # costs_per_timestep[step] = 0  # Dummy
        A = [states[step].x, states[step].y]
        B = [obstacle_states[step].x, obstacle_states[step].y]
        cost = alpha*math.exp(-1*calculate_distance(A,B))+beta*calculate_orientation(states[step].theta, obstacle_states[step].theta) + gamma*max((min_dist-calculate_distance(A,B)),0)
        # print(step,":",calculate_distance(A,B))
        # print(step,":",calculate_distance(A,B),"||",calculate_orientation(states[step].theta, obstacle_states[step].theta),"||",calculate_distance(A,B)<min_dist,"||",cost)
        # print(alpha*math.exp(-1*calculate_distance(A,B)), beta*calculate_orientation(states[step].theta, obstacle_states[step].theta),gamma*max((min_dist-calculate_distance(A,B)),0))
        costs_per_timestep[step] = cost##alpha*math.exp(-1*calculate_distance(A,B))+beta*calculate_orientation(states[step].theta, obstacle_states[step].theta) + gamma*np.max(min_dist-calculate_distance(A,B),0)
#########################
## YOUR_CODE_GOES_HERE ##
#########################

    # Finally, we compute a single cost from the cost per timestep,
    # and also a cost for the lateral offset.
    # Note that you might need to do some weighting of the different
    # cost terms to get the desired results, depending on how you define
    # the terms.
    total_cost = np.array([0])  # dummy
    total_cost = np.average(costs_per_timestep)
    gamma=0.75
    total_cost = total_cost + gamma*((lat_offset))

#########################
## YOUR_CODE_GOES_HERE ##
#########################
    # -------------------------------------------------------------- #

    # Some functions to test the cost function
    assert(costs_per_timestep.size == nsteps)
    assert(total_cost.size == 1)

    # Return the total cost and the cost per time step
    return total_cost, costs_per_timestep



