# Compute shortest path from start to goal in a graph
#
# Input:
# - V         : number of vertices
# - start     : vertex idx of start position
# - goal      : vertex idx of goal position
# - cost_func : cost function g, computes cost of edges between vertices
# - reachable : A cell array that lists per vertex all neighboring other vertices
#               i.e. reachable{i} is vector with vertex indices that can be reached from vertex i
#
# Output:
# - path      : list of vertices that defines the shortest path from start to
#               goal
# - info      : a struct with some information
#

import numpy as np

def search_shortest_path(V, start, goal, reachable, cost_func):

    # initialize
    backpoint = np.full([V],None)
    backpoint[start] = 0

    costs = np.full([V], np.inf)
    costs[start] = 0
    heurs = np.zeros(V)
    scores = np.zeros(V)

    queued_idxs = np.full(1, start) # the "frontier"
    # explore the graph until goal has been reached ...
    for iter in range(V*2):
        assert queued_idxs.shape!=0, "goal could not be reached"

        # get next vertex to explore from frontier,
        # i.e. the vertex with lowest COST
        ja = np.argmin(costs[queued_idxs])
        v = queued_idxs[ja]
        queued_idxs = np.delete(queued_idxs, ja)

        # test: did we reach the goal?
        if v == goal:
            print('goal reached')
            break

        # find 'neighbors' of current vertex, i.e. those connected with an edge
        nidxs = reachable[v] # Shape changes per iteration?

        # compute the distances and scores for the neighbors
        cost = costs[v] # distance of current vertices
        ncosts = cost + cost_func(v, nidxs) # distances for each neighbor

        # only keep neighbors for which we can improve their distance
        mask = np.squeeze(np.less(np.transpose(ncosts), np.squeeze(costs[nidxs])))
        nidxs = nidxs[mask]
        ncosts = ncosts[mask]

        # append neighbors to frontier
        queued_idxs = np.append(queued_idxs, nidxs)

        # update values of neighbors in backpoint log
        backpoint[nidxs] = v # update back pointing log
        costs[nidxs] = np.reshape(ncosts, costs[nidxs].shape)


        # debug info
        if iter % 10000 == 0:
            # show some debug information
            print('iter ', iter, ' v ', v, ' cost ', cost)

    # ------------------------------
    # Exercise 1.1
    # ------------------------------
    # Backtracking
    #   Now that the goal has been found, we can reconstruct the shortest
    #   path by backtracking from goal to start following the `backpoint'
    #   links.
    #
    #  It performs the following steps:
    #   1. Initialize empty list, and set idx at goal vertex
    #   2. While idx not is 0 (the backpoint value for start) ...
    #       3. add idx to path
    #       4. update idx to backpoint(idx)
    #   5. If in step 3 each idx was appended to path, then the recovered 
    #      path currently moves from [goal, ..., start]. In that case
    #      reverse it such that it goes from [start, ..., goal].

    # Initialize
    v = goal
    path = []

    while v != 0:
        v = backpoint[v]
        path.append(v)
    
    path = path[::-1]

    # Recursively follow the pointer to the shortest path
#########################
## YOUR_CODE_GOES_HERE ##
#########################

    # ok, done
    info = dict()
    info["start"] = start
    info["goal"] = goal
    info["path_length"] = costs[goal]
    info["costs"] = costs
    info["backpoint"] = backpoint
    info["iterations"] = iter

    return path, info
