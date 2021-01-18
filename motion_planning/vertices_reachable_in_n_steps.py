import numpy as np

def vertices_reachable_in_n_steps(start_idx, reachable, nsteps):
    to_idxs = [] # list of reachable nodes within n steps
    from_idxs = [] # for each node in to_idxs, a node to get to it

    # The list of nodes to explore, initially only the start node
    expand_idxs = [start_idx]

    # iterate over the given number of time steps
    for step in range(nsteps):

        new_idxs = []
        for from_idx in expand_idxs:

            # reachable nodes in graph from current node 'from_idx'
            nidxs = reachable[int(from_idx)]

            # repeat 'from_idx' for each reachable node
            from_idx = np.tile(from_idx[..., None], (1, nidxs.size))

            # ok, store the (from, to) pairs
            from_idxs = np.append(from_idxs, from_idx)
            to_idxs = np.append(to_idxs, nidxs)

            # keep track of the newly reached nodes
            new_idxs = np.append(new_idxs, nidxs)

        # Remove any previously reached nodes from the new nodes, and use these to expand further in the new time step
        expand_idxs = np.setdiff1d(new_idxs, from_idxs)

    return to_idxs, from_idxs
