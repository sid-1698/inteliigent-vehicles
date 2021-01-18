import numpy as np

# -- Exercise 2.3: Remove occupied vertices from all the lists of reachable vertices. -- #
#
# This function removes edges that connect 'non-occupied' vertices
#   to 'occupied' vertices, given the original list of reachable vertices
#   and information on which vertices are 'occupied'.
#
# Input:
#  - reachable: a 1 x V cell array
#               reachable{v} contains the list of vertex indices of the
#               neighboring vertices reachable from vertex index v.
#               For example, reachable{1} = [4, 7] means that there is
#               are two edges from vertex 1, namely to vertex 4 and to 7.
#  - state_occupancy: a (4D) array containing values 0 and 1 indicating.
#               For example, if state_occupancy(3,6,4,1) == 0, then
#               cell with 4D index [3,6,4,1] is not occupied.
#               NOTE: temp_occupancy is a converted list so that you
#               can use a vertex index directly on state_occpancy,
#               i.e. state_occupancy(3235) gives the
#               state occupancy of vertex 3235.
#
# Output:
#  - reachable: a 1 x V cell array
#               similar to the input, but now vertices that are
#               occupied have been removed from each list of reachable
#               vertices.
def remove_unreachable_cells(reachable, state_occupancy):

    V = len(reachable)
    to_delete = []
    # Convert the occupancy array into a 1D array for easier indexing
    temp_occupancy = np.reshape(state_occupancy, (V), order='F')
    for v in range(V):
        if temp_occupancy[v] == 1:
            to_delete.append(v)
    reachable = np.delete(reachable,to_delete)
    reachable = list(reachable)
    # Loop over all vertices v, and remove from the list reachable{v}
    #   all neighboring vertex indices for which state_occupancy is true.

#########################
## YOUR_CODE_GOES_HERE ##
#########################
    return reachable
