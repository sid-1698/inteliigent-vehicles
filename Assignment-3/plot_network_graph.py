import numpy
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def plot_network_graph(G, XY):
    # G: n x n connectivity matrix
    # XY: n x 2 spatial position (2D) of each node
    indices = numpy.argwhere(G)

    # G is symmetric, meaning that 'indices' also contains each connection twice: From A to B is the same as from B to A, so that can be filtered to improve on efficiency
    indices = indices[indices[:,0] < indices[:,1]]

    # Line segments, defined with start and end points
    XY_start_points  = XY[indices[:,0]]
    XY_end_points    = XY[indices[:,1]]

    # Many line segments can be plotted efficiently using a 'line collection'
    line_segments = []
    for i in range(len(indices)): # Data needs to be converted to the right type and shape for a 'line collection'
        line_segments.append([numpy.array(XY_start_points[i,:]), numpy.array(XY_end_points[i,:])])
    lc = mc.LineCollection(line_segments, colors=(0.5,0.5,0.5,0.5), linewidths=1)

    # Actual plot
    fig, ax = plt.subplots(figsize=(5,5))
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
