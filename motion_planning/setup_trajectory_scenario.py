
import numpy as np

# Define the road/lane following scenario
# Input:
#  - rradius : road curvature radius (meter), rradius --> inf = straight
#
# Output:
#  - road : struct containing the x,y points of the different lines
class road:

    def __init__(self, rradius = 50):

        rlength = 40  # length of road segment (meter)
        rwidth = 4  # width of road / lane (meter)
        
        rsteps = 100 # discretization steps of road representation
        rlong = np.linspace(0, rlength, rsteps) # distance in meters along road
        
        # compute world-coordinates of road left, center and right border
        [rx, ry, rtheta] = make_road_xy(rradius, rlong, np.array([0, -rwidth/2, +rwidth/2, -rwidth*1.5]))
        
        # Set class variables
        self.rradius = rradius
        self.rwidth = rwidth
        self.rlength = rlength
        self.rsteps = rsteps
        self.rlong = rlong
    
        self.rx = rx
        self.ry = ry
        self.rtheta = rtheta


def make_road_xy(rradius, rlong, rlat = np.array([0])):
        
        if np.isinf(rradius):
            # special case
            L = len(rlong)
            D = len(rlat)

            rtheta = np.zeros([1, L])
            rx = rlat * np.ones([1, L])
            ry = np.ones([D, 1]) * rlong
            return

        # compute road center line
        rtheta = rlong / rradius

        # Make sure both variables have 2 dimensions
        rlat = np.expand_dims(np.atleast_1d(rlat), axis=1)
        rtheta = np.expand_dims(np.atleast_1d(rtheta), axis=0)

        rx = (-rradius + rlat) * (np.cos(-rtheta))
        ry = (-rradius + rlat) * (np.sin(-rtheta))
        
        # let (0,0) not be center of turn, but start point of curve
        rx = rx + rradius
        
        return np.squeeze(rx), np.squeeze(ry), np.squeeze(rtheta)
