import numpy as np

def int_to_image(X):

    reshaped_image = np.reshape(X, newshape=(48,96))
    reshaped_image = np.rot90(reshaped_image, 3)

    return reshaped_image
