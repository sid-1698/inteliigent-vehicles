import matplotlib.pyplot as plt
import cv2
# Scikit-Learn ≥0.20 is required
import sklearn

# Common imports
import numpy as np
import os



NUM_ORIENTATIONS = 8
PIXELS_PER_CELL = 8


def image_to_int(X):
    return np.transpose(X).flatten()

def norm_images(X_train_int, X_test_int):
    min_int = X_train_int.min() #TODO Normalize based on training set or entire set
    X_train_int -= min_int
    X_test_int -= min_int

    max_int = X_train_int.max() #TODO Normalize based on training set or entire set
    X_train_int = X_train_int / max_int * 255.0
    X_test_int = X_test_int / max_int * 255.0
    return X_train_int, X_test_int


def pyramid(image, scale=1.1, minSize=(200, 200)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = image_resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
    y_min = 0.3
    y_max = 0.6
    x_min = 0
    x_max = 1

    y_min = int(y_min * image.shape[0])
    y_max = int(y_max * image.shape[0])
    x_min = int(x_min * image.shape[1])
    x_max = int(x_max * image.shape[1])

    for y in range(y_min, y_max, stepSize):
	    for x in range(x_min, x_max, stepSize):
		    # yield the current window
		    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def sliding_window_demo(image_path = "lab1_data/test_sequence/im_0001.png"):
    if not os.path.isfile(image_path):
    #ignore if no such file is present.
        print("not found")
        return

    (winW, winH) = (48, 96)
    img = cv2.imread(image_path)
    figure = plt.figure()
    
    for (x, y, window) in sliding_window(img, stepSize=60, windowSize=(winW, winH)):
        cv2.rectangle(img, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def increase_intensity(original, increase):
    image = original.copy()
    to_add = increase
    image[image > (255 - to_add)] = 255 - to_add
    image += to_add
    return image


    #import image_processing.sliding_window_demo()
