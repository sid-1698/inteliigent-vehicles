import cv2
import numpy as np
from skimage import feature
import image_processing as ip
from descriptor import Descriptor
from calculate_hog import calculate_hog
from image_processing import image_to_int

height = 96
width = 48
IMSIZE = (width, height)

class HogDescriptor(Descriptor):

	"""
	Class that combines feature descriptors into a single descriptor
	to produce a feature vector for an input image.
	"""

	def __init__(self):
		return

	def get_feature_vector(self, image):
		"""Return the feature vector for an image."""
		if image.ndim == 1:
			image = ip.int_to_image(image)

		if image.ndim == 3 and image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if image.shape[:2] != IMSIZE:
			image = cv2.resize(image, IMSIZE, interpolation=cv2.INTER_AREA)

		int_vector = image_to_int(image)
		int_vector = np.expand_dims(int_vector, axis=0)
		hog_vector = calculate_hog(int_vector)
		hog_vector = hog_vector[0]
		return hog_vector



