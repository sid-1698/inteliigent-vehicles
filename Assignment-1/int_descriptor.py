from descriptor import Descriptor
from int_to_image import int_to_image
from image_processing import image_to_int
import cv2

IMSIZE = (96,48)

class IntDescriptor(Descriptor):

	def __init__(self):
		return

	def get_feature_vector(self, image):
		"""Return the feature vector for an image."""
		if image.ndim == 1:
			image = int_to_image(image)

		if image.ndim == 3 and image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if image.shape[:2] != IMSIZE:
			image = cv2.resize(image, IMSIZE, interpolation=cv2.INTER_AREA)

		vector = image_to_int(image)
		return vector

