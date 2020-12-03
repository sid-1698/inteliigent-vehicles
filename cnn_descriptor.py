from descriptor import Descriptor
from calculate_cnn import image_to_convfeat
import cv2
from tensorflow.keras.applications.mobilenet import MobileNet

height = 96
width = 48
IMSIZE = (width, height)

class CnnDescriptor(Descriptor):

	def __init__(self, model = MobileNet(weights='imagenet', include_top=False)):
		self.model = model

	def get_feature_vector(self, image):
		"""Return the feature vector for an image."""
		if image.ndim == 1:
			image = int_to_image(image)

		if image.ndim == 3 and image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if image.shape[:2] != IMSIZE:
			image = cv2.resize(image, IMSIZE, interpolation=cv2.INTER_AREA)

		return image_to_convfeat(self.model, image)
