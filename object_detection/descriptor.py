## Adapter class so that the code of the assignment detector can work with sklearn-hog/opencv-hog, keras-mobilenet and intensity features.

class Descriptor:

	def __init__(self):
		raise NotImplementedError

	def get_feature_vector():
		raise NotImplementedError

