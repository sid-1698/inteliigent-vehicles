from skimage.feature import hog
from int_to_image import int_to_image

def calculate_hog(X):
	features = []
	for image in X:
		image = int_to_image(image)
		feature = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
		features.append(feature)

	return features
