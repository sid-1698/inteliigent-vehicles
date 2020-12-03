from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure
from int_to_image import int_to_image

def visualize_hog_orientations(X_train_int):

	for image in X_train_int:
		image = int_to_image(image)
		_, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('Input image')	

		# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

		ax2.axis('off')
		ax2.imshow(hog_image, cmap=plt.cm.gray)
		ax2.set_title('Histogram of Oriented Gradients')
		plt.show()
	
def visualize_hog_pixels_per_cell(X_train_int):
	
	for image in X_train_int:
		image = int_to_image(image)
		_, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2), visualize=True)
		_, hog_image_2 = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=True)
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('Input image')	

		# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
		# hog_image_rescaled_2 = exposure.rescale_intensity(hog_image_2, in_range=(0, 10))

		ax2.axis('off')
		ax2.imshow(hog_image, cmap=plt.cm.gray)
		ax2.set_title('HOG (4,4)')

		ax3.axis('off')
		ax3.imshow(hog_image_2, cmap=plt.cm.gray)
		ax3.set_title('HOG (16,16)')
		plt.show()

