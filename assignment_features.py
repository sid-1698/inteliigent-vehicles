from load_data import load_data
from int_to_image import int_to_image
import matplotlib.pyplot as plt
from visualize_hog import visualize_hog_orientations
from visualize_hog import visualize_hog_pixels_per_cell
from calculate_hog import calculate_hog
from tensorflow.keras.applications.mobilenet import MobileNet
from calculate_cnn import calculate_cnn
import numpy as np

def assignment_features(plot=False):
	# Load data
# # 	X_train_int: Training set images as 1-dimensional feature vectors
# # 	X_test_int: Test set images as 1-dimensional feature vectors
# # 	y_train: Labels of the training data
# 	y_test Labels of the test data
	X_train_int, X_test_int, y_train, y_test = load_data()
	## Exercise 1.1: Visualize the intensity data.
	# You will need to complete the code in
	#    int_to_image.py
	if plot:
		samples = X_train_int[::300,:]
		num_of_examples = samples.shape[0]
		fig = plt.figure(figsize=(5,2))
		for i in range(num_of_examples):
			ax = fig.add_subplot(2, 5, i+1)
			img = int_to_image(samples[i,:])
			ax.imshow((img), cmap='gray', vmin=0, vmax=255)
			plt.axis("off")
		plt.show()

	## Exercise 1.2: Visualize HOG for varying number of orientations
	# You will need to complete the code in
	#    visualize_hog.py
	if plot:
		samples = [500, 1000, 1500, 2000] # Choose your own samples here
		visualize_hog_orientations(X_train_int[samples])

	## Exercise 1.3: Visualize HOG for varying pixels-per-cell
	# You will need to complete the code in
	#    visualize_hog.py
	if plot:
		visualize_hog_pixels_per_cell(X_train_int[samples])

	## Exercise 1.4: Calculate the HOG features for the entire dataset
	# You will need to complete the code in
	#    calculate_hog.py
	X_train_hog = calculate_hog(X_train_int)
	X_test_hog = calculate_hog(X_test_int)

	# Exercise 1.5: Calculate the MobileNet features
	# You will need to complete the code in
	#    calculate_cnn.py

	model = MobileNet(weights='imagenet', include_top=False)
	X_train_cnn = calculate_cnn(model, X_train_int)
	X_test_cnn = calculate_cnn(model, X_test_int)


   # Store the features in a dictionary for use in the next assignments
	data = {}
	data["features"] = {}
	data["features"]["int"] = {}
	data["features"]["int"]["train"]  = X_train_int
	data["features"]["int"]["test"]  = X_test_int
	data["features"]["hog"] = {}
	data["features"]["hog"]["train"] = X_train_hog
	data["features"]["hog"]["test"]  = X_test_hog
	data["features"]["cnn"] = {}
	data["features"]["cnn"]["train"] = X_train_cnn
	data["features"]["cnn"]["test"]  = X_test_cnn
	data["y_train"] = y_train
	data["y_test"] = y_test

	# Save the results to disk to use in later exercises
	np.save('assignment_features.npy', data)

	return data

if __name__ == '__main__':
	assignment_features(plot=True)
