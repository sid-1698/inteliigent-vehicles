import scipy.io
import numpy as np

## Load the data set
def load_data():
	## Load data
	ped_train_int, garb_train_int, ped_test_int, garb_test_int = mat_to_image_sets()
	ped_train_labels, garb_train_labels, ped_test_labels, garb_test_labels = mat_to_labels()

	# Concatenate the pedestrian and and non-pedestrian data
	X_train_int = np.concatenate((ped_train_int, garb_train_int))
	X_test_int = np.concatenate((ped_test_int, garb_test_int))
	y_train = np.hstack((ped_train_labels, garb_train_labels))
	y_test = np.hstack((ped_test_labels, garb_test_labels))

	# Normalize data
	return X_train_int, X_test_int, y_train, y_test


### load intensity images from mat file
def mat_to_image_sets(file_path = "lab1_data/new_data_int.mat"):
	data_int = scipy.io.loadmat(file_path)

	# intensity images
	ped_train_int = data_int['ped_train_int']
	garb_train_int = data_int['garb_train_int']
	ped_test_int = data_int['ped_test_int']
	garb_test_int = data_int['garb_test_int']

	return ped_train_int, garb_train_int, ped_test_int, garb_test_int


### load class labels from mat file
def mat_to_labels(file_path =  "lab1_data/new_data_labels.mat"):
	labels = scipy.io.loadmat(file_path)
	# labels
	ped_train_labels = labels['ped_train_labels'].flatten() > 0
	garb_train_labels = labels['garb_train_labels'].flatten() > 0
	ped_test_labels = labels['ped_test_labels'].flatten() > 0
	garb_test_labels = labels['garb_test_labels'].flatten() > 0

	return ped_train_labels, garb_train_labels, ped_test_labels, garb_test_labels


