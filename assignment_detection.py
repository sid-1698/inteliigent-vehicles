import cv2
from detector import Detector
from int_descriptor import IntDescriptor
from hog_descriptor import HogDescriptor
from cnn_descriptor import CnnDescriptor
from pathlib import Path
import numpy as np
import sklearn
from assignment_classification import assignment_classification
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

## Load the features (requires assignment 1 to be completed)
# if Path('assignment_classification.npy').is_file():
# 	data = np.load('assignment_features.npy', allow_pickle=True).item()
# else:
# 	data = assignment_classification()
data = np.load('assignment_features.npy', allow_pickle=True).item()
## Exercise 3.1
# Find the best performing classifier
# #YOUR_CODE_HERE

pca  = PCA(n_components=(35))
clf = SVC(kernel = "rbf", gamma = "scale", C=5)
model = Pipeline(steps=[("PCA", pca), ("Classifier", clf)])
data["features"]["hog"]["all"] = np.concatenate([data["features"]["hog"]["train"], data["features"]["hog"]["test"]])
data["y_all"] = np.concatenate([data["y_train"], data["y_test"]])
model.fit(data["features"]["hog"]["all"], data["y_all"])
print("Model Trained")
# Load video
cap = cv2.VideoCapture("pedestrian.mp4")

## Exercise 3.2
# Apply the detection algorithm by constructing a detector (see detector.py)
# Use the feature type and best performing parameters based on your findings in exercise 3.2
# Create a HogDescriptor or CnnDescriptor to pass to the Detector and run your classifier on the detection problem
# #YOUR_CODE_HERE
classifier = model
descriptor = HogDescriptor()
detector = Detector(classifier, descriptor)

# Apply the detection algorithm to the video and save the result to disk
detector.detectVideo(
	video_capture=cap,
	write = True,
	show_video=True,
	write_fps = 10
)

