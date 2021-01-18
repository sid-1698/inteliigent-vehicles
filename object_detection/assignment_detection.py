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
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

## Load the features (requires assignment 1 to be completed)
if Path('assignment_classification.npy').is_file():
	data = np.load('assignment_features.npy', allow_pickle=True).item()
else:
	data = assignment_classification()

# Exercise 3.1
# Find the best performing classifier
pca  = PCA()
clf = SVC(kernel = "rbf", gamma = "scale")
pipe = Pipeline(steps=[("PCA", pca), ("Classifier", clf)])
param_grid = {"PCA__n_components" : [15,20,25,30,35,40,45],
              "Classifier__C" : [0.001, 0.005, 0.01, 0.02, 0.04, 0.1, 0.5, 1, 2, 5, 10]}
data["features"]["hog"]["all"] = np.concatenate([data["features"]["hog"]["train"], data["features"]["hog"]["test"]])
data["y_all"] = np.concatenate([data["y_train"], data["y_test"]])
X_train = data["features"]["hog"]["all"]
y_train = data["y_all"]

data["features"]["cnn"]["all"] = np.concatenate([data["features"]["cnn"]["train"], data["features"]["cnn"]["test"]])
data["y_all"] = np.concatenate([data["y_train"], data["y_test"]])


grid = GridSearchCV(estimator=pipe, param_grid=param_grid, refit=True, verbose=3)
grid.fit(data["features"]["hog"]["train"], data["y_train"])
model = grid.best_estimator_


_,(ax1,ax2)=plt.subplots(2,1,figsize=(15,15))
model.fit(data["features"]["hog"]["train"], data["y_train"])
plot_confusion_matrix(model, data["features"]["hog"]["test"], data["y_test"], cmap="Blues", ax=ax1)

plt.show()


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

grid = GridSearchCV(estimator=pipe, param_grid=param_grid, refit=True, verbose=3)
grid.fit(data["features"]["cnn"]["train"], data["y_train"])
model = grid.best_estimator_

_,(ax1,ax2)=plt.subplots(2,1,figsize=(15,15))
model.fit(data["features"]["cnn"]["train"], data["y_train"])
plot_confusion_matrix(model, data["features"]["cnn"]["test"], data["y_test"], cmap="Blues", ax=ax2)
plt.show()


model.fit(data["features"]["cnn"]["all"], data["y_all"])
print("Model Trained")
# Load video
cap = cv2.VideoCapture("pedestrian.mp4")

## Exercise 3.2
# Apply the detection algorithm by constructing a detector (see detector.py)
# Use the feature type and best performing parameters based on your findings in exercise 3.2
# Create a HogDescriptor or CnnDescriptor to pass to the Detector and run your classifier on the detection problem
# #YOUR_CODE_HERE
classifier = model
descriptor = CnnDescriptor()
detector = Detector(classifier, descriptor)

# Apply the detection algorithm to the video and save the result to disk
detector.detectVideo(
	video_capture=cap,
	write = True,
	show_video=False,
	write_fps = 10
)

