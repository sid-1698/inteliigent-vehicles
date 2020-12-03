#!/usr/bin/python3
import cv2
from detector import Detector
from int_descriptor import IntDescriptor
from hog_descriptor import HogDescriptor
from cnn_descriptor import CnnDescriptor
from pathlib import Path
import numpy as np
import sklearn
from assignment_classification import assignment_classification

## Load the features (requires assignment 1 to be completed)
if Path('assignment_classification.npy').is_file():
	data = np.load('assignment_classification.npy', allow_pickle=True).item()
else:
	data = assignment_classification()

## Exercise 3.1
# Find the best performing classifier
# #YOUR_CODE_HERE


# Load video
cap = cv2.VideoCapture("pedestrian.mp4")

## Exercise 3.2
# Apply the detection algorithm by constructing a detector (see detector.py)
# Use the feature type and best performing parameters based on your findings in exercise 3.2
# Create a HogDescriptor or CnnDescriptor to pass to the Detector and run your classifier on the detection problem
# #YOUR_CODE_HERE

# Apply the detection algorithm to the video and save the result to disk
detector.detectVideo(
	video_capture=cap,
	write = True,
	show_video=False,
	write_fps = 10
)

