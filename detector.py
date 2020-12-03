from collections import deque
from datetime import datetime
import os
import pickle
import time
import cv2
import numpy as np
from scipy.ndimage.measurements import label
from descriptor import Descriptor
from image_processing import pyramid
from image_processing import sliding_window
import skimage

class Detector:

    """
    Class for finding objects in a video stream. Loads and utilizes a
    pretrained classifier.
    """

    def __init__(self, classifier, descriptor):

        self.classifier = classifier
        self.descriptor = descriptor
        self.windows = None

    def classify(self, feature_vectors):

        """
        Classify windows of an image as "positive" (containing the desired
        object) or "negative". Return a list of positively classified windows.
        """

        #predictions = self.classifier.predict(feature_vectors)
        scores = self.classifier.decision_function(feature_vectors)
        #print(np.mean(scores[predictions==1]))
        #print("----")
        #print(np.mean(scores[predictions==0]))

        #return [self.windows[ind] for ind in np.argwhere(predictions == 1)[:,0]] #return coordinates of windows where classifier said Positive
        return [self.windows[ind] for ind in np.argwhere(scores > 0.05)[:,0]] #return coordinates of windows where classifier said Positive

    def detectVideo(self, video_capture=None, num_frames=9, threshold=120,
            min_bbox=None, show_video=True, draw_heatmap=True,
            draw_heatmap_size=0.2, write=False, write_fps=24):

        """
        Find objects in each frame of a video stream by integrating bounding
        boxes over several frames to produce a heatmap of pixels with high
        prediction density, ignoring pixels below a threshold, and grouping
        the remaining pixels into objects. Draw boxes around detected objects.

        @param video_capture (VideoCapture): cv2.VideoCapture object.
        @param num_frames (int): Number of frames to sum over.
        @param threshold (int): Threshold for heatmap pixel values.
        @param min_bbox (int, int): Minimum (width, height) of a detection
            bounding box in pixels. Boxes smaller than this will not be drawn.
            Defaults to 2% of image size.
        @param show_video (bool): Display the video.
        @param draw_heatmap (bool): Display the heatmap in an inset in the
            upper left corner of the video.
        @param draw_heatmap_size (float): Size of the heatmap inset as a
            fraction between (0, 1) of the image size.
        @param write (bool): Write the resulting video, with detection
            bounding boxes and/or heatmap, to a video file.
        @param write_fps (num): Frames per second for the output video.
        """

        (winW, winH) = (48, 96) #TODO HARD CODED FOR NOW
        cap = video_capture
        if not cap.isOpened():
            raise RuntimeError("Error opening VideoCapture.")
        (grabbed, frame) = cap.read()
        (h, w) = frame.shape[:2]


        scale = 1.3


        if write:
            vidFilename = datetime.now().strftime("%Y%m%d%H%M") + ".avi"
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            writer = cv2.VideoWriter(vidFilename, fourcc, write_fps, (w, h))


        while True:
            (grabbed, frame) = cap.read()

            if not grabbed:
                break

            Feature_vectors = []
            self.windows = []
            scale_counter = -1
            for resized in pyramid(frame, scale=scale): #iterating scales
                scale_counter = scale_counter + 1
                scaler = scale ** scale_counter
                scaled_w = int(winW * scaler)
                scaled_h = int(winH * scaler)
                # loop over the sliding window for each layer of the pyramid
                for (x, y, window) in sliding_window(resized, stepSize=24, windowSize=(winW, winH)):
                    # if the window does not meet our desired window size, ignore it
                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue

                    if window.ndim == 3 and window.shape[2] == 3:
                        window = skimage.color.gray2rgb(window)[:,:,0]

                    Feature_vectors.append(self.descriptor.get_feature_vector(window)) #extract features from given window

                    x = int(x * scaler)
                    y = int(y * scaler)
                    self.windows.append([x,y,scaled_w,scaled_h]) #saving window coordinates

            detections = self.classify(np.array(Feature_vectors)) #check all the feature vectors at once
            for detection in detections:
                x,y,scaled_w,scaled_h = detection
                cv2.rectangle(frame, (x, y), (x + scaled_w, y + scaled_h), (0, 255, 0), 2)
            self.windows = []

            if write:
                writer.write(frame)

            if show_video:
                cv2.imshow("Detection", frame)
                cv2.waitKey(1)

        cap.release()

        if write:
            writer.release()
