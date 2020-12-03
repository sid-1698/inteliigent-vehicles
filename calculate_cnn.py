import skimage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from int_to_image import int_to_image

def image_to_convfeat(model, image):
	color = skimage.color.gray2rgb(image)
	img_data = np.expand_dims(color, axis=0)
	processed = preprocess_input(img_data)
	feature = model.predict(processed)
	return feature.flatten()

def calculate_cnn(model, X):
	
    features = []
    for image in X:
        image = int_to_image(image)
        features.append(image_to_convfeat(model, image))
    return features
