import skimage
import numpy
from image_processing import int_to_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import cv2

IMSIZE = (48, 96)

def image_to_convfeat(model, img):
    if img.shape != IMSIZE:
        img = cv2.resize(img, IMSIZE, interpolation=cv2.INTER_LINEAR) #width height

    img = skimage.color.gray2rgb(img)

    img_data = img

    img_data = numpy.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = model.predict(img_data)
    return feature.flatten()

def images_to_convfeats(model, X):
    conv_feats = []
    for x in X:
        image = int_to_image(x)
        conv_feat = image_to_convfeat(model, image)
        conv_feats.append(conv_feat)

    conv_feats = numpy.array(conv_feats)
    return conv_feats


