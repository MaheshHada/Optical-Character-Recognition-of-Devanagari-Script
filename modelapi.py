import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Layer
from keras.layers import InputLayer, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.models import model_from_yaml
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import load_model

BASE = 2304
MAX_DIM = 218
IMG_SHAPE = (32, 32)
BATCH = 32
EPOCHS = 90
CLASSES = 128

def preproc_image(img):
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = img.astype(np.float32) / 255
    height, width = img.shape
    pad_top_size = (MAX_DIM - img.shape[0]) // 2
    pad_bottom_size = MAX_DIM - pad_top_size - height
    pad_left_size = (MAX_DIM - img.shape[1]) // 2
    pad_right_size = MAX_DIM - pad_left_size - width
    pad_top = np.zeros(shape=(pad_top_size, img.shape[1]))
    pad_bottom = np.zeros(shape=(pad_bottom_size, img.shape[1]))        
    img = np.vstack([pad_top, img, pad_bottom])        
    pad_left = np.zeros(shape=(img.shape[0], pad_left_size))
    pad_right = np.zeros(shape=(img.shape[0], pad_right_size))
    img = np.hstack([pad_left, img, pad_right])
    img = cv2.resize(img, IMG_SHAPE)
    return img

def predict(img):
	img = preproc_image(img)
	yaml_file = open('model.yaml', 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	loaded_model = model_from_yaml(loaded_model_yaml);
	loaded_model.load_weights("model.h5")
	loaded_model.compile(loss=categorical_crossentropy, 
	                      optimizer=RMSprop(), 
	                      metrics=[top_k_categorical_accuracy])
	probs = loaded_model.predict(img.reshape(1, IMG_SHAPE[0], IMG_SHAPE[1], 1)).reshape(CLASSES)
	top5probs = probs/np.max(probs)
	top5 = np.argsort(top5probs)[-5:][::-1]
	cumsum = [np.sum(top5probs[top5][i:]) for i in range(5)]
	cumsum /= np.max(cumsum)
	cumsum = cumsum[cumsum > 0.1]
	return list(top5[:len(cumsum)] + BASE)