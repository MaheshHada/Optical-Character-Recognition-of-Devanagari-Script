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
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
# %matplotlib inline

BASE = 2304
MAX_DIM = 218
IMG_SHAPE = (32, 32)
BATCH = 32
EPOCHS = 75
CLASSES = 128

def show_image(img, name=''):
    plt.axis('off')
    plt.imshow(img)
    plt.title(name)

def load_data():
    images = []
    labels = []
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    for f in glob.glob('q_train_images/*.png'):
        img = cv2.imread(f, 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = img.astype(np.float32) / 255
        images.append(img)
        lab = f.split('_')[5:]
        lab[-1] = lab[-1].split('.')[0]
#         print(lab)
        lab = list(map(int, lab))
        lab = [x - BASE for x in lab]
        labels.append(lab)
    return images, labels


images, labels = load_data()
print(len(images))

def resize_images(images):
    resized_images = []
    for i in range(len(images)):
        img = images[i]
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
        resized_images.append(img)
    return resized_images

images = resize_images(images)
print(len(images))
print(np.unique(images[0]))


lab = np.zeros(shape=(images.shape[0], 128))
for i, lb in enumerate(labels):
    for j in lb:
        lab[i, j] = 1.

images = np.expand_dims(images, axis=3)

class Top5Round(Layer):
    
    def __init__(self, **kwargs):
        super(Top5Round, self).__init__(**kwargs)
    
    def get_output(self, train=False):
        X = self.get_input(train)
        sorted_idx = K.argsort(X)
        X[sorted_idx[-5:]] = K.round(X[sorted_idx[-5:]])
        X[sorted_idx[:-5]] = 0
        return X
    
    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(Top5Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

model = Sequential()
model.add(InputLayer(input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 1)))

model.add(Conv2D(filters=32, 
                 kernel_size=5, 
                 padding='same', 
                 activation='relu'))
model.add(Conv2D(filters=32, 
                 kernel_size=5, 
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,  
                 kernel_size=3, 
                 padding='same', 
                 activation='relu'))
model.add(Conv2D(filters=64, 
                 kernel_size=3, 
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2, 
                       strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,  
                 kernel_size=3, 
                 padding='same', 
                 activation='relu'))
model.add(Conv2D(filters=64, 
                 kernel_size=3, 
                 padding='same', 
                 activation='relu'))
model.add(Conv2D(filters=64, 
                 kernel_size=3, 
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2, 
                       strides=2))
model.add(Dropout(0.25))
          
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(CLASSES, activation='sigmoid'))
model.add(Top5Round())


model.compile(loss=categorical_crossentropy, 
              optimizer=RMSprop(), 
              metrics=[top_k_categorical_accuracy])

train_datagen =  ImageDataGenerator(rotation_range=5, 
                                width_shift_range=0.1, 
                                height_shift_range=0.1, 
                                shear_range=0.1, 
                                zoom_range=0.2)

train_gen = train_datagen.flow(Xtr, ytr, batch_size=BATCH)

model.fit_generator(train_gen, 
                    steps_per_epoch=Xtr.shape[0]//BATCH,  
                    epochs=EPOCHS, 
                    verbose=1,  
                    validation_data=(Xval, yval))

probs = model.predict(img.reshape(1, IMG_SHAPE[0], IMG_SHAPE[1], 1)).reshape(CLASSES)

top10probs = probs / np.max(probs)
top10 = np.argsort(top10probs)[-10:][::-1]
top10hex = [hex(BASE + x) for x in top10]
list(zip(top10, top10hex, top10probs[top10]))